"""Conversation validation: GREET-by-name (KV-prime) AND continue a conversation.

Per name: (a) prime the NAME (teacher-force exact training prefix), (b) free-gen the greeting
(silence on user stream), (c) INJECT a TTS user turn on rows 9-16 (8 mimi codebooks encoded
with the .venv 0.2.4a1 mimi -- same codec as training), (d) free-gen the response (silence).
Output is segmented by frame window: GREETING (frames before injection) vs RESPONSE (frames
after injection end), decoded SEPARATELY so we can report greeting-fired? and response-text
independently. The user-turn wav is pre-synthesized once (TTS/p226) and reused.
"""
import argparse, json, os, re
import numpy as np
import sphn
import torch
import whisper
from moshi.models import loaders
from native.generate_kv_shift import KVPrimeLMGen, load, prefix_ids

SAMPLE_RATE = 24000
ASR_SR = 16000
P = 12
TEXT_PADDING_ID = 3


def asr_array(pcm):
    a16 = sphn.resample(pcm[None, :], src_sample_rate=SAMPLE_RATE, dst_sample_rate=ASR_SR)[0]
    return a16.astype(np.float32)


def name_hit(transcript, name):
    t = re.sub(r"[^a-z]", "", transcript.lower())
    n = re.sub(r"[^a-z]", "", name.lower())
    if n in t:
        return True
    return len(n) >= 4 and n[:len(n) - 1] in t


def rms(pcm):
    return float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) if pcm.size else 0.0


def encode_user_turn(mimi, wav_path, device):
    wav, sr = sphn.read(wav_path)
    if wav.ndim > 1:
        wav = wav[0]
    if sr != SAMPLE_RATE:
        wav = sphn.resample(wav[None], src_sample_rate=sr, dst_sample_rate=SAMPLE_RATE)[0]
    x = torch.from_numpy(wav).to(device).float()[None, None]
    with torch.no_grad():
        codes = mimi.encode(x)[0]
    return codes


def gen_conversation(model, mimi, spm, name, device, user_codes,
                     greet_frames, gap_frames, resp_frames, temp, temp_text, seed):
    torch.manual_seed(seed)
    pref = prefix_ids(spm, name)
    lm_gen = KVPrimeLMGen(model, use_sampling=(temp_text > 0), temp=temp, temp_text=max(temp_text, 1e-4))
    lm_gen.set_prefix(pref)
    minus1 = torch.full((1, 8, 1), -1, device=device, dtype=torch.long)
    with torch.no_grad():
        sil = mimi.encode(torch.zeros(1, 1, 1920 * 80, device=device))
    Tu = user_codes.shape[1]
    user_codes = user_codes.to(device).long()[None]
    greet_out, resp_out = [], []
    with torch.no_grad(), mimi.streaming(1), lm_gen.streaming(1):
        for _ in range(P):
            lm_gen.step(minus1)
        for fi in range(greet_frames):
            j = fi % sil.shape[2]
            o = lm_gen.step(sil[:, :, j:j + 1])
            if o is not None:
                greet_out.append(o)
        for fi in range(gap_frames):
            j = fi % sil.shape[2]
            lm_gen.step(sil[:, :, j:j + 1])
        for fi in range(Tu):
            lm_gen.step(user_codes[:, :, fi:fi + 1])
        for fi in range(resp_frames):
            j = fi % sil.shape[2]
            o = lm_gen.step(sil[:, :, j:j + 1])
            if o is not None:
                resp_out.append(o)

    def decode(frames):
        if not frames:
            return chr(32).strip(), np.zeros(0, np.float32)
        tok = torch.cat(frames, dim=2)
        txt = spm.decode([x for x in tok[0, 0].cpu().tolist() if x > 3])
        with torch.no_grad():
            audio = mimi.decode(tok[:, 1:].clamp(min=0))
        return txt, audio[0, 0].float().cpu().numpy().astype(np.float32)
    g_txt, g_pcm = decode(greet_out)
    r_txt, r_pcm = decode(resp_out)
    return {'greet_text': g_txt, 'greet_pcm': g_pcm, 'resp_text': r_txt, 'resp_pcm': r_pcm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--user-wav", required=True, help="pre-synthesized user-turn wav (p226)")
    ap.add_argument("--out", default="runs/conv_demo")
    ap.add_argument("--seen", nargs="+", required=True)
    ap.add_argument("--unseen", nargs="+", required=True)
    ap.add_argument("--greet-frames", type=int, default=45)
    ap.add_argument("--gap-frames", type=int, default=5)
    ap.add_argument("--resp-frames", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--temp-text", type=float, default=0.0)
    ap.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-bf16")
    args = ap.parse_args()
    device = "cuda"; dtype = torch.bfloat16
    lp = args.lora
    if os.path.isdir(lp):
        lp = os.path.join(lp, "lora.safetensors")
    rank = 128
    cfgp = os.path.join(os.path.dirname(lp), "config.json")
    if os.path.exists(cfgp):
        rank = int(json.load(open(cfgp)).get("lora_rank", 128))
    print("[load] lora_rank=%d from %s" % (rank, cfgp))
    model, mimi, spm = load(lp, args.hf_repo, device, dtype, lora_rank=rank)
    asr = whisper.load_model("base.en", device=device)
    user_codes = encode_user_turn(mimi, args.user_wav, device)
    print("[user-turn] %s -> codes %s" % (args.user_wav, tuple(user_codes.shape)))
    os.makedirs(args.out, exist_ok=True)

    summary = {}
    for gname, names in [("SEEN", args.seen), ("UNSEEN", args.unseen)]:
        for name in names:
            best = None
            greet_hits = 0; resp_count = 0
            for s in range(args.seeds):
                r = gen_conversation(model, mimi, spm, name, device, user_codes,
                                     args.greet_frames, args.gap_frames, args.resp_frames,
                                     args.temp, args.temp_text, 1000 + s)
                g_tr = asr.transcribe(asr_array(r["greet_pcm"]), language="en", fp16=True)["text"].strip() if r["greet_pcm"].size else ""
                r_tr = asr.transcribe(asr_array(r["resp_pcm"]), language="en", fp16=True)["text"].strip() if r["resp_pcm"].size else ""
                g_hit = name_hit(g_tr, name)
                r_rms = rms(r["resp_pcm"])
                responded = (r_rms > 0.003) and (len(r_tr) >= 2 or len(r["resp_text"].strip()) >= 2)
                greet_hits += int(g_hit); resp_count += int(responded)
                print("  [%s s%d] greet_hit=%s greet_asr=%r | resp_rms=%.4f responded=%s resp_emit=%r resp_asr=%r" % (
                      name, s, g_hit, g_tr, r_rms, responded, r["resp_text"], r_tr))
                if best is None or (g_hit and responded and not (best[1] and best[2])):
                    best = (r, g_hit, responded, g_tr, r_tr, r_rms)
            if best is not None:
                rr = best[0]
                full = np.concatenate([rr["greet_pcm"], np.zeros(int(0.4*SAMPLE_RATE), np.float32), rr["resp_pcm"]])
                sphn.write_wav(os.path.join(args.out, name + "_full.wav"), full, SAMPLE_RATE)
                sphn.write_wav(os.path.join(args.out, name + "_greet.wav"), rr["greet_pcm"], SAMPLE_RATE)
                if rr["resp_pcm"].size:
                    sphn.write_wav(os.path.join(args.out, name + "_resp.wav"), rr["resp_pcm"], SAMPLE_RATE)
            summary[name] = {"group": gname, "greet_rate": greet_hits/args.seeds,
                             "resp_rate": resp_count/args.seeds,
                             "best_greet_asr": best[3] if best else "",
                             "best_resp_asr": best[4] if best else "",
                             "best_resp_emit": best[0]["resp_text"] if best else ""}
            print("=== %s [%s] greet=%d/%d resp=%d/%d | best_resp_asr=%r emit=%r ===" % (
                  name, gname, greet_hits, args.seeds, resp_count, args.seeds,
                  summary[name]["best_resp_asr"], summary[name]["best_resp_emit"]))
    print("SUMMARY " + json.dumps(summary))
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
