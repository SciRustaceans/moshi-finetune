"""KV-prefill validation v2: teacher-forced priming + greedy text + full-frame debug.

PRIME by teacher-forcing the EXACT training prefix: row0=name text token, audio rows 1-8
written as ZEROS (matches -1 zero-embed in training), WITHOUT running depformer -> cache
bit-matches training. Greedy text option. --debug decodes the WHOLE text row (prime+call)
to see if Hi {name} is generated anywhere.
"""
import argparse, json, os, re
import numpy as np
import sphn
import torch
import whisper
from moshi.models.lm import LMGen
from moshi.utils.sampling import sample_token
from moshi.models import loaders

SAMPLE_RATE = 24000
ASR_SR = 16000
P = 12
TEXT_PADDING_ID = 3


class KVPrimeLMGen(LMGen):
    def set_prefix(self, prefix_text_ids):
        self._prefix = list(prefix_text_ids)
        self._idx = 0

    @torch.no_grad()
    def step(self, input_tokens):
        state = self._streaming_state
        lm_model = self.lm_model
        B, Ki, S = input_tokens.shape
        CT = state.cache.shape[2]
        for q_other in range(input_tokens.shape[1]):
            k = lm_model.dep_q + 1 + q_other
            delay = lm_model.delays[k]
            wp = (state.offset + delay) % CT
            state.cache[:, k, wp:wp + 1] = input_tokens[:, q_other]
        position = state.offset % CT
        for k, delay in enumerate(lm_model.delays):
            if state.offset <= delay:
                state.cache[:, k, position] = state.initial[:, k, 0]
        input_ = state.cache[:, :, position:position + 1]
        transformer_out, text_logits = state.graphed_main(input_, state.condition_sum)
        text_token = sample_token(text_logits.float(), self.use_sampling, self.temp_text, self.top_k_text)
        text_token = text_token[:, 0, 0]
        priming = self._idx < len(self._prefix)
        if priming:
            forced = self._prefix[self._idx]
            if forced >= 0:
                text_token = torch.full_like(text_token, forced)
        self._idx += 1
        if priming:
            audio_tokens = torch.full((B, lm_model.dep_q), -1, dtype=torch.long, device=text_token.device)  # FAITHFUL -1 zero_token_id prime (was torch.zeros=token0)
        else:
            audio_tokens = state.graphed_depth(text_token, transformer_out)
        state.offset += 1
        position = state.offset % CT
        state.cache[:, 0, position] = text_token
        state.cache[:, 1:lm_model.dep_q + 1, position] = audio_tokens
        if state.offset <= self.max_delay:
            return None
        gen_delays_cuda = self.delays_cuda[: lm_model.dep_q + 1]
        index = (((state.offset - self.max_delay + gen_delays_cuda) % CT).view(1, -1, 1).expand(B, -1, 1))
        return state.cache.gather(dim=2, index=index)


def load(lora_path, hf_repo, device, dtype, lora_rank=128):
    ci = loaders.CheckpointInfo.from_hf_repo(hf_repo, lora_weights=lora_path)
    overrides = {"lora": True, "lora_rank": lora_rank, "lora_scaling": 2.0}
    model = ci.get_moshi(device=device, dtype=dtype, lm_kwargs_overrides=overrides)
    model.eval()
    mimi = ci.get_mimi(device=device); mimi.eval()
    spm = ci.get_text_tokenizer()
    return model, mimi, spm


def prefix_ids(spm, name):
    ids = spm.encode(name)[:P]
    row = [TEXT_PADDING_ID] * P
    for i, t in enumerate(ids):
        row[i] = t
    return row


def name_hit(transcript, name):
    t = re.sub(r"[^a-z]", "", transcript.lower())
    n = re.sub(r"[^a-z]", "", name.lower())
    if n in t:
        return True
    return len(n) >= 4 and n[:len(n) - 1] in t


def asr_array(pcm):
    a16 = sphn.resample(pcm[None, :], src_sample_rate=SAMPLE_RATE, dst_sample_rate=ASR_SR)[0]
    return a16.astype(np.float32)


def gen_once(model, mimi, spm, name, device, call_frames, temp, temp_text, seed, debug=False):
    torch.manual_seed(seed)
    pref = prefix_ids(spm, name)
    lm_gen = KVPrimeLMGen(model, use_sampling=(temp_text > 0), temp=temp, temp_text=max(temp_text, 1e-4))
    lm_gen.set_prefix(pref)
    minus1 = torch.full((1, 8, 1), -1, device=device, dtype=torch.long)
    with torch.no_grad():
        sil = mimi.encode(torch.zeros(1, 1, 1920 * 125, device=device))
    coll_call, coll_all = [], []
    with torch.no_grad(), mimi.streaming(1), lm_gen.streaming(1):
        for fi in range(P):
            o = lm_gen.step(minus1)
            if o is not None and debug:
                coll_all.append(o)
        for fi in range(call_frames):
            j = fi % sil.shape[2]
            o = lm_gen.step(sil[:, :, j:j + 1])
            if o is not None:
                coll_call.append(o)
                if debug:
                    coll_all.append(o)
    if not coll_call:
        return None, "", ""
    tok = torch.cat(coll_call, dim=2)
    emit = spm.decode([x for x in tok[0, 0].cpu().tolist() if x > 3])
    allemit = ""
    if debug and coll_all:
        allt = torch.cat(coll_all, dim=2)
        allemit = spm.decode([x for x in allt[0, 0].cpu().tolist() if x > 3])
    with torch.no_grad():
        n_neg = int((tok[:, 1:] < 0).sum().item())
        if n_neg: print("  [neg-audio-tokens before clamp]", n_neg)
        audio = mimi.decode(tok[:, 1:].clamp(min=0))  # clamp -1 prime leak (delay boundary) to valid code
    pcm = audio[0, 0].float().cpu().numpy().astype(np.float32)
    return pcm, emit, allemit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", default="runs/kv_demo")
    ap.add_argument("--seen", nargs="+", default=["Akane", "Astraya", "Aaliha"])
    ap.add_argument("--unseen", nargs="+", default=["Maria", "Haruko", "Bennito", "Knovah", "Veneita"])
    ap.add_argument("--call-frames", type=int, default=70)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--temp-text", type=float, default=0.0)
    ap.add_argument("--debug", action="store_true")
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
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for gname, names in [("SEEN", args.seen), ("UNSEEN", args.unseen)]:
        for name in names:
            hits = 0; rows = []; best = None
            for s in range(args.seeds):
                pcm, emit, allemit = gen_once(model, mimi, spm, name, device,
                                              args.call_frames, args.temp, args.temp_text,
                                              1000 + s, debug=args.debug)
                if pcm is None:
                    rows.append("<empty>"); continue
                tr = asr.transcribe(asr_array(pcm), language="en", fp16=True)["text"].strip()
                hit = name_hit(tr, name)
                hits += int(hit)
                line = ("[HIT] " if hit else "[miss] ") + "asr=" + repr(tr) + " | emit=" + repr(emit)
                if args.debug:
                    line += " | ALL=" + repr(allemit)
                rows.append(line)
                if best is None or (hit and not best[2]):
                    best = (pcm, tr, hit)
            if best is not None:
                sphn.write_wav(os.path.join(args.out, name + ".wav"), best[0], SAMPLE_RATE)
            rate = hits / args.seeds
            summary[name] = {"group": gname, "clean_rate": rate, "hits": hits, "seeds": args.seeds}
            print("=== %s [%s] clean-rate=%d/%d=%.2f ===" % (name, gname, hits, args.seeds, rate))
            for r in rows:
                print("   ", r)
    print("SUMMARY " + json.dumps(summary))
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
