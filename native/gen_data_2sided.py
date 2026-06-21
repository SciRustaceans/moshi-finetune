import argparse, json, os, random
import numpy as np
import sphn
import torch
import torchaudio.functional as AF
import whisper_timestamped as whisper
from moshi.models.tts import TTSModel, DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO
from moshi.models.loaders import CheckpointInfo

SAMPLE_RATE = 24000
WHISPER_SR = 16000

SCRIPTS = [
    [('moshi', 'Hi {name}, how are you today?'), ('user', 'I am doing well, thank you. And you?'), ('moshi', 'I am great, thanks for asking. How can I help you today?'), ('user', 'I had a quick question about my account.'), ('moshi', 'Of course, I would be happy to help with that.')],
    [('moshi', 'Hello {name}, how is it going?'), ('user', 'Pretty good, thanks. How about you?'), ('moshi', 'Doing great, thank you. What can I do for you?'), ('user', 'I wanted to check on my recent order.'), ('moshi', 'Sure, let me take a look at that for you.')],
    [('moshi', 'Hey {name}, how are you doing?'), ('user', 'I am good, thanks for asking.'), ('moshi', 'Glad to hear it. How can I help you today?'), ('user', 'Could you tell me about my billing?'), ('moshi', 'Absolutely, I can walk you through that.')],
    [('moshi', 'Good morning {name}, how are you?'), ('user', 'Doing well, thank you. And yourself?'), ('moshi', 'I am doing great, thanks. What brings you in today?'), ('user', 'I need some help resetting my password.'), ('moshi', 'No problem at all, I can help with that.')],
    [('moshi', 'Hey {name}, how is your day going?'), ('user', 'It is going well, thanks. How is yours?'), ('moshi', 'Really good, thank you for asking. How can I assist you?'), ('user', 'I have a question about my subscription.'), ('moshi', 'I am happy to help with your subscription.')],
    [('moshi', 'Hi {name}, how are you today?'), ('user', 'I am well, thank you so much.'), ('moshi', 'Wonderful. Is there something I can help you with?'), ('user', 'Yes, I would like to update my address.'), ('moshi', 'Of course, I can take care of that for you.')],
    [('moshi', 'Hello {name}, nice to talk with you. How are you?'), ('user', 'I am doing fine, thank you.'), ('moshi', 'Great to hear. What can I do for you today?'), ('user', 'I am having trouble logging in.'), ('moshi', 'I am sorry to hear that. Let us get it sorted.')],
    [('moshi', 'Good afternoon {name}, how are you doing?'), ('user', 'Quite well, thanks for asking. And you?'), ('moshi', 'I am doing very well, thank you. How may I help?'), ('user', 'I wanted to ask about your services.'), ('moshi', 'I would be glad to tell you more about them.')],
    [('moshi', 'Hi {name}, thanks for calling. How are you?'), ('user', 'I am great, thank you. How are you?'), ('moshi', 'I am doing well, thank you. What can I help you with?'), ('user', 'I would like to change my appointment time.'), ('moshi', 'Certainly, let me help you with that.')],
    [('moshi', 'Hello {name}, how are you this morning?'), ('user', 'Not bad, thank you. How about yourself?'), ('moshi', 'I am doing nicely, thanks. How can I help today?'), ('user', 'I have a question about my recent invoice.'), ('moshi', 'I would be happy to look into that for you.')],
    [('moshi', 'Hey {name}, good to hear from you. How are you?'), ('user', 'I am doing okay, thanks. Can you help me with something?'), ('moshi', 'Of course. What can I do for you today?'), ('user', 'I need to update the email on my account.'), ('moshi', 'No problem, I can take care of that right away.')],
    [('moshi', 'Hi {name}, welcome back. How are you doing?'), ('user', 'I am well, thank you. I had a question for you.'), ('moshi', 'Glad to hear it. Please go ahead, I am listening.'), ('user', 'Could you check the status of my request?'), ('moshi', 'Absolutely, let me check on that for you now.')],
]


def synth_turn(tts, text, voice_path, cfg_cond, cfg_is_no_prefix, cfg_is_no_text):
    entries = [tts.prepare_script([text], padding_between=1)]
    voices = [voice_path] if tts.multi_speaker else []
    attr = [tts.make_condition_attributes(voices, cfg_cond)]
    prefixes = None if tts.multi_speaker else [tts.get_prefix(voice_path)]
    result = tts.generate(entries, attr, prefixes=prefixes,
                          cfg_is_no_prefix=cfg_is_no_prefix, cfg_is_no_text=cfg_is_no_text)
    frames = []
    with torch.no_grad(), tts.mimi.streaming(1):
        for frame in result.frames[tts.delay_steps:]:
            frames.append(tts.mimi.decode(frame[:, 1:]))
    wav = torch.cat(frames, dim=-1)
    end_step = result.end_steps[0]
    if end_step is None:
        wav_len = wav.shape[-1]
    else:
        wav_len = int(tts.mimi.sample_rate * (end_step + tts.final_padding) / tts.mimi.frame_rate)
    return wav[0, 0, :wav_len].clamp(-1, 1).float().cpu().numpy()


def words_for_turn(w_model, mono_wav, start_sec, language="en"):
    x = torch.from_numpy(mono_wav)[None]
    x16 = AF.resample(x, SAMPLE_RATE, WHISPER_SR).numpy()[0]
    out = whisper.transcribe(w_model, x16, language=language, best_of=5, beam_size=5,
                             temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), verbose=None)
    words = []
    for seg in out["segments"]:
        for wd in seg.get("words", []):
            words.append({"w": wd["text"], "start": wd["start"] + start_sec, "end": wd["end"] + start_sec})
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/wav_2s")
    ap.add_argument("--names", default="data/greet/names.txt")
    ap.add_argument("--heldout", default="data/greet/heldout_names.txt")
    ap.add_argument("--prior", default="data/greet/prior_names.txt")
    ap.add_argument("--moshi-voice", default="vctk/p225_023.wav")
    ap.add_argument("--user-voice", default="vctk/p226_023.wav")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--gap", type=float, default=0.4)
    ap.add_argument("--tail", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--cfg-coef", type=float, default=2.0)
    ap.add_argument("--whisper-model", default="medium")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    def load_set(p):
        return set(l.strip() for l in open(p) if l.strip()) if os.path.exists(p) else set()
    heldout = load_set(args.heldout)
    prior = load_set(args.prior)
    all_names = [l.strip() for l in open(args.names) if l.strip()]
    pool = [n for n in all_names if n not in heldout and n not in prior]
    seen = set(); uniq = []
    for n in pool:
        if n not in seen:
            seen.add(n); uniq.append(n)
    assert len(uniq) >= args.n, "not enough free names: %d < %d" % (len(uniq), args.n)
    chosen = rng.sample(uniq, args.n)
    assert set(chosen).isdisjoint(heldout) and set(chosen).isdisjoint(prior)
    tmpl_rng = random.Random(args.seed + 1)
    tmpl_idx = [tmpl_rng.randrange(len(SCRIPTS)) for _ in range(args.n)]
    my_idx = [i for i in range(args.n) if i % args.shards == args.shard]

    dtype = torch.bfloat16
    ci = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
    tts = TTSModel.from_checkpoint_info(
        ci, voice_repo=DEFAULT_DSM_TTS_VOICE_REPO, n_q=32, temp=args.temp,
        cfg_coef=args.cfg_coef, device=args.device, dtype=dtype)
    if tts.valid_cfg_conditionings:
        cfg_cond = tts.cfg_coef
        tts.cfg_coef = 1.0
        cfg_is_no_text = False
        cfg_is_no_prefix = False
    else:
        cfg_cond = None
        cfg_is_no_text = True
        cfg_is_no_prefix = True
    moshi_vp = tts.get_voice_path(args.moshi_voice) if tts.multi_speaker else args.moshi_voice
    user_vp = tts.get_voice_path(args.user_voice) if tts.multi_speaker else args.user_voice
    w_model = whisper.load_model(args.whisper_model, device=args.device)
    print("[shard %d/%d] %d clips | Moshi(L)=%s User(R)=%s" % (
        args.shard, args.shards, len(my_idx), args.moshi_voice, args.user_voice), flush=True)

    done = 0
    for gi in my_idx:
        name = chosen[gi]
        script = [(role, txt.format(name=name)) for (role, txt) in SCRIPTS[tmpl_idx[gi]]]
        cid = "twosided_%04d" % gi
        outwav = os.path.join(args.out, cid + ".wav")
        if os.path.exists(outwav) and os.path.exists(os.path.join(args.out, cid + ".json")):
            done += 1; continue
        turns = []
        cursor = 0.0
        for (role, text) in script:
            voice = moshi_vp if role == "moshi" else user_vp
            wav = synth_turn(tts, text, voice, cfg_cond, cfg_is_no_prefix, cfg_is_no_text)
            dur = wav.shape[0] / SAMPLE_RATE
            turns.append({"role": role, "text": text, "wav": wav, "start": cursor, "end": cursor + dur})
            cursor += dur + args.gap
        last_end = max(t["end"] for t in turns)
        total = int(round((last_end + args.tail) * SAMPLE_RATE))
        stereo = np.zeros((2, total), dtype=np.float32)
        for t in turns:
            ch = 0 if t["role"] == "moshi" else 1
            s = int(round(t["start"] * SAMPLE_RATE))
            seg = t["wav"]
            n = min(seg.shape[0], total - s)
            stereo[ch, s:s + n] = seg[:n]
        sphn.write_wav(outwav, stereo, SAMPLE_RATE)

        out_turns = []
        for t in turns:
            spk = t["role"]
            words = words_for_turn(w_model, t["wav"], t["start"]) if spk == "moshi" else []
            out_turns.append({"speaker": spk, "text": t["text"],
                              "start": round(t["start"], 3), "end": round(t["end"], 3),
                              "words": words})
        schedule = [{"role": t["speaker"], "start": t["start"], "end": t["end"], "text": t["text"]} for t in out_turns]
        with open(os.path.join(args.out, cid + ".json"), "w") as fh:
            json.dump({"name": name, "sr": SAMPLE_RATE, "turns": out_turns, "schedule": schedule}, fh, ensure_ascii=False)
        with open(os.path.join(args.out, cid + ".txt"), "w") as fh:
            fh.write("name\t%s\n" % name)
        done += 1
        mw = " ".join(w["w"].strip() for t in out_turns if t["speaker"] == "moshi" for w in t["words"])
        print("[shard %d] %s name=%r dur=%.2fs turns=%d moshi=%r" % (
            args.shard, cid, name, total / SAMPLE_RATE, len(turns), mw[:70]), flush=True)
    print("[shard %d] DONE: %d clips -> %s" % (args.shard, done, args.out), flush=True)


if __name__ == "__main__":
    main()
