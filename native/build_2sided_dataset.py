"""Build STEREO two-sided dataset samples [17, P+T] for Moshi conversation finetuning.
row0=Moshi monologue (greeting+response words at frames; pad(3) during user turn).
rows1-8=stereo rows0-7=LEFT=Moshi audio. rows9-16=stereo rows8-15=RIGHT=User audio.
PREFIX P=12: row0=name SP tokens+pad(3); rows1-16=-1. full=[17,P+T] int32, prefix_len=12.
"""
import argparse, glob, json, os
import numpy as np
import torch
from moshi.models import loaders
from finetune.data.interleaver import Interleaver

TEXT_PADDING_ID = 3
ZERO_TOKEN_ID = -1
EOT_PADDING_ID = 0
SPK_MAIN = "SPEAKER_MAIN"


def build_row0(interleaver, turns, T, frame_rate):
    """row0 [T] int32: Moshi-turn words at their frames; pad(3) during user turn."""
    aligns = []
    for t in turns:
        if t["speaker"] != "moshi":
            continue
        for w in t["words"]:
            aligns.append((w["w"], (float(w["start"]), float(w["end"])), SPK_MAIN))
    seg_dur = T / frame_rate
    row0 = interleaver.prepare_item(aligns, seg_dur, main_speaker=SPK_MAIN)
    row0 = row0[0, 0].cpu().numpy().astype(np.int32)
    if row0.shape[0] < T:
        row0 = np.concatenate([row0, np.full(T - row0.shape[0], TEXT_PADDING_ID, np.int32)])
    else:
        row0 = row0[:T]
    return row0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="data/audio_codes_2s")
    ap.add_argument("--sched", default="data/wav_2s")
    ap.add_argument("--out", default="data/dataset_2s")
    ap.add_argument("--prefix-len", type=int, default=12)
    ap.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-bf16")
    args = ap.parse_args()

    P = args.prefix_len
    ci = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)
    spm = ci.get_text_tokenizer()
    mimi = ci.get_mimi(device="cuda"); mimi.eval()
    fr = mimi.frame_rate
    interleaver = Interleaver(spm, fr, TEXT_PADDING_ID, EOT_PADDING_ID,
                              ZERO_TOKEN_ID, keep_main_only=True)
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.codes, "*.npz")))
    files = [f for f in files if os.path.basename(f) != "index.npz"]
    index = []
    for f in files:
        cid = os.path.basename(f).replace(".npz", "")
        codes16 = np.load(f)["codes"].astype(np.int32)  # [16, T]
        assert codes16.shape[0] == 16, codes16.shape
        T = codes16.shape[1]
        with open(os.path.join(args.sched, cid + ".json")) as fh:
            sched = json.load(fh)
        name = sched["name"]
        row0 = build_row0(interleaver, sched["turns"], T, fr)  # [T]

        call = np.full((17, T), ZERO_TOKEN_ID, dtype=np.int32)
        call[0, :] = row0
        call[1:17, :] = codes16  # rows1-8=L/Moshi, rows9-16=R/User

        name_ids = spm.encode(name)[:P]
        prefix = np.full((17, P), ZERO_TOKEN_ID, dtype=np.int32)
        prefix[0, :len(name_ids)] = name_ids
        prefix[0, len(name_ids):] = TEXT_PADDING_ID

        full = np.concatenate([prefix, call], axis=1)  # [17, P+T]
        dec = spm.decode([int(x) for x in prefix[0, :len(name_ids)]])
        np.savez(os.path.join(args.out, cid + ".npz"), codes=full, prefix_len=np.int32(P))
        index.append({"id": cid, "name": name, "prefix_len": P, "T": int(T),
                      "shape": list(full.shape), "name_decode": dec})
        print("%s name=%r prefix_decode=%r shape=%s T=%d" % (cid, name, dec, full.shape, T))

    with open(os.path.join(args.out, "index.jsonl"), "w") as fh:
        for r in index:
            fh.write(json.dumps(r) + "\n")
    print("DONE: %d samples -> %s (prefix_len=%d)" % (len(index), args.out, P))


if __name__ == "__main__":
    main()
