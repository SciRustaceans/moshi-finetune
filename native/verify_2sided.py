import argparse, glob, json, os
import numpy as np
import torch
from moshi.models import loaders

TEXT_PADDING_ID = 3
ZERO = -1
FRAME_RATE = 12.5
SIL_MAX = 6
REAL_MIN = 24


def decode_row0(spm, row0_seg):
    toks = [int(x) for x in row0_seg if x not in (TEXT_PADDING_ID, 0, ZERO)]
    return spm.decode(toks) if toks else ""


def block_status(block):
    n = block.shape[1]
    if n == 0:
        return "EMPTY", 0.0
    if np.all(block == ZERO):
        return "ZERO(-1)", 0.0
    uniq = float(np.mean([len(np.unique(block[r])) for r in range(block.shape[0])]))
    norm = uniq / max(1, n)
    if uniq <= SIL_MAX:
        label = "SILENCE-codes"
    elif uniq >= REAL_MIN or norm >= 0.5:
        label = "REAL-audio"
    else:
        label = "DEADZONE(uniq=%.1f)" % uniq
    return label, uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset_kv_2sided")
    ap.add_argument("--sidecar", default="data/wav_2s_clean")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-bf16")
    args = ap.parse_args()
    ci = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)
    spm = ci.get_text_tokenizer()
    print("metric: SILENCE if mean-unique-codes/frame <= %d, REAL if >= %d (else DEADZONE)" % (SIL_MAX, REAL_MIN))
    print("note: inactive channel = encoded-silence (low-entropy mimi codes), NOT -1. -1 only in prefix rows1-16.")
    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    files = [f for f in files if os.path.basename(f) != "index.npz"]
    if args.ids:
        files = [f for f in files if os.path.basename(f).replace(".npz", "") in args.ids]
    onsets = {}
    idx_path = os.path.join(args.data, "index.jsonl")
    if os.path.exists(idx_path):
        for ln in open(idx_path):
            r = json.loads(ln); onsets[r["id"]] = r.get("onset", 0)
    for f in files:
        cid = os.path.basename(f).replace(".npz", "")
        d = np.load(f)
        codes = d["codes"].astype(np.int32)
        P = int(d["prefix_len"])
        call = codes[:, P:]
        meta = json.load(open(os.path.join(args.sidecar, cid + ".json")))
        onset = onsets.get(cid, 0)
        print("=" * 78)
        print("%s shape=%s P=%d call=%d onset=%d name=%r" % (cid, codes.shape, P, call.shape[1], onset, meta["name"]))
        pref = codes[:, :P]
        name_ids = [int(x) for x in pref[0] if x != TEXT_PADDING_ID]
        dec = spm.decode(name_ids)
        rows_neg1 = bool(np.all(pref[1:17] == ZERO))
        print("  PREFIX: row0 name_ids=%s decode=%r | rows1-16 all -1? %s" % (name_ids, dec, rows_neg1))
        for j, s in enumerate(meta["schedule"]):
            a = int(np.floor(s["start"] * FRAME_RATE)) - onset
            b = int(np.ceil(s["end"] * FRAME_RATE)) - onset
            a = max(0, a); b = min(call.shape[1], b)
            if b <= a:
                continue
            row0txt = decode_row0(spm, call[0, a:b])
            ml, mu = block_status(call[1:9, a:b])
            ul, uu = block_status(call[9:17, a:b])
            tag = "GREETING" if j == 0 else ("MOSHI-RESP" if s["role"] == "moshi" else "USER-TURN")
            print("  [%s f%d-%d role=%s]" % (tag, a, b, s["role"]))
            print("      row0(text)=%r" % row0txt)
            print("      rows1-8(Moshi/L)=%s[u=%.0f]  rows9-16(user/R)=%s[u=%.0f]" % (ml, mu, ul, uu))
    print("VERIFY DONE")


if __name__ == "__main__":
    main()
