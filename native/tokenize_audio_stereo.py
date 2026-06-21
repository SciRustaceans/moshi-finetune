"""Encode a directory of STEREO wavs into Mimi audio codes [16, T].

For each <id>.wav (stereo, ch0=LEFT/Moshi, ch1=RIGHT/user), encodes BOTH channels with
mimi.encode and flattens to 16 codebooks via view(1,-1,T): rows 0-7 = LEFT (Moshi), rows
8-15 = RIGHT (user). This matches the interleaver stereo convention (mimi.encode of
[2,1,samples] -> [2,8,T] -> view(1,-1,T) -> [1,16,T]); confirmed empirically (LEFT block
== rows 0-7). Output audio_codes_2s/<id>.npz with int16 array codes [16, T].

Mono fallback: if a wav has 1 channel, the user (RIGHT) block is set to encoded silence so
shape stays [16, T] (not expected for two-sided data; warns).
"""
import argparse, os
import numpy as np
import sphn
import torch
from moshi.models import loaders

SAMPLE_RATE = 24000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-bf16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ci = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)
    mimi = ci.get_mimi(device=args.device)
    mimi.eval()

    wavs = sorted(f for f in os.listdir(args.inp) if f.lower().endswith(".wav"))
    for fn in wavs:
        stem = os.path.splitext(fn)[0]
        wav, sr = sphn.read(os.path.join(args.inp, fn))
        if wav.ndim == 1:
            wav = wav[None]
        if sr != SAMPLE_RATE:
            wav = sphn.resample(wav, src_sample_rate=sr, dst_sample_rate=SAMPLE_RATE)
        if wav.shape[0] == 1:
            print("WARN mono wav, duplicating user=silence:", stem)
            mono = torch.from_numpy(wav).to(args.device).float()
            with torch.no_grad():
                c = mimi.encode(mono[:, None])[0]
                sil = mimi.encode(torch.zeros_like(mono)[:, None])[0]
            codes = torch.cat([c, sil], dim=0)
        else:
            wav = wav[:2]
            audio = torch.from_numpy(wav).to(args.device).float()
            with torch.no_grad():
                enc = mimi.encode(audio[:, None])
            T = enc.shape[-1]
            codes = enc.view(1, -1, T)[0]
        codes = codes.to(torch.int16).cpu().numpy()
        assert codes.shape[0] == 16, codes.shape
        np.savez(os.path.join(args.out, stem + ".npz"), codes=codes)
        print("%s: codes %s dtype %s" % (stem, codes.shape, codes.dtype))


if __name__ == "__main__":
    main()
