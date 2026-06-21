"""KV-prefill training: samples [17, P+C] = PREFIX(name) ++ CALL(greeting).
Loss only on CALL frames: forward full tensor (model sees name in KV) but zero the first
prefix_len mask columns. compute_loss_with_mask divides by sum(weights) so prefix drops out.
Adapted from native/train_ddp.py.
"""
import argparse, glob, json, os
import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from finetune.loss import compute_loss_with_mask
from native.init import build_lora_model, count_params, LORA_RANK, LORA_SCALING


class NpzKV(Dataset):
    def __init__(self, data_dir):
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.files = [f for f in files if os.path.basename(f) != "index.npz"]
        assert self.files, "no npz in %s" % data_dir
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        return d["codes"].astype(np.int64), int(d["prefix_len"])


def collate(batch):
    Tmax = max(a.shape[1] for a, _ in batch)
    out = np.full((len(batch), 17, Tmax), -1, dtype=np.int64)
    plens = np.zeros(len(batch), dtype=np.int64)
    for i, (a, p) in enumerate(batch):
        out[i, :, : a.shape[1]] = a
        plens[i] = p
    return torch.from_numpy(out), torch.from_numpy(plens)


def compute_losses(ddp, lm, codes, prefix_lens):
    output = ddp(codes=codes, condition_tensors=None)
    text_mask = output.text_mask.clone()
    audio_mask = output.mask.clone()
    for i, p in enumerate(prefix_lens.tolist()):
        text_mask[i, :, :p] = False
        audio_mask[i, :, :p] = False
    text_loss = compute_loss_with_mask(
        output.text_logits, codes[:, : lm.audio_offset], text_mask,
        mode="text", text_padding_weight=0.2,
        text_padding_ids={lm.text_padding_token_id, lm.end_of_text_padding_id})
    audio_loss = compute_loss_with_mask(
        output.logits, codes[:, lm.audio_offset : lm.audio_offset + lm.dep_q], audio_mask,
        mode="audio", first_codebook_weight_multiplier=100.0)
    return text_loss, audio_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rank", type=int, default=LORA_RANK)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--find-unused", action="store_true")
    args = ap.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = "cuda:%d" % local_rank
    dist.init_process_group(backend="nccl")
    dtype = getattr(torch, args.dtype)
    lm, ci = build_lora_model(device=device, dtype=dtype, lora_rank=args.rank, gradient_checkpointing=args.grad_ckpt)
    lm.train()
    if rank == 0:
        tr, tot = count_params(lm)
        print("[rank0] trainable=%d total=%d" % (tr, tot), flush=True)
    ddp = torch.nn.parallel.DistributedDataParallel(
        lm, device_ids=[local_rank], find_unused_parameters=args.find_unused,
        static_graph=(args.grad_ckpt and not args.find_unused))
    ds = NpzKV(args.data)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(ds, batch_size=args.batch, sampler=sampler,
                        collate_fn=collate, num_workers=2, drop_last=True)
    trainable = [p for p in ddp.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.wd)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    os.makedirs(args.out, exist_ok=True)
    log_fh = open(os.path.join(args.out, "train.log"), "w") if rank == 0 else None
    def log(msg):
        if rank == 0:
            print(msg, flush=True); log_fh.write(msg + "\n"); log_fh.flush()
    step, epoch, done = 0, 0, False
    while not done:
        sampler.set_epoch(epoch)
        for codes, plens in loader:
            codes = codes.to(device)
            optimizer.zero_grad()
            tl, al = compute_losses(ddp, lm, codes, plens)
            loss = tl + al
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step(); scheduler.step()
            if step % 25 == 0 or step == args.steps - 1:
                log("step %d loss %.4f (text %.4f audio %.4f) lr %.2e" % (
                    step, loss.item(), tl.item(), al.item(), scheduler.get_last_lr()[0]))
            step += 1
            if step >= args.steps:
                done = True; break
        epoch += 1
    if rank == 0:
        peak = torch.cuda.max_memory_allocated(local_rank) / 1e9
        log("[rank0] peak_mem=%.1f GB" % peak)
        sd = ddp.module.state_dict()
        lora_sd = {k: v.contiguous().to(torch.bfloat16) for k, v in sd.items() if "lora" in k}
        save_file(lora_sd, os.path.join(args.out, "lora.safetensors"))
        with open(os.path.join(args.out, "config.json"), "w") as fh:
            json.dump({"lora": True, "lora_rank": args.rank, "lora_scaling": LORA_SCALING,
                       "hf_repo": "kyutai/moshiko-pytorch-bf16"}, fh, indent=2)
        log("saved %d lora tensors to %s/lora.safetensors" % (len(lora_sd), args.out))
        log_fh.close()
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
