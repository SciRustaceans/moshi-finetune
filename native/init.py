"""Build moshiko + LoRA on a single GPU, freeze all params except LoRA.

Uses the in-repo loader path (moshi.models.loaders) directly. We intentionally do not
reuse finetune.wrapped_model.get_fsdp_model: that helper needs an initialized process
group (get_rank / barrier) for the multi-GPU FSDP path. The direct loader path runs the
same meta-build, weight load, and LoRA replacement that get_moshi_lm does internally.
"""

import torch
from moshi.models import loaders

HF_REPO = "kyutai/moshiko-pytorch-bf16"
LORA_RANK = 128
LORA_SCALING = 2.0


def build_lora_model(device="cuda", dtype=torch.bfloat16,
                     lora_rank=LORA_RANK, lora_scaling=LORA_SCALING, hf_repo=HF_REPO,
                     gradient_checkpointing=True):
    """Return (model, checkpoint_info).

    model has LoRALinear layers; every param frozen except lora_A / lora_B (trainable).
    lora_B is zero-initialised so the adapter is identity at init (the direct-to-device
    build path does not zero it for us).
    """
    ci = loaders.CheckpointInfo.from_hf_repo(hf_repo)
    model = ci.get_moshi(
        device=device,
        dtype=dtype,
        lm_kwargs_overrides={
            "gradient_checkpointing": gradient_checkpointing,
            "lora": True,
            "lora_rank": lora_rank,
            "lora_scaling": lora_scaling,
        },
    )
    for name, p in model.named_parameters():
        p.requires_grad = "lora" in name
    with torch.no_grad():
        for name, module in model.named_modules():
            if name.split(".")[-1] == "lora_B" and hasattr(module, "weight"):
                module.weight.zero_()
    return model, ci


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def pad_to_17(codes9):
    """[B,9,T] (1 text + 8 Moshi audio) -> [B,17,T]; user rows 9..16 filled with -1.

    moshiko.forward asserts K==17. User rows are zero_token_id (-1): no loss, no input.
    """
    assert codes9.dim() == 3 and codes9.shape[1] == 9, codes9.shape
    B, _, T = codes9.shape
    user = torch.full((B, 8, T), -1, dtype=codes9.dtype, device=codes9.device)
    return torch.cat([codes9, user], dim=1)


if __name__ == "__main__":
    m, ci = build_lora_model()
    tr, tot = count_params(m)
    print(f"trainable={tr:,} total={tot:,} ({100*tr/tot:.3f}%)")
    bad = [n for n, p in m.named_parameters() if p.requires_grad and "lora" not in n]
    assert not bad, bad
    print("OK: only lora_* trainable")
