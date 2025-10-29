
# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
# File: viz/vit_rollout.py
# Purpose: Attention rollout for ViT (timm models). Returns PIL overlay.
# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤

from typing import List
import torch
from PIL import Image
import numpy as np

@torch.no_grad()
def attention_rollout(model, x: torch.Tensor, discard_ratio: float=0.0) -> torch.Tensor:
    """Compute attention rollout from timm ViT. Returns [B, H, W] mask in [0,1]."""
    model.eval()
    attns: List[torch.Tensor] = []
    # grab attention from each block
    for blk in getattr(model, 'blocks', []):  # DeiT has .blocks
        attn = blk.attn.get_attention_map(x) if hasattr(blk.attn, 'get_attention_map') else None
        if attn is None:
            # fallback by running forward with hooks would be needed in some timm versions
            pass
        else:
            attns.append(attn)  # [B, heads, tokens, tokens]
    if not attns:
        # naive fallback: uniform map
        B = x.size(0)
        return torch.ones(B, 14, 14)
    # average heads then rollout
    rollout = None
    for A in attns:
        A = A.mean(dim=1)  # [B, T, T]
        if discard_ratio>0:
            flat = A.view(A.size(0), -1)
            v, idx = torch.topk(flat, k=int(flat.size(1)*discard_ratio), largest=False)
            flat.scatter_(1, idx, 0.0)
            A = flat.view_as(A)
        I = torch.eye(A.size(-1), device=A.device).unsqueeze(0)
        A = A + I
        A = A / A.sum(dim=-1, keepdim=True)
        rollout = A if rollout is None else rollout @ A
    # from [CLS] to patch tokens attention
    mask = rollout[:, 0, 1:]
    # assuming 14x14 tokens for 224/16
    side = int(mask.size(-1)**0.5)
    mask = mask.view(mask.size(0), side, side)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)
    return mask


def overlay_rollout_on_pil(img_pil: Image.Image, mask: np.ndarray, color=(0,255,0)) -> Image.Image:
    img = img_pil.convert('RGBA')
    h,w = img.size[1], img.size[0]
    heat = Image.fromarray((mask*255).astype(np.uint8)).resize((w,h), Image.BILINEAR)
    col = Image.new('RGBA', (w,h), color + (0,))
    col.putalpha(heat)
    return Image.alpha_composite(img, col).convert('RGB')

