
# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
# File: viz/gradcam.py
# Purpose: Grad-CAM for CNNs (e.g., MobileNetV3). Returns PIL overlay.
# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤

from typing import Tuple
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.fmap = None
        self.grad = None
        def f_hook(m, i, o): self.fmap = o.detach()
        def b_hook(m, gi, go): self.grad = go[0].detach()
        target_layer.register_forward_hook(f_hook)
        target_layer.register_backward_hook(b_hook)

    def __call__(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)
        w = self.grad.mean(dim=(2,3), keepdim=True)  # GAP over H,W
        cam = (w * self.fmap).sum(dim=1)             # [B, H, W]
        cam = F.relu(cam)
        cam = cam - cam.min(dim=(1,2), keepdim=True)[0]
        cam = cam / (cam.max(dim=(1,2), keepdim=True)[0] + 1e-9)
        return cam  # [B,H,W] in [0,1]


def overlay_cam_on_pil(img_pil: Image.Image, cam: np.ndarray, color=(255,0,0)) -> Image.Image:
    img = img_pil.convert('RGBA')
    h,w = img.size[1], img.size[0]
    cam = Image.fromarray((cam*255).astype(np.uint8)).resize((w,h), Image.BILINEAR)
    heat = Image.new('RGBA', (w,h), color + (0,))
    heat.putalpha(cam)
    return Image.alpha_composite(img, heat).convert('RGB')

