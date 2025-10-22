"""
Waste Classification Demo — upgraded app.py
- Checkpoint selector (supports timm MobileNetV3 / DeiT-ViT names)
- Temperature scaling (reads metrics/metrics_calibration.json if present)
- Unsure threshold control
- Explain: Grad-CAM for CNN, Attention Rollout for ViT (inline implementations)
- Evidence export (input, heatmap, JSON result)

Folder expectations:
- checkpoints/baseline.pt           # a PyTorch checkpoint with keys: model, state_dict, classes
- (optional) checkpoints/vit_tiny.pt
- (optional) metrics/metrics_calibration.json  # {"T*": 1.23, ...}

Requirement: pip install gradio timm torch torchvision pillow numpy

Run:  python app1.py
"""

from __future__ import annotations

import os
import json
import time
import uuid
from functools import lru_cache
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image

# -----------------------------
# Config
# -----------------------------
DEFAULT_IMG_SIZE = 224
DEFAULT_CKPT = "checkpoints/baseline.pt"
CALIB_JSON = "metrics/metrics_calibration.json"

# Optional: replace with a CSV reader if you maintain labels_mapping.csv
COARSE_MAPPING: Dict[str, str] = {
    # fine -> coarse (edit to your project)
    "cardboard": "Recyclable",
    "glass": "Recyclable",
    "metal": "Recyclable",
    "paper": "Recyclable",
    "plastic": "Recyclable",
    "trash": "Trash",
    "organic": "Organic",
    "recyclable": "Recyclable",
}


# -----------------------------
# Utilities: calibration, loading, preprocessing
# -----------------------------
def load_Tstar(path: str = CALIB_JSON) -> float:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return float(obj.get("T*", 1.0))
    except Exception:
        return 1.0


@lru_cache(maxsize=8)
def load_ckpt(ckpt_path: str) -> Tuple[nn.Module, List[str], str]:
    """
    Returns: (model, classes, model_name)
    Expects checkpoint to have keys: 'model', 'state_dict', 'classes'
    """
    ck = torch.load(ckpt_path, map_location="cpu")
    model_name = ck["model"]
    classes = ck["classes"]
    model = timm.create_model(model_name, pretrained=False, num_classes=len(classes))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, classes, model_name


def preprocess_pil_to_tensor(img_pil: Image.Image, size: int = DEFAULT_IMG_SIZE) -> torch.Tensor:
    """Resize to size x size, normalize to ImageNet stats, return [1,3,H,W] float32 tensor."""
    img = img_pil.convert("RGB").resize((size, size), Image.BILINEAR)
    x = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)  # [1,3,H,W]


# -----------------------------
# Inference (with calibration + unsure + coarse aggregation)
# -----------------------------
@torch.no_grad()
def predict_one(
    img_pil: Image.Image,
    ckpt_path: str,
    unsure_threshold: float = 0.60,
    use_calibration: bool = True,
) -> Dict:
    t0 = time.perf_counter()
    model, classes, model_name = load_ckpt(ckpt_path)
    x = preprocess_pil_to_tensor(img_pil, DEFAULT_IMG_SIZE)

    logits = model(x)
    Tstar = load_Tstar() if use_calibration else 1.0
    probs = torch.softmax(logits / Tstar, dim=1).cpu().numpy()[0]  # (C,)

    # fine-grained top-3
    idx_top3 = np.argsort(-probs)[:3]
    top3 = [(classes[i], float(probs[i])) for i in idx_top3]
    pmax = float(probs[idx_top3[0]])
    unsure = pmax < float(unsure_threshold)

    # coarse aggregation (3-way), using COARSE_MAPPING
    coarse_scores: Dict[str, float] = {}
    for fine, p in zip(classes, probs):
        coarse = COARSE_MAPPING.get(fine, fine)
        coarse_scores[coarse] = coarse_scores.get(coarse, 0.0) + float(p)
    s = sum(coarse_scores.values()) + 1e-12
    for k in list(coarse_scores.keys()):
        coarse_scores[k] /= s
    pred_coarse = max(coarse_scores.items(), key=lambda kv: kv[1])[0]

    return {
        "model_name": model_name,
        "classes": classes,
        "top3": top3,
        "pmax": pmax,
        "unsure": unsure,
        "pred_coarse": pred_coarse,
        "latency_ms": (time.perf_counter() - t0) * 1000.0,
        "Tstar": Tstar,
    }


# -----------------------------
# Explainability — Grad-CAM (CNN)
# -----------------------------
class _GradCAMPipe:
    """
    Minimal Grad-CAM: automatically find a Conv2d layer if target not provided.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model.eval()
        self.target = target_layer or self._auto_pick_last_conv(model)
        self._fmap = None
        self._grad = None

        def f_hook(m, i, o):  # forward
            self._fmap = o.detach()

        def b_hook(m, gi, go):  # backward
            self._grad = go[0].detach()

        self._fh = self.target.register_forward_hook(f_hook)
        self._bh = self.target.register_backward_hook(b_hook)

    @staticmethod
    def _auto_pick_last_conv(model: nn.Module) -> nn.Module:
        last = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is None:
            raise RuntimeError("No Conv2d layer found for Grad-CAM.")
        return last

    def __call__(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        # weights = global-average-pooled gradients
        w = self._grad.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (w * self._fmap).sum(dim=1)  # [B, H, W]
        cam = torch.relu(cam)
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-9)
        return cam  # [B,H,W] in [0,1]


def _overlay_mask_on_pil(img_pil: Image.Image, mask: np.ndarray, color=(255, 0, 0)) -> Image.Image:
    """Alpha-overlay a single-channel mask [H,W] onto PIL image."""
    img = img_pil.convert("RGBA")
    w, h = img.size
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    color_layer = Image.new("RGBA", (w, h), color + (0,))
    color_layer.putalpha(mask_img)
    out = Image.alpha_composite(img, color_layer).convert("RGB")
    return out


# -----------------------------
# Explainability — ViT Attention Rollout (hook-based, timm compatible)
# -----------------------------
class _ViTRollout:
    """
    Attention rollout for timm ViT/DeiT models. Collects attn matrices via hooks.
    Produces a [H,W] mask (normalized to [0,1]) assuming 224/16 -> 14x14 patches.
    """

    def __init__(self, model: nn.Module):
        self.model = model.eval()
        self.attns: List[torch.Tensor] = []
        self.hooks = []
        # Register hooks on blocks' attention
        blocks = getattr(model, "blocks", [])
        if not blocks:
            raise RuntimeError("No .blocks found on ViT model.")
        for blk in blocks:
            attn_mod = getattr(blk, "attn", None)
            if attn_mod is None:
                continue

            def hook_fn(module, inputs, outputs):
                # outputs may be attn output; try to fetch attention weights if exposed
                # Fallback: if not exposed, try module.attn_drop or use module's saved 'attn' if present
                attn = getattr(module, "attn", None)
                if attn is not None and isinstance(attn, torch.Tensor):
                    self.attns.append(attn.detach())
                else:
                    # Try to infer by searching in module internals (version dependent)
                    # If unavailable, skip; rollout will degrade to uniform mask.
                    pass

            # Prefer using forward_pre_hook to grab internal attn if module stores it
            h = attn_mod.register_forward_hook(hook_fn)
            self.hooks.append(h)

    def __del__(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.attns = []
        _ = self.model(x)
        if not self.attns:
            # Fallback: uniform attention mask (avoid crash on older timm versions)
            return torch.ones(1, 14, 14)

        # attn: list of [B, heads, tokens, tokens]
        rollout = None
        for A in self.attns:
            A = A.mean(dim=1)  # average heads  -> [B,T,T]
            I = torch.eye(A.size(-1), device=A.device).unsqueeze(0)
            A = A + I
            A = A / A.sum(dim=-1, keepdim=True)
            rollout = A if rollout is None else rollout @ A

        # From CLS token to patch tokens
        mask = rollout[:, 0, 1:]  # [B, T-1]
        side = int((mask.size(-1)) ** 0.5)  # assume square grid
        mask = mask.view(mask.size(0), side, side)
        # normalize to [0,1]
        mask = (mask - mask.amin()) / (mask.amax() - mask.amin() + 1e-9)
        return mask


# -----------------------------
# Explain dispatcher
# -----------------------------
@torch.no_grad()
def explain_image(img_pil: Image.Image, ckpt_path: str, method: str = "auto") -> Image.Image:
    model, classes, model_name = load_ckpt(ckpt_path)
    x = preprocess_pil_to_tensor(img_pil, DEFAULT_IMG_SIZE)
    logits = model(x)
    pred_idx = int(logits.argmax(dim=1))

    is_vit = ("vit" in model_name.lower()) or ("deit" in model_name.lower())

    # CNN via Grad-CAM
    if method == "gradcam" or (method == "auto" and not is_vit):
        try:
            # try common final conv aliases first
            target = getattr(model, "conv_head", None) or getattr(model, "conv_stem", None)
        except Exception:
            target = None
        cam = _GradCAMPipe(model, target_layer=target)
        mask = cam(x, pred_idx)[0].cpu().numpy()
        return _overlay_mask_on_pil(img_pil, mask, color=(255, 0, 0))

    # ViT via attention rollout
    try:
        rollout = _ViTRollout(model)
        mask = rollout(x)[0].cpu().numpy()
        return _overlay_mask_on_pil(img_pil, mask, color=(0, 255, 0))
    except Exception:
        # If attention not accessible, return input as-is to avoid breaking demo
        return img_pil.copy()


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Waste Classification Demo (Calibrated + Explainable)") as demo:
    gr.Markdown(
        "# Waste Classification Demo\n"
        "Calibrated probabilities, Unsure threshold, and explainability (Grad-CAM / ViT attention).\n"
        "Tip: keep preprocessing consistent with training for best accuracy."
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            # Inputs / controls
            img_input = gr.Image(type="pil", label="Upload Image or use Webcam", sources=["upload", "webcam"], height=320)

            ckpt_dropdown = gr.Dropdown(
                choices=[DEFAULT_CKPT, "checkpoints/vit_tiny.pt"],
                value=DEFAULT_CKPT,
                label="Checkpoint"
            )
            unsure_slider = gr.Slider(0.20, 0.95, value=0.60, step=0.01, label="Unsure threshold")
            calibrated_checkbox = gr.Checkbox(value=True, label="Use calibration (T*)")
            explain_method_radio = gr.Radio(["auto", "gradcam", "attn_rollout"], value="auto", label="Explain method")

            predict_btn = gr.Button("Predict ▶", variant="primary")
            explain_btn = gr.Button("Explain 🔍")
            export_btn = gr.Button("Export ⬇")

        with gr.Column(scale=3):
            # Outputs
            pred_text = gr.Textbox(label="Coarse Prediction (3-way)", interactive=False)
            topk_label = gr.Label(label="Top-3 (fine)")  # expects dict: {class: prob}
            unsure_chk = gr.Checkbox(label="Unsure", interactive=False)
            info_md = gr.Markdown()

            heat_image = gr.Image(label="Heatmap", height=320)
            export_file = gr.File(label="Download manifest")

    # Keep last result state for export/explain
    state_last_result = gr.State(value={})
    state_last_heatmap = gr.State(value=None)

    # ---- Callbacks ----
    def on_predict(img_pil, ckpt_path, thr, calibrated):
        if img_pil is None:
            raise gr.Error("Please upload an image first.")
        res = predict_one(img_pil, ckpt_path, thr, calibrated)
        info = f"Model: `{res['model_name']}` · T*: **{res['Tstar']:.2f}** · Latency: **{res['latency_ms']:.1f} ms**"
        topk_dict = {k: v for k, v in res["top3"]}
        return res["pred_coarse"], topk_dict, bool(res["unsure"]), info, res

    predict_btn.click(
        fn=on_predict,
        inputs=[img_input, ckpt_dropdown, unsure_slider, calibrated_checkbox],
        outputs=[pred_text, topk_label, unsure_chk, info_md, state_last_result],
    )

    def on_explain(img_pil, ckpt_path, method, state):
        if img_pil is None:
            raise gr.Error("Please upload and run a prediction first.")
        heat = explain_image(img_pil, ckpt_path, method)
        return heat, heat

    explain_btn.click(
        fn=on_explain,
        inputs=[img_input, ckpt_dropdown, explain_method_radio, state_last_result],
        outputs=[heat_image, state_last_heatmap],
    )

    def export_case(img_pil, heat_pil, state: dict):
        if not state:
            raise gr.Error("No prediction to export. Run Predict first.")
        out_dir = os.path.join("evidence", f"case_{uuid.uuid4().hex[:8]}")
        os.makedirs(out_dir, exist_ok=True)
        if img_pil:
            img_pil.save(os.path.join(out_dir, "input.jpg"), quality=95)
        if heat_pil:
            heat_pil.save(os.path.join(out_dir, "heatmap.jpg"), quality=95)
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        return os.path.join(out_dir, "result.json")

    export_btn.click(
        fn=export_case,
        inputs=[img_input, state_last_heatmap, state_last_result],
        outputs=[export_file],
    )

    gr.Markdown(
        "---\n"
        "**Notes**: \n"
        "- If `metrics/metrics_calibration.json` is missing, T* defaults to 1.0 (no calibration).\n"
        "- Grad-CAM automatically picks the last Conv2d if a specific layer isn’t set.\n"
        "- ViT attention rollout may fallback when attention weights aren’t exposed in your timm version.\n"
    )

if __name__ == "__main__":
    demo.launch()
