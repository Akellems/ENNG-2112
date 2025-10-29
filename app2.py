#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Gradio app for waste classification that works with team-trained models
without requiring a fixed folder layout.

Key design goals:
- Auto-discover artifacts (*.pt/*.pth/*.onnx) anywhere under the current folder.
- PyTorch checkpoints: support multiple formats:
    1) {"model": "<timm_name>", "state_dict": <...>, "classes": [...]}
    2) {"state_dict": <...>}  (need user to provide timm model name + classes)
    3) plain state_dict       (need user to provide timm model name + classes)
- ONNX: use onnxruntime if installed; otherwise hide .onnx entries.
- Classes: try to read from checkpoint; else read sidecar labels file; else let
  user paste a list (one per line). Default to common 7-class list if empty.
- Unsure threshold: mark "Unsure" if max prob < threshold.
- Coarse mapping (optional): aggregate fine classes to {Recyclable, Organic, Trash}.
- Evidence export: create folder if missing and dump input + JSON result.

Dependencies:
  pip install torch torchvision timm gradio pillow numpy pathlib
  # optional for ONNX:
  # pip install onnxruntime onnx
  
  # For heatmap explanation:
  pip install opencv-python

Run:
  python app2.py
"""

import os, io, json, time, glob, sys, math, traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image

from pathlib import Path
from typing import Tuple

import gradio as gr

# Optional imports (tolerate absence)
_has_torch = True
try:
    import torch
    import torch.nn.functional as F
except Exception:
    _has_torch = False

_has_timm = True
try:
    import timm
except Exception:
    _has_timm = False

_has_onnx = True
try:
    import onnxruntime as ort
except Exception:
    _has_onnx = False

APP_NAME = "Waste Classification"
DEFAULT_TORCH_MODEL_NAME = "mobilenetv3_small_100"  # user-editable UI fallback
DEFAULT_CLASSES = ["cardboard","glass","metal","paper","plastic","trash","organic"]

# ---------- Utilities ----------

def ensure_dir(p: str):
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass

def now_millis():
    return int(time.time() * 1000)

def pil_to_bytes(pil_img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()

def to_device_cpu(t):
    if _has_torch and isinstance(t, torch.Tensor):
        return t.detach().cpu()
    return t

# ---------- Artifact discovery ----------

def discover_artifacts(max_depth: int = 2) -> List[str]:
    """
    Find *.pt, *.pth, *.onnx in current directory and shallow subfolders.
    """
    patterns = ["*.pt", "*.pth"]
    if _has_onnx:
        patterns.append("*.onnx")
    found = set()
    base = Path(".").resolve()

    # depth 0..max_depth
    for depth in range(max_depth + 1):
        glob_patterns = []
        if depth == 0:
            for pat in patterns:
                glob_patterns.append(str(base / pat))
        else:
            star = "/".join(["*"] * depth)
            for pat in patterns:
                glob_patterns.append(str(base / star / pat))

        for gp in glob_patterns:
            for f in glob.glob(gp):
                # hide .onnx if onnxruntime unavailable
                if f.lower().endswith(".onnx") and not _has_onnx:
                    continue
                found.add(str(Path(f).resolve()))

    # Sort by mtime desc (newest first) then by path
    def sort_key(p: str):
        try:
            return (-os.path.getmtime(p), p)
        except Exception:
            return (0, p)

    return sorted(list(found), key=sort_key)

# ---------- Labels handling ----------

def load_sidecar_labels(art_path: str) -> Optional[List[str]]:
    """
    Try to find a labels file next to the artifact.
    Accept: labels.txt / classes.txt (one class per line).
    """
    p = Path(art_path)
    candidates = [p.with_suffix(".labels.txt"), p.with_suffix(".classes.txt"),
                  p.parent / "labels.txt", p.parent / "classes.txt"]
    for c in candidates:
        try:
            if c.exists():
                with open(c, "r", encoding="utf-8") as fh:
                    lines = [ln.strip() for ln in fh.readlines() if ln.strip()]
                    if lines:
                        return lines
        except Exception:
            continue
    return None

def parse_classes_text(text: str) -> List[str]:
    if not text.strip():
        return DEFAULT_CLASSES.copy()
    # allow comma or newline separated
    if "," in text and "\n" not in text.strip():
        # simple comma separated
        items = [x.strip() for x in text.split(",")]
    else:
        items = [x.strip() for x in text.splitlines()]
    # drop empties
    return [x for x in items if x]

# ---------- Coarse mapping ----------

def default_coarse_map(classes: List[str]) -> Dict[str, str]:
    recyclable_keys = {
        "recyclable", "cardboard", "paper", "paperboard", "carton",
        "plastic", "plastics", "bottle", "bottles",
        "glass", "jar", "jars",
        "metal", "can", "cans", "tin", "tins",
        "aluminum", "aluminium", "steel"
    }
    organic_keys = {
        "organic", "food", "kitchen", "biodegradable", "compost",
        "vegetable", "fruit", "leftover", "banana", "apple", "peel"
    }
    trash_keys = {
        "trash", "other", "unknown", "garbage", "waste",
        "cloth", "clothes", "shoe", "shoes", "ceramic", "styrofoam"
    }

    mapping = {}
    for c in classes:
        base = c.strip().lower().replace("_", " ").replace("-", " ")
        if base in recyclable_keys or any(k in base for k in recyclable_keys):
            mapping[c] = "Recyclable"
        elif base in organic_keys or any(k in base for k in organic_keys):
            mapping[c] = "Organic"
        elif base in trash_keys or any(k in base for k in trash_keys):
            mapping[c] = "Trash"
        else:
            mapping[c] = "Trash"
    return mapping




# ---------- PyTorch loading & prediction ----------

@dataclass
class TorchModelPack:
    model: Any
    classes: List[str]
    arch_name: str

def _try_infer_num_classes_from_state_dict(sd: Dict[str, Any]) -> Optional[int]:
    # Heuristic: find the last classifier weight
    # This is best-effort; works for common convnets (MobileNet/EfficientNet/etc.)
    candidates = []
    for k, v in sd.items():
        if not (isinstance(v, torch.Tensor) and v.ndim >= 1):
            continue
        # classifier weights often are 2D: [num_classes, in_features]
        if v.ndim == 2 and ("classifier" in k or "fc." in k or k.endswith("weight")):
            candidates.append(v.shape[0])
    if candidates:
        # choose the smallest plausible head (often the correct num_classes)
        return int(sorted(candidates)[0])
    return None

def load_torch_checkpoint(art_path: str,
                          user_arch_name: str,
                          user_classes: List[str]) -> TorchModelPack:
    if not _has_torch or not _has_timm:
        raise RuntimeError("Torch/Timm not available. Please install torch and timm.")

    raw = torch.load(art_path, map_location="cpu")

    arch_name = None
    state_dict = None
    classes = None

    if isinstance(raw, dict):
        if "model" in raw and "state_dict" in raw:
            arch_name = raw["model"]
            state_dict = raw["state_dict"]
            classes = raw.get("classes", None)
        elif "state_dict" in raw:
            state_dict = raw["state_dict"]
        else:
            # could be state_dict itself keyed by param names
            # (common when saved with torch.save(model.state_dict(), ...))
            # Verify by sampling a tensor
            if all(isinstance(v, torch.Tensor) for v in raw.values()):
                state_dict = raw
    elif isinstance(raw, (list, tuple)):
        # Extremely rare formats; treat as unsupported unless it's (state_dict, meta)
        for item in raw:
            if isinstance(item, dict) and all(isinstance(v, torch.Tensor) for v in item.values()):
                state_dict = item
                break

    if classes is None:
        # Priority: sidecar labels -> user input -> default
        sidecar = load_sidecar_labels(art_path)
        classes = sidecar if sidecar else user_classes

    if arch_name is None:
        arch_name = user_arch_name or DEFAULT_TORCH_MODEL_NAME

    # If head shape mismatches, we will rebuild with num_classes=len(classes)
    num_classes = len(classes)
    model = timm.create_model(arch_name, pretrained=False, num_classes=num_classes)

    # Try strict load; if fails, try non-strict; if still fails, try to trim head keys.
    err = None
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e1:
        err = e1
        try:
            model.load_state_dict(state_dict, strict=False)
            err = None
        except Exception as e2:
            err = e2

    if err is not None:
        # As last resort, attempt to drop head keys and load feature weights
        trimmed = {k: v for k, v in state_dict.items()
                   if not any(x in k for x in ["classifier", "head.fc", "fc.", "head.linear"])}
        try:
            model.load_state_dict(trimmed, strict=False)
        except Exception as e3:
            # give a more helpful error
            inferred = _try_infer_num_classes_from_state_dict(state_dict)
            hint = f"(inferred num_classes={inferred})" if inferred else ""
            raise RuntimeError(f"Failed to load state_dict into arch '{arch_name}'. "
                               f"Consider confirming the arch name and classes {hint}. "
                               f"Last error: {e3}") from e3

    model.eval()
    return TorchModelPack(model=model, classes=classes, arch_name=arch_name)

def preprocess_pil_to_tensor(img: Image.Image, img_size: int = 224) -> "torch.Tensor":
    # Minimal, deterministic preprocess matching common ImageNet models
    if not _has_torch:
        raise RuntimeError("torch is required for PyTorch models.")
    img = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    # standard ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    x = torch.from_numpy(arr)[None, ...]  # [1,3,H,W]
    return x

def predict_torch(pack: TorchModelPack,
                  img: Image.Image,
                  Tstar: float = 1.0) -> Tuple[np.ndarray, List[str]]:
    x = preprocess_pil_to_tensor(img, img_size=224)
    with torch.no_grad():
        logits = pack.model(x)
        logits = to_device_cpu(logits).numpy()
    probs = logits / (Tstar if Tstar and Tstar > 0 else 1.0)
    probs = np.exp(probs - probs.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs[0], pack.classes

# ---------- ONNX prediction ----------

@dataclass
class OnnxPack:
    sess: Any
    in_name: str
    out_name: str
    classes: List[str]
    arch_name: str

def _preprocess_np(img: Image.Image, img_size: int = 224) -> np.ndarray:
    img = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # [1,3,H,W]
    return arr

def load_onnx_model(art_path: str, user_classes: List[str]) -> OnnxPack:
    if not _has_onnx:
        raise RuntimeError("onnxruntime not available.")
    sess = ort.InferenceSession(art_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    sidecar = load_sidecar_labels(art_path)
    classes = sidecar if sidecar else user_classes
    return OnnxPack(sess=sess, in_name=in_name, out_name=out_name, classes=classes,
                    arch_name="onnx")

def predict_onnx(pack: OnnxPack,
                 img: Image.Image,
                 Tstar: float = 1.0) -> Tuple[np.ndarray, List[str]]:
    x = _preprocess_np(img, img_size=224)
    logits = pack.sess.run([pack.out_name], {pack.in_name: x})[0]
    probs = logits / (Tstar if Tstar and Tstar > 0 else 1.0)
    probs = np.exp(probs - probs.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs[0], pack.classes

# ---------- Inference wrapper ----------

@dataclass
class AnyModel:
    is_torch: bool
    torch_pack: Optional[TorchModelPack] = None
    onnx_pack: Optional[OnnxPack] = None

def build_model(art_path: str,
                user_arch_name: str,
                classes_text: str) -> Tuple[AnyModel, List[str], str]:
    classes = parse_classes_text(classes_text)
    if art_path.lower().endswith(".onnx"):
        pack = load_onnx_model(art_path, classes)
        return AnyModel(is_torch=False, onnx_pack=pack), pack.classes, pack.arch_name
    else:
        pack = load_torch_checkpoint(art_path, user_arch_name, classes)
        return AnyModel(is_torch=True, torch_pack=pack), pack.classes, pack.arch_name

def softmax_topk(probs: np.ndarray, classes: List[str], k: int = 3):
    idx = np.argsort(-probs)[:k]
    return [(classes[i], float(probs[i])) for i in idx]

def aggregate_coarse(probs: np.ndarray, classes: List[str],
                     mapping: Dict[str, str]) -> Dict[str, float]:
    buckets: Dict[str, float] = {}
    for c, p in zip(classes, probs):
        b = mapping.get(c, "Other")
        buckets[b] = buckets.get(b, 0.0) + float(p)
    # Normalize to 1.0 (safe guard numerical drift)
    s = sum(buckets.values()) or 1.0
    for k in list(buckets.keys()):
        buckets[k] = buckets[k] / s
    return buckets

# ---------- Evidence export ----------

def export_evidence(img: Image.Image, result: Dict[str, Any]) -> str:
    ts = now_millis()
    out_dir = Path("evidence") / f"case_{ts}"
    ensure_dir(out_dir.as_posix())
    # save input
    img.save(out_dir / "input.png")
    # save json
    with open(out_dir / "result.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return str(out_dir.resolve())

# ====== Heatmap helpers: Grad-CAM for CNN, Attention Rollout for ViT ======
if _has_torch:
    import torch.nn as nn

def _pick_target_conv_by_runtime(model: "nn.Module", sample: "torch.Tensor") -> "nn.Module":
    conv_feats = []
    hooks = []

    def _hook(m, i, o):
        if isinstance(o, torch.Tensor) and o.ndim == 4:
            _, _, H, W = o.shape
            conv_feats.append((m, H * W))

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_hook))

    with torch.no_grad():
        _ = model(sample)

    for h in hooks:
        try: h.remove()
        except: pass

    if not conv_feats:
        raise RuntimeError("No Conv2d outputs captured; model may be ViT or non-CNN.")
    big = [t for t in conv_feats if t[1] >= 4]
    return (big[-1][0] if big else conv_feats[-1][0])


def explain_torch_gradcam(pack: "TorchModelPack", img_pil: "Image.Image", img_size: int = 224):
    model = pack.model
    model.eval()

    x = preprocess_pil_to_tensor(img_pil, img_size=img_size)
    x.requires_grad_(True)

    target = _pick_target_conv_by_runtime(model, x.detach())

    feats, grads = [], []
    def f_hook(m, i, o): feats.append(o.detach())
    def b_hook(m, gi, go): grads.append(go[0].detach())

    h1 = target.register_forward_hook(f_hook)
    if hasattr(target, "register_full_backward_hook"):
        h2 = target.register_full_backward_hook(b_hook)
    else:
        h2 = target.register_backward_hook(b_hook)

    out = model(x) 
    cls = int(out.argmax(dim=1))
    score = out[0, cls]
    model.zero_grad()
    score.backward(retain_graph=True)

    try: h1.remove()
    except: pass
    try: h2.remove()
    except: pass

    if not feats or not grads:
        raise RuntimeError("Grad-CAM hooks did not capture data (feats/grads empty).")

    feat = feats[-1][0]   # [C,H,W]
    grad = grads[-1][0]   # [C,H,W]
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = (weights * feat).sum(dim=0)
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-7)
    cam = cam.detach().cpu().numpy()

    try:
        import cv2
    except Exception as e:
        raise RuntimeError("opencv-python not installed (pip install opencv-python)") from e

    img = np.array(img_pil.convert("RGB"))
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam_uint8 = np.uint8(255 * cam_resized)
    heat = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    over = (0.45 * heat + 0.55 * img).astype(np.uint8)
    return Image.fromarray(over), cam
# ---------------------------------------------------------------


def _tensor_to_cam(feat: "torch.Tensor", grad: "torch.Tensor") -> np.ndarray:
    # feat: [C,H,W], grad: [C,H,W]
    weights = grad.mean(dim=(1, 2), keepdim=True)             # [C,1,1]
    cam = (weights * feat).sum(dim=0)                         # [H,W]
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-7)
    return cam.detach().cpu().numpy()

def _overlay_heatmap_on_pil(img_pil: Image.Image, cam: np.ndarray, alpha=0.45) -> Image.Image:
    import cv2
    img = np.array(img_pil.convert("RGB"))
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam_uint8 = np.uint8(255 * cam_resized)
    heat = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    over = (alpha * heat + (1 - alpha) * img).astype(np.uint8)
    return Image.fromarray(over)



def explain_vit_rollout(pack: "TorchModelPack", img_pil: "Image.Image", img_size: int = 224,
                        head_fuse="mean") -> Tuple[Image.Image, np.ndarray]:
    """timm VisionTransformer attention rollout heatmap."""
    model = pack.model
    model.eval()

    att_mats = []
    hooks = []

    def _attn_hook(m, i, o):
        if isinstance(o, torch.Tensor) and o.ndim == 4:
            att_mats.append(o.detach().cpu())

    for blk in getattr(model, "blocks", []):
        attn = getattr(blk, "attn", None)
        if attn is None:
            continue
        target = None
        if hasattr(attn, "attn_drop"):   # timm
            target = attn.attn_drop
        elif hasattr(attn, "drop_attn"):
            target = attn.drop_attn
        else:
            target = attn
        hooks.append(target.register_forward_hook(_attn_hook))

    x = preprocess_pil_to_tensor(img_pil, img_size=img_size)
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        try: h.remove()
        except: pass

    if not att_mats:
        raise RuntimeError("No ViT attention captured; check timm version or block.attn.* names.")

    att = torch.stack(att_mats)[:, 0]  # [L, heads, T, T]
    att = att.mean(dim=1) if head_fuse == "mean" else att.max(dim=1).values  # [L,T,T]

    # rollout
    eye = torch.eye(att.shape[-1])
    att = att + eye
    att = att / att.sum(dim=-1, keepdim=True)
    joint = att[0]
    for i in range(1, att.shape[0]):
        joint = joint @ att[i]

    cls_attn = joint[0, 1:]
    L = int((cls_attn.numel()) ** 0.5)
    rollout = cls_attn.reshape(L, L)
    rollout = (rollout - rollout.min()) / (rollout.max() - rollout.min() + 1e-7)
    rollout = rollout.detach().cpu().numpy()

    try:
        import cv2
    except Exception as e:
        raise RuntimeError("opencv-python not installed (pip install opencv-python)") from e

    img = np.array(img_pil.convert("RGB"))
    att_resized = cv2.resize(rollout, (img.shape[1], img.shape[0]))
    att_uint8 = np.uint8(255 * att_resized)
    heat = cv2.applyColorMap(att_uint8, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    over = (0.45 * heat + 0.55 * img).astype(np.uint8)
    return Image.fromarray(over), rollout

# ====== end heatmap helpers ======



# ---------- Gradio UI callbacks ----------

def ui_refresh_artifacts() -> List[str]:
    arts = discover_artifacts(max_depth=2)
    if not arts:
        return ["<No model artifacts found>"]
    return arts

def ui_predict(artifact_path, unsure_threshold, use_coarse, tstar, image):
    if image is None:
        return ("No image.", "", {}, "")

    arts = ui_refresh_artifacts()
    if artifact_path not in arts:
        return ("Model file not found. Click 馃攧 to rescan.", "", {}, "")

    try:
        model, classes, arch_used = build_model(
    artifact_path,
    DEFAULT_TORCH_MODEL_NAME,
    "\n".join(DEFAULT_CLASSES)
)

        t0 = time.perf_counter()
        if model.is_torch:
            probs, classes = predict_torch(model.torch_pack, image, Tstar=tstar)
        else:
            probs, classes = predict_onnx(model.onnx_pack, image, Tstar=tstar)
        dt_ms = int((time.perf_counter() - t0) * 1000)

        coarse_map = default_coarse_map(classes)
        coarse_probs = aggregate_coarse(probs, classes, coarse_map)


        for k in ["Recyclable", "Organic", "Trash"]:
            coarse_probs.setdefault(k, 0.0)

        ranked = sorted(coarse_probs.items(), key=lambda x: -x[1])
        max_bucket, max_p = ranked[0][0], float(ranked[0][1])
        unsure = (max_p < unsure_threshold)

        label_text = max_bucket if not unsure else f"Unsure (< {unsure_threshold:.2f})"
        top3_text = "\n".join([f"{k}: {v:.3f}" for k, v in ranked])

        status = f"arch={arch_used} | coarse_maxP={max_p:.3f} | latency={dt_ms}ms | unsure={unsure}"

        return (label_text,
                top3_text,
                {"Recyclable": float(coarse_probs["Recyclable"]),
                 "Organic": float(coarse_probs["Organic"]),
                 "Trash": float(coarse_probs["Trash"])},
                status)

    except Exception as e:
        trace = traceback.format_exc()
        return (f"Error: {e}", "", {}, trace)
    
    
def ui_explain(artifact_path: str,
               arch_name_text: str,
               classes_text: str,
               tstar: float,
               image: Image.Image) -> Tuple[Image.Image | None, str]:
    import traceback

    def _ok(img, msg):   return (img, f"OK: {msg}")
    def _err(img, msg):  return (img, f"ERROR: {msg}")

    try:
        if image is None:
            return _err(None, "No image provided to Explain().")

        arts = ui_refresh_artifacts()
        if artifact_path not in arts:
            return _err(image, "Model artifact not found. Click Rescan.")

        model, classes, arch_used = build_model(artifact_path, arch_name_text, classes_text)

        if not model.is_torch:
            return _err(image, "ONNX model: heatmap not supported (needs PyTorch gradients).")

        # Grad-CAM
        try:
            overlay, cam = explain_torch_gradcam(model.torch_pack, image, img_size=224)
            return _ok(overlay, f"Grad-CAM on {arch_used} | cam.shape={getattr(cam, 'shape', None)}")
        except Exception as e1:
            msg1 = f"Grad-CAM failed: {repr(e1)}"

        # ViT Rollout
        try:
            overlay, att = explain_vit_rollout(model.torch_pack, image, img_size=224)
            return _ok(overlay, f"ViT Rollout on {arch_used} | att.shape={getattr(att, 'shape', None)} (fallback after Grad-CAM)")
        except Exception as e2:
            msg2 = f"ViT Rollout failed: {repr(e2)}"
            return _err(image, msg1 + " | " + msg2)

    except Exception as e:
        return _err(image if image is not None else None,
                    f"{repr(e)}\n{traceback.format_exc()}")




def ui_export(img: Image.Image,
              last_label: str,
              top3_text: str,
              coarse_probs: Dict[str, float],
              status_text: str) -> str:
    if img is None:
        return "No image to export."
    result = {
        "label_or_unsure": last_label,
        "top3": top3_text,
        "coarse": coarse_probs,
        "status": status_text,
        "app": APP_NAME,
        "ts": now_millis(),
    }
    path = export_evidence(img, result)
    return f"Exported to: {path}"

# ---------- Build UI ----------

def build_ui():
    with gr.Blocks(title=APP_NAME) as demo:
        gr.Markdown(f"# {APP_NAME}\n"
                    "Upload an image of waste and classify it using team-trained models. ")

        with gr.Row():
            with gr.Column(scale=2):
                artifact = gr.Dropdown(label="Model artifact", choices=ui_refresh_artifacts(), value=None)
                refresh_btn = gr.Button("Rescan models", variant="secondary")
                arch_name = gr.Textbox(label="(For PyTorch .pt only) Timm model name",
                       value=DEFAULT_TORCH_MODEL_NAME, visible=False)
                classes_txt = gr.Textbox(label="Classes (one per line or comma-separated)",
                         value="\n".join(DEFAULT_CLASSES), lines=5, visible=False)
                unsure = gr.Slider(0.0, 1.0, value=0.60, step=0.01, label="Unsure threshold")
                tstar = gr.Slider(0.1, 3.0, value=1.0, step=0.05, label="Temperature T* (calibration)")
                coarse = gr.Checkbox(value=True, label="Enable coarse 3-class aggregation", interactive=False)

            with gr.Column(scale=3):
                image = gr.Image(type="pil", label="Input image", sources=["upload", "clipboard", "webcam"])                
                with gr.Row():
                    predict_btn = gr.Button("Predict", variant="primary")
                    explain_btn = gr.Button("Explain", variant="secondary")
                    export_btn = gr.Button("Export evidence", variant="secondary")

                label_out = gr.Textbox(label="Label / Unsure", interactive=False)
                top3_out = gr.Textbox(label="Top-3", interactive=False)
                coarse_out = gr.JSON(label="Coarse probs", value={})
                status_out = gr.Textbox(label="Status", interactive=False)
                heatmap_out = gr.Image(type="pil", label="Heatmap", interactive=False)
                heatmap_status = gr.Textbox(label="Heatmap Status", interactive=False)


        # events
        refresh_btn.click(fn=ui_refresh_artifacts, outputs=artifact)
        predict_btn.click(fn=ui_predict,
                  inputs=[artifact, unsure, coarse, tstar, image],
                  outputs=[label_out, top3_out, coarse_out, status_out])
        export_btn.click(fn=ui_export,
                         inputs=[image, label_out, top3_out, coarse_out, status_out],
                         outputs=[status_out])
        explain_btn.click(
                    fn=ui_explain,
                    inputs=[artifact, arch_name, classes_txt, tstar, image],
                    outputs=[heatmap_out, heatmap_status])



    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
