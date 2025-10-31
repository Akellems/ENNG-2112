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


def extract_model_arch_name(filepath: str) -> str:
    """
    Extract the model architecture name from a filepath.
    E.g., vit_small.pt -> vit_small
          vit_small_trashnet.pt -> vit_small
          mobilenetv3_small_realwaste.pt -> mobilenetv3_small
    """
    filename = Path(filepath).stem  # remove extension
    # Remove common dataset suffixes
    for suffix in ["_trashnet", "_realwaste", "_kaggle"]:
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)]
            break
    return filename


def group_artifacts_by_arch(max_depth: int = 2) -> Dict[str, List[str]]:
    """
    Group model files by architecture name for ensemble inference.
    Returns: {arch_name: [path1, path2, ...]}
    E.g., {"vit_small": ["vit_small.pt", "vit_small_trashnet.pt", "vit_small_realwaste.pt"]}
    """
    artifacts = discover_artifacts(max_depth)
    grouped = {}
    
    for art in artifacts:
        arch = extract_model_arch_name(art)
        if arch not in grouped:
            grouped[arch] = []
        grouped[arch].append(art)
    
    return grouped


def group_artifacts_by_dataset(max_depth: int = 2) -> Dict[str, List[str]]:
    """
    Group model files by dataset suffix for compatible ensembling.
    Returns: {dataset_name: [path1, path2, ...]}
    E.g., {"kaggle": ["vit_small.pt", "mobilenetv3_small.pt"], 
           "trashnet": ["vit_small_trashnet.pt", ...]}
    """
    artifacts = discover_artifacts(max_depth)
    grouped = {"kaggle": [], "trashnet": [], "realwaste": [], "mixed": []}
    
    for art in artifacts:
        filename = Path(art).stem
        if "_trashnet" in filename:
            grouped["trashnet"].append(art)
        elif "_realwaste" in filename:
            grouped["realwaste"].append(art)
        elif any(x in filename for x in ["_kaggle", "vit_small", "vit_tiny", "mobilenetv3_small"]) and "_trashnet" not in filename and "_realwaste" not in filename:
            grouped["kaggle"].append(art)
        else:
            grouped["mixed"].append(art)
    
    # Remove empty groups
    return {k: v for k, v in grouped.items() if v}

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
    # Match training preprocessing: Resize(256) -> CenterCrop(224)
    # This preserves aspect ratio and prevents distortion
    if not _has_torch:
        raise RuntimeError("torch is required for PyTorch models.")
    
    img = img.convert("RGB")
    
    # Step 1: Resize shorter edge to 256 (preserves aspect ratio)
    w, h = img.size
    if w < h:
        new_w, new_h = 256, int(256 * h / w)
    else:
        new_w, new_h = int(256 * w / h), 256
    img = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Step 2: Center crop to 224x224
    left = (new_w - img_size) // 2
    top = (new_h - img_size) // 2
    img = img.crop((left, top, left + img_size, top + img_size))
    
    # Step 3: Convert to tensor with ImageNet normalization
    arr = np.asarray(img).astype("float32") / 255.0
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
    # Match training preprocessing: Resize(256) -> CenterCrop(224)
    img = img.convert("RGB")
    
    # Resize shorter edge to 256 (preserves aspect ratio)
    w, h = img.size
    if w < h:
        new_w, new_h = 256, int(256 * h / w)
    else:
        new_w, new_h = int(256 * w / h), 256
    img = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Center crop to 224x224
    left = (new_w - img_size) // 2
    top = (new_h - img_size) // 2
    img = img.crop((left, top, left + img_size, top + img_size))
    
    # Convert to tensor with ImageNet normalization
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


# ---------- Ensemble Prediction ----------

def predict_ensemble(arch_name: str,
                     model_paths: List[str],
                     img: Image.Image,
                     user_arch_name: str,
                     classes_text: str,
                     Tstar: float = 1.0) -> Tuple[Dict[str, float], List[str], str]:
    """
    Ensemble prediction: run inference on all models and aggregate coarse predictions.
    
    Returns:
        - aggregated_coarse_probs: Dict mapping {"Recyclable", "Organic", "Trash"} to probabilities
        - model_info: List of model names/paths used
        - status_msg: String with timing and model info
    """
    if not model_paths:
        raise ValueError("No model paths provided for ensemble")
    
    all_coarse_probs = []
    successful_models = []
    errors = []
    
    t0 = time.perf_counter()
    
    for path in model_paths:
        try:
            # Load and predict with this model
            model, classes, arch_used = build_model(path, user_arch_name, classes_text)
            
            if model.is_torch:
                probs, classes = predict_torch(model.torch_pack, img, Tstar=Tstar)
            else:
                probs, classes = predict_onnx(model.onnx_pack, img, Tstar=Tstar)
            
            # Map to coarse categories
            coarse_map = default_coarse_map(classes)
            coarse_probs = aggregate_coarse(probs, classes, coarse_map)
            
            # Ensure all 3 categories exist
            for k in ["Recyclable", "Organic", "Trash"]:
                coarse_probs.setdefault(k, 0.0)
            
            all_coarse_probs.append(coarse_probs)
            successful_models.append(Path(path).name)
            
        except Exception as e:
            errors.append(f"{Path(path).name}: {str(e)[:50]}")
            continue
    
    if not all_coarse_probs:
        raise RuntimeError(f"All models failed. Errors: {'; '.join(errors)}")
    
    # Average the coarse probabilities across all models
    ensemble_probs = {"Recyclable": 0.0, "Organic": 0.0, "Trash": 0.0}
    for coarse_prob in all_coarse_probs:
        for k in ensemble_probs:
            ensemble_probs[k] += coarse_prob[k]
    
    # Normalize by number of successful models
    n_models = len(all_coarse_probs)
    for k in ensemble_probs:
        ensemble_probs[k] /= n_models
    
    dt_ms = int((time.perf_counter() - t0) * 1000)
    
    status = f"ensemble={arch_name} | models={n_models}/{len(model_paths)} | latency={dt_ms}ms"
    if errors:
        status += f" | errors={len(errors)}"
    
    return ensemble_probs, successful_models, status

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

def ui_refresh_artifacts(mode: str = "individual") -> List[str]:
    """
    Return list of models for dropdown based on mode.
    mode: 'individual', 'ensemble_arch', or 'ensemble_dataset'
    """
    if mode == "individual":
        # List all individual models
        artifacts = discover_artifacts(max_depth=2)
        if not artifacts:
            return ["<No model artifacts found>"]
        return [str(Path(p).name) for p in artifacts]
    
    elif mode == "ensemble_dataset":
        # Group by dataset (compatible classes)
        grouped = group_artifacts_by_dataset(max_depth=2)
        if not grouped:
            return ["<No model artifacts found>"]
        choices = []
        for dataset, paths in sorted(grouped.items()):
            choices.append(f"Dataset: {dataset} ({len(paths)} model{'s' if len(paths) > 1 else ''})")
        return choices
    
    else:  # ensemble_arch
        # Group by architecture (cross-dataset - may have incompatible classes!)
        grouped = group_artifacts_by_arch(max_depth=2)
        if not grouped:
            return ["<No model artifacts found>"]
        choices = []
        for arch, paths in sorted(grouped.items()):
            choices.append(f"Arch: {arch} ({len(paths)} model{'s' if len(paths) > 1 else ''})")
        return choices

def ui_predict(inference_mode, artifact_choice, unsure_threshold, use_coarse, tstar, image):
    """Predict using selected model(s) based on inference mode"""
    if image is None:
        return ("No image.", "", {}, "")

    if "<No model artifacts found>" in artifact_choice:
        return ("No models found. Click 🔧 to rescan.", "", {}, "")

    try:
        model_paths = []
        ensemble_name = ""
        
        if inference_mode == "Individual Model":
            # Single model inference
            all_artifacts = discover_artifacts(max_depth=2)
            matching = [p for p in all_artifacts if Path(p).name == artifact_choice]
            if not matching:
                return ("Model file not found. Click 🔧 to rescan.", "", {}, "")
            model_paths = [matching[0]]
            ensemble_name = Path(matching[0]).stem
            
        elif inference_mode == "Ensemble by Dataset (Recommended)":
            # Extract dataset name (e.g., "Dataset: kaggle (3 models)" -> "kaggle")
            dataset_name = artifact_choice.split("Dataset: ")[1].split(" (")[0]
            grouped = group_artifacts_by_dataset(max_depth=2)
            if dataset_name not in grouped:
                return ("Dataset not found. Click 🔧 to rescan.", "", {}, "")
            model_paths = grouped[dataset_name]
            ensemble_name = f"dataset_{dataset_name}"
            
        else:  # Ensemble by Architecture
            # Extract arch name (e.g., "Arch: vit_small (3 models)" -> "vit_small")
            arch_name = artifact_choice.split("Arch: ")[1].split(" (")[0]
            grouped = group_artifacts_by_arch(max_depth=2)
            if arch_name not in grouped:
                return ("Architecture not found. Click 🔧 to rescan.", "", {}, "")
            model_paths = grouped[arch_name]
            ensemble_name = f"arch_{arch_name}"
        
        # Run prediction (single or ensemble)
        if len(model_paths) == 1:
            # Single model
            model, classes, arch_used = build_model(
                model_paths[0],
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
            
            status = f"single={arch_used} | latency={dt_ms}ms"
            successful_models = [Path(model_paths[0]).name]
        else:
            # Ensemble
            coarse_probs, successful_models, status = predict_ensemble(
                arch_name=ensemble_name,
                model_paths=model_paths,
                img=image,
                user_arch_name=DEFAULT_TORCH_MODEL_NAME,
                classes_text="\n".join(DEFAULT_CLASSES),
                Tstar=tstar
            )
        
        # Determine final prediction
        ranked = sorted(coarse_probs.items(), key=lambda x: -x[1])
        max_bucket, max_p = ranked[0][0], float(ranked[0][1])
        unsure = (max_p < unsure_threshold)

        label_text = max_bucket if not unsure else f"Unsure (< {unsure_threshold:.2f})"
        top3_text = "\n".join([f"{k}: {v:.3f}" for k, v in ranked])
        
        # Add model info
        if len(successful_models) > 1:
            top3_text += f"\n\n📊 Ensemble ({len(successful_models)} models):\n" + "\n".join([f"  • {m}" for m in successful_models])
        else:
            top3_text += f"\n\n📊 Model: {successful_models[0]}"

        status += f" | coarse_maxP={max_p:.3f} | unsure={unsure}"

        return (label_text,
                top3_text,
                {"Recyclable": float(coarse_probs["Recyclable"]),
                 "Organic": float(coarse_probs["Organic"]),
                 "Trash": float(coarse_probs["Trash"])},
                status)

    except Exception as e:
        trace = traceback.format_exc()
        return (f"Error: {e}", "", {}, trace)


def ui_explain(artifact_choice: str,
               arch_name_text: str,
               classes_text: str,
               tstar: float,
               image: Image.Image) -> Tuple[Image.Image | None, str]:
    """Generate heatmap using the first/selected model"""
    import traceback

    def _ok(img, msg):   return (img, f"OK: {msg}")
    def _err(img, msg):  return (img, f"ERROR: {msg}")

    try:
        if image is None:
            return _err(None, "No image provided to Explain().")

        if "<No model artifacts found>" in artifact_choice:
            return _err(image, "No models found. Click Rescan.")

        # Determine model path based on choice format
        model_path = None
        
        if artifact_choice.startswith("Dataset: "):
            # Extract dataset and use first model
            dataset_name = artifact_choice.split("Dataset: ")[1].split(" (")[0]
            grouped = group_artifacts_by_dataset(max_depth=2)
            if dataset_name in grouped:
                model_path = grouped[dataset_name][0]
        elif artifact_choice.startswith("Arch: "):
            # Extract architecture and use first model
            arch_name = artifact_choice.split("Arch: ")[1].split(" (")[0]
            grouped = group_artifacts_by_arch(max_depth=2)
            if arch_name in grouped:
                model_path = grouped[arch_name][0]
        else:
            # Individual model - find by filename
            all_artifacts = discover_artifacts(max_depth=2)
            matching = [p for p in all_artifacts if Path(p).name == artifact_choice]
            if matching:
                model_path = matching[0]
        
        if model_path is None:
            return _err(image, "Model not found. Click Rescan.")
        
        model_filename = Path(model_path).name
        model, classes, arch_used = build_model(model_path, arch_name_text, classes_text)

        if not model.is_torch:
            return _err(image, "ONNX model: heatmap not supported (needs PyTorch gradients).")

        # Grad-CAM
        try:
            overlay, cam = explain_torch_gradcam(model.torch_pack, image, img_size=224)
            return _ok(overlay, f"Grad-CAM on {arch_used} ({model_filename}) | cam.shape={getattr(cam, 'shape', None)}")
        except Exception as e1:
            msg1 = f"Grad-CAM failed: {repr(e1)}"

        # ViT Rollout
        try:
            overlay, att = explain_vit_rollout(model.torch_pack, image, img_size=224)
            return _ok(overlay, f"ViT Rollout on {arch_used} ({model_filename}) | att.shape={getattr(att, 'shape', None)} (fallback after Grad-CAM)")
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
                    "Upload an image of waste and classify it using trained models. "
                    "Choose between individual models or ensemble inference.")

        with gr.Row():
            with gr.Column(scale=2):
                # Inference Mode Selector
                inference_mode = gr.Radio(
                    choices=["Individual Model", "Ensemble by Dataset (Recommended)", "Ensemble by Architecture (Cross-Dataset)"],
                    value="Individual Model",
                    label="Inference Mode",
                    info="⚠️ Cross-dataset ensemble may give incorrect results if models have different class structures!"
                )
                
                artifact = gr.Dropdown(label="Select Model/Ensemble", choices=ui_refresh_artifacts("individual"), value=None)
                refresh_btn = gr.Button("🔧 Rescan models", variant="secondary")
                
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
                top3_out = gr.Textbox(label="Top-3", interactive=False, lines=8)
                coarse_out = gr.JSON(label="Coarse probs", value={})
                status_out = gr.Textbox(label="Status", interactive=False)
                heatmap_out = gr.Image(type="pil", label="Heatmap", interactive=False)
                heatmap_status = gr.Textbox(label="Heatmap Status", interactive=False)

        # Helper function to update dropdown based on mode
        def update_dropdown(mode):
            if mode == "Individual Model":
                return gr.update(choices=ui_refresh_artifacts("individual"), value=None, label="Select Model")
            elif mode == "Ensemble by Dataset (Recommended)":
                return gr.update(choices=ui_refresh_artifacts("ensemble_dataset"), value=None, label="Select Dataset Ensemble")
            else:
                return gr.update(choices=ui_refresh_artifacts("ensemble_arch"), value=None, label="Select Architecture Ensemble")

        # events
        inference_mode.change(fn=update_dropdown, inputs=[inference_mode], outputs=[artifact])
        refresh_btn.click(fn=update_dropdown, inputs=[inference_mode], outputs=[artifact])
        predict_btn.click(fn=ui_predict,
                  inputs=[inference_mode, artifact, unsure, coarse, tstar, image],
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
