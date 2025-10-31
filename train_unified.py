#!/usr/bin/env python3
"""
Training script for unified merged dataset

Run after: python merge_datasets.py

This will train models on ~22,000 images from 3 combined datasets!
"""

from __future__ import annotations
import random
from pathlib import Path
import platform

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ====== Config ======
DATA_ROOT = Path("merged_dataset")
SEED = 56
EPOCHS = 20
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 0.05

# ====== Setup ======
def set_seed(seed: int = 56):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_num_workers() -> int:
    return 0 if platform.system() == 'Windows' else 4

set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Using device: {device}")

# ====== Load Dataset ======
if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"Merged dataset not found at {DATA_ROOT}.\\n"
        "Please run: python merge_datasets.py first!"
    )

print(f"[OK] Using dataset: {DATA_ROOT}")

def is_img(p):
    return str(p).lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))

full = datasets.ImageFolder(root=str(DATA_ROOT), transform=None, is_valid_file=is_img)
print(f"[OK] Found {len(full.classes)} classes: {full.classes}")

targets = [y for _, y in full.samples]
print(f"[OK] Total images: {len(targets)}")

# Class distribution
print("\\n[*] Class distribution:")
for i, cls in enumerate(full.classes):
    count = sum(1 for t in targets if t == i)
    print(f"  {cls:15s}: {count:5d} images")

# ====== Train/Val/Test Split ======
tr_val_idx, te_idx = train_test_split(
    np.arange(len(targets)), test_size=0.1, random_state=SEED, stratify=targets
)

tr_targets = [targets[i] for i in tr_val_idx]
tr_idx, va_idx = train_test_split(
    tr_val_idx, test_size=0.222, random_state=SEED, stratify=tr_targets
)

print(f"\\n[OK] Train: {len(tr_idx)} | Val: {len(va_idx)} | Test: {len(te_idx)}")

# ====== Transforms & DataLoaders ======
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

val_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

train_ds = Subset(datasets.ImageFolder(str(DATA_ROOT), transform=train_tfms, is_valid_file=is_img), tr_idx)
val_ds = Subset(datasets.ImageFolder(str(DATA_ROOT), transform=val_tfms, is_valid_file=is_img), va_idx)
test_ds = Subset(datasets.ImageFolder(str(DATA_ROOT), transform=val_tfms, is_valid_file=is_img), te_idx)

# Weighted sampler for class balancing
train_targets = np.array([targets[i] for i in tr_idx])
counts = np.bincount(train_targets)
class_weights = 1.0 / np.clip(counts, 1, None)
sample_weights = class_weights[train_targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(tr_idx), replacement=True)

num_workers = get_num_workers()
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, 
                      num_workers=num_workers, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, 
                    num_workers=num_workers, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=128, shuffle=False,
                     num_workers=num_workers, pin_memory=True)

print(f"[OK] DataLoaders created")

# ====== Training Function ======
def train_model(model, model_name, train_dl, val_dl, epochs=EPOCHS, lr=LR, wd=WEIGHT_DECAY):
    from torch.amp import autocast, GradScaler
    
    print(f"\\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=3)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs-3)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[3])
    
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler('cuda', enabled=(device == "cuda"))
    best = {"f1": -1, "state": None, "epoch": 0}
    
    for ep in range(epochs):
        # Train
        model.train()
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=(device == "cuda")):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        
        # Validate
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device, non_blocking=True)
                logits = model(x)
                preds.append(logits.argmax(1).cpu())
                gts.append(y)
        
        p = torch.cat(preds).numpy()
        g = torch.cat(gts).numpy()
        f1 = f1_score(g, p, average="macro")
        acc = accuracy_score(g, p)
        
        if f1 > best["f1"]:
            best = {"f1": f1, "state": model.state_dict(), "epoch": ep+1}
        
        marker = " <- BEST" if f1 == best["f1"] else ""
        print(f"Epoch {ep+1:2d}/{epochs}: Acc {acc:.4f} | MacroF1 {f1:.4f}{marker}")
    
    print(f"\\n[OK] Best F1: {best['f1']:.4f} at epoch {best['epoch']}")
    model.load_state_dict(best["state"])
    return model, best["f1"]

# ====== Evaluation Function ======
def evaluate_on_test(model, test_dl, classes):
    model.eval()
    preds, gts = [], []
    
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds.append(logits.argmax(1).cpu())
            gts.append(y)
    
    p = torch.cat(preds).numpy()
    g = torch.cat(gts).numpy()
    
    f1 = f1_score(g, p, average="macro")
    acc = accuracy_score(g, p)
    
    print(f"\\n[*] Test Set: Acc={acc:.4f}, MacroF1={f1:.4f}")
    print("\\n" + classification_report(g, p, target_names=classes, digits=3))
    
    return acc, f1

# ====== Train Models ======
num_classes = len(full.classes)

# MobileNetV3-Small
mobilenet = timm.create_model("mobilenetv3_small_100", pretrained=True, 
                               drop_rate=0.2, drop_path_rate=0.1, num_classes=num_classes)
mobilenet_trained, mobilenet_f1 = train_model(mobilenet, "MobileNetV3-Small", train_dl, val_dl)

# ViT-Small
vit = timm.create_model("vit_small_patch16_224", pretrained=True,
                        drop_rate=0.2, drop_path_rate=0.1, num_classes=num_classes)
vit_trained, vit_f1 = train_model(vit, "ViT-Small", train_dl, val_dl)

# ====== Test Evaluation ======
print("\\n" + "="*70)
print("TEST SET EVALUATION")
print("="*70)

print("\\nMobileNetV3-Small:")
mobilenet_test_acc, mobilenet_test_f1 = evaluate_on_test(mobilenet_trained, test_dl, full.classes)

print("\\nViT-Small:")
vit_test_acc, vit_test_f1 = evaluate_on_test(vit_trained, test_dl, full.classes)

# ====== Save Models ======
print("\\n" + "="*70)
print("SAVING MODELS")
print("="*70)

mobilenet_ckpt = {
    "model": "mobilenetv3_small_100",
    "state_dict": mobilenet_trained.state_dict(),
    "classes": full.classes,
    "val_f1": mobilenet_f1,
    "test_f1": mobilenet_test_f1,
    "test_acc": mobilenet_test_acc,
    "dataset": "unified_merged",
    "num_images": len(targets)
}
torch.save(mobilenet_ckpt, "mobilenetv3_small_unified.pt")
print("[OK] Saved: mobilenetv3_small_unified.pt")

vit_ckpt = {
    "model": "vit_small_patch16_224",
    "state_dict": vit_trained.state_dict(),
    "classes": full.classes,
    "val_f1": vit_f1,
    "test_f1": vit_test_f1,
    "test_acc": vit_test_acc,
    "dataset": "unified_merged",
    "num_images": len(targets)
}
torch.save(vit_ckpt, "vit_small_unified.pt")
print("[OK] Saved: vit_small_unified.pt")

print("\\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"MobileNetV3-Small: Val F1={mobilenet_f1:.4f}, Test F1={mobilenet_test_f1:.4f}, Test Acc={mobilenet_test_acc:.4f}")
print(f"ViT-Small:         Val F1={vit_f1:.4f}, Test F1={vit_test_f1:.4f}, Test Acc={vit_test_acc:.4f}")
print("\\nYou can now use these models in app2.py!")
print("Select 'Individual Model' mode and choose *_unified.pt files")

