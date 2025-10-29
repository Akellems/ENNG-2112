# ─────────────────────────────────────────────────────────────
# File: scripts/calibrate_unsure.py
# Purpose: Temperature scaling calibration + threshold sweep for Unsure
# Output: metrics/metrics_calibration.json, metrics/unsure_sweep.json
# ─────────────────────────────────────────────────────────────

import argparse, json, os, numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

from train_baseline import accuracy

@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval(); logits_list=[]; targets_list=[]
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        logits_list.append(logits.cpu()); targets_list.append(y.cpu())
    return torch.cat(logits_list), torch.cat(targets_list)

@torch.no_grad()
def nll_criterion(logits, targets, T=1.0):
    logits = logits / T
    logp = torch.log_softmax(logits, dim=1)
    return -logp[torch.arange(targets.size(0)), targets].mean().item()

@torch.no_grad()
def ece_score(logits, targets, T=1.0, bins=15):
    logits = logits / T
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(1)
    acc = (pred==targets).float()
    ece=0.0
    edges = torch.linspace(0,1,bins+1)
    for i in range(bins):
        m = (conf>edges[i]) & (conf<=edges[i+1])
        if m.any():
            gap = acc[m].float().mean().item() - conf[m].float().mean().item()
            ece += abs(gap) * (m.float().mean().item())
    return float(ece)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--split', type=str, default='val', choices=['val','test'])
    ap.add_argument('--img', type=int, default=224)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tfm = transforms.Compose([
        transforms.Resize((args.img,args.img)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])

    ck = torch.load(args.ckpt, map_location='cpu')
    model = timm.create_model(ck['model'], pretrained=False, num_classes=len(ck['classes']))
    model.load_state_dict(ck['state_dict']); model.to(device)

    ds = datasets.ImageFolder(os.path.join(args.data, args.split), tfm)
    dl = DataLoader(ds, batch_size=64, shuffle=False)

    logits, targets = collect_logits(model, dl, device)

    # temperature search (simple grid)
    grid = np.linspace(0.5, 3.0, 26)
    best = (1.0, 1e9)
    for T in grid:
        nll = nll_criterion(logits, targets, T)
        if nll < best[1]:
            best = (float(T), float(nll))

    Tstar = best[0]
    ece_before = ece_score(logits, targets, 1.0)
    ece_after  = ece_score(logits, targets, Tstar)

    # threshold sweep (using calibrated probs)
    probs = torch.softmax(logits/Tstar, dim=1)
    conf, pred = probs.max(1)
    correct = (pred==targets)

    results = []
    for thr in np.linspace(0.3, 0.9, 13):
        unsure = conf < thr
        kept = ~unsure
        kept_acc = float(correct[kept].float().mean().item()) if kept.any() else 0.0
        unsure_rate = float(unsure.float().mean().item())
        results.append({'threshold': float(thr), 'kept_acc': kept_acc, 'unsure_rate': unsure_rate})

    Path('metrics').mkdir(exist_ok=True)
    with open('metrics/metrics_calibration.json','w') as f:
        json.dump({'T*': Tstar, 'ece_before': ece_before, 'ece_after': ece_after}, f, indent=2)
    with open('metrics/unsure_sweep.json','w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
