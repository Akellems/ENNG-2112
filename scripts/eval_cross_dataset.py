
# ─────────────────────────────────────────────────────────────
# File: scripts/eval_cross_dataset.py
# Purpose: Cross-dataset evaluation D1→D2 and D2→D1; compute new-data drop
# Output: metrics/metrics_cross_dataset.json
# ─────────────────────────────────────────────────────────────

import argparse, json, os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

from train_baseline import accuracy, macro_f1

@torch.no_grad()
def eval_model(model, loader, device, num_classes):
    model.eval(); acc=0; mf1=0; n=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        acc += accuracy(logits, y)*x.size(0)
        mf1 += macro_f1(logits, y, num_classes)*x.size(0)
        n += x.size(0)
    return acc/n, mf1/n

def load_ckpt(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu')
    model = timm.create_model(ck['model'], pretrained=False, num_classes=len(ck['classes']))
    model.load_state_dict(ck['state_dict'])
    return model, ck['classes']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--d1', type=str, required=True, help='dataset root #1 with val/test')
    ap.add_argument('--d2', type=str, required=True, help='dataset root #2 with val/test')
    ap.add_argument('--split', type=str, default='test', choices=['val','test'])
    ap.add_argument('--img', type=int, default=224)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tfm = transforms.Compose([
        transforms.Resize((args.img,args.img)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])

    model, classes = load_ckpt(args.ckpt); model.to(device)
    ncls = len(classes)

    d1 = datasets.ImageFolder(os.path.join(args.d1, args.split), tfm)
    d2 = datasets.ImageFolder(os.path.join(args.d2, args.split), tfm)
    l1 = DataLoader(d1, batch_size=64, shuffle=False)
    l2 = DataLoader(d2, batch_size=64, shuffle=False)

    acc11, mf111 = eval_model(model, l1, device, ncls)
    acc12, mf112 = eval_model(model, l2, device, ncls)

    # Train-on-1 test-on-2 drop is approximated by (perf on D2 - perf on D1)
    drop12 = float(acc11 - acc12)
    md = {
        'ckpt': args.ckpt,
        'D1': args.d1,
        'D2': args.d2,
        'split': args.split,
        'acc_D1': acc11,
        'acc_D2': acc12,
        'acc_drop_1to2': drop12,
        'note': 'If you also have a ckpt trained on D2, run again swapping roles to get 2→1.'
    }
    Path('metrics').mkdir(exist_ok=True)
    with open('metrics/metrics_cross_dataset.json','w') as f:
        json.dump(md, f, indent=2)

if __name__ == '__main__':
    main()
