# File: scripts/train_baseline.py
# Purpose: Train a baseline classifier (MobileNetV3 / DeiT-Tiny) with unified CLI

import argparse, json, os, time, math
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

# ---------- Losses ----------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.softmax(logits, dim=1).gather(1, targets.view(-1,1)).squeeze(1)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction=='mean' else loss.sum()


# ---------- Utils ----------
def save_ckpt(path: Path, model_name: str, model: nn.Module, classes: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
    'model': model_name,
    'state_dict': model.state_dict(),
    'classes': classes,
    }, path)


def accuracy(logits, targets):
    preds = logits.argmax(1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def macro_f1(logits, targets, num_classes):
    preds = logits.argmax(1)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        cm[t, p] += 1
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2*prec*rec / (prec+rec+1e-9)
        f1s.append(f1)
    return float(sum(f1s)/len(f1s))

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='dataset root with train/val folders')
    ap.add_argument('--out', type=str, default='checkpoints/baseline.pt')
    ap.add_argument('--model', type=str, default='mobilenetv3_small_100',
    choices=['mobilenetv3_small_100','mobilenetv3_large_100','deit_tiny_patch16_224'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--loss', type=str, default='ce', choices=['ce','ce_ls','focal'])
    ap.add_argument('--img', type=int, default=224)
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # transforms (keep in sync with app.py inference)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tf_train = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)
    ])
    tf_val = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.ToTensor(), transforms.Normalize(mean, std)
    ])


    train_set = datasets.ImageFolder(os.path.join(args.data, 'train'), tf_train)
    val_set = datasets.ImageFolder(os.path.join(args.data, 'val'), tf_val)
    classes = train_set.classes


    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)


    # model
    model = timm.create_model(args.model, pretrained=True, num_classes=len(classes))
    model.to(device)
    
    # loss
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'ce_ls':
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    else:
        class_counts = torch.tensor([0]*len(classes))
        for _, y in train_set:
            class_counts[y] += 1
            weights = 1.0 / (class_counts.float() + 1e-6)
            weights = len(classes) * weights / weights.sum()
            criterion = FocalLoss(gamma=2.0, alpha=weights.to(device))


    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


    best_val = -1.0
    metrics_dir = Path('metrics'); metrics_dir.mkdir(exist_ok=True)


    for epoch in range(1, args.epochs+1):
        model.train(); loss_sum=0; acc_sum=0; n=0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            loss_sum += loss.item()*x.size(0)
            acc_sum += accuracy(logits.detach(), y)*x.size(0)
            n += x.size(0)
        scheduler.step()


# val
        model.eval(); val_acc=0; val_mf1=0; vn=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                val_acc += accuracy(logits, y)*x.size(0)
                val_mf1 += macro_f1(logits, y, len(classes))*x.size(0)
                vn += x.size(0)
        val_acc/=vn; val_mf1/=vn
        print(f"Epoch {epoch}: val_acc={val_acc:.4f} mf1={val_mf1:.4f}")


        if val_mf1 > best_val:
            best_val = val_mf1
            save_ckpt(Path(args.out), args.model, model, classes)
            with open(metrics_dir/"metrics_baseline.json", 'w') as f:
                json.dump({'val_acc': val_acc, 'val_macro_f1': val_mf1, 'epoch': epoch,
                            'model': args.model, 'loss': args.loss}, f, indent=2)


if __name__ == '__main__':
    main()  