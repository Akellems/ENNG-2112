
# ─────────────────────────────────────────────────────────────
# File: scripts/benchmark_latency_size.py
# Purpose: Measure mean/p90 latency and file sizes for models
# Output: metrics/latency_size_comparison.csv
# ─────────────────────────────────────────────────────────────

import argparse, os, time, statistics
from pathlib import Path
import torch, timm


def run_once(model, x, warmup=3, runs=50):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        times=[]
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            times.append((time.perf_counter()-t0)*1000)
    return statistics.mean(times), statistics.quantiles(times, n=10)[8]  # p90 approx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True, help='list of .pt checkpoints')
    ap.add_argument('--img', type=int, default=224)
    ap.add_argument('--out', type=str, default='metrics/latency_size_comparison.csv')
    args = ap.parse_args()

    device = torch.device('cpu')  # measure on CPU for fairness
    x = torch.randn(1,3,args.img,args.img)

    rows = ["ckpt,model,params_m,pt_size_mb,mean_ms,p90_ms"]

    for ckpt_path in args.ckpts:
        ck = torch.load(ckpt_path, map_location='cpu')
        model = timm.create_model(ck['model'], pretrained=False, num_classes=len(ck['classes']))
        model.load_state_dict(ck['state_dict']); model.to(device)

        mean_ms, p90_ms = run_once(model, x)
        params_m = sum(p.numel() for p in model.parameters())/1e6
        pt_size_mb = os.path.getsize(ckpt_path)/1e6
        rows.append(f"{ckpt_path},{ck['model']},{params_m:.2f},{pt_size_mb:.2f},{mean_ms:.1f},{p90_ms:.1f}")

    Path('metrics').mkdir(exist_ok=True)
    with open(args.out,'w') as f:
        f.write("\n".join(rows))

if __name__ == '__main__':
    main()

