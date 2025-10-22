# scripts/prepare_split_from_csv.py  鈥斺€�  CSV鏍囨敞鐗�
import argparse, csv, random, json, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of images")
    ap.add_argument("--labels", required=True, help="labels.csv with filename,label")
    ap.add_argument("--dst", default="data_split", help="output root")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "splits must sum to 1.0"
    random.seed(args.seed)

    img_root = Path(args.images)
    dst = Path(args.dst)
    (dst/"train").mkdir(parents=True, exist_ok=True)
    (dst/"val").mkdir(exist_ok=True)
    (dst/"test").mkdir(exist_ok=True)

    # 璇诲彇 labels.csv锛堜袱鍒楋細filename,label锛�
    by_class = {}
    with open(args.labels, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get("filename")
            lb = row.get("label")
            if not fn or not lb:
                raise ValueError("labels.csv must have columns: filename,label")
            by_class.setdefault(lb, []).append(img_root / fn)

    manifest = {
        "seed": args.seed,
        "splits": {"train": args.train, "val": args.val, "test": args.test},
        "classes": {}
    }

    for cls, paths in by_class.items():
        paths = [p for p in paths if p.exists()]
        random.shuffle(paths)
        n = len(paths)
        n_tr = int(n * args.train)
        n_val = int(n * args.val)
        n_te = n - n_tr - n_val

        splits = {
            "train": paths[:n_tr],
            "val": paths[n_tr:n_tr+n_val],
            "test": paths[n_tr+n_val:]
        }
        manifest["classes"][cls] = {k: len(v) for k, v in splits.items()}

        for sp, files in splits.items():
            out = dst / sp / cls
            out.mkdir(parents=True, exist_ok=True)
            for fp in files:
                dst_fp = out / fp.name
                try:
                    if args.copy:
                        shutil.copy2(fp, dst_fp)
                    else:
                        dst_fp.symlink_to(fp.resolve())
                except Exception:
                    shutil.copy2(fp, dst_fp)

    (dst / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Done ->", dst.resolve())

if __name__ == "__main__":
    main()
