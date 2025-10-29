import argparse, random, json, shutil
from pathlib import Path

def is_image(p: Path):
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source root, e.g., data_raw/")
    ap.add_argument("--dst", default="data_split", help="output root with train/val/test")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlink")
    args = ap.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "splits must sum to 1.0"
    random.seed(args.seed)

    src = Path(args.src)
    dst = Path(args.dst)
    (dst / "train").mkdir(parents=True, exist_ok=True)
    (dst / "val").mkdir(parents=True, exist_ok=True)
    (dst / "test").mkdir(parents=True, exist_ok=True)

    manifest = {"seed": args.seed, "splits": {"train": args.train, "val": args.val, "test": args.test}, "classes": {}}

    # 閬嶅巻绫诲埆鏂囦欢澶�
    for cls_dir in sorted([p for p in src.iterdir() if p.is_dir()]):
        cls = cls_dir.name
        imgs = sorted([p for p in cls_dir.rglob("*") if p.is_file() and is_image(p)])
        if not imgs:
            print(f"[WARN] empty class: {cls}")
            continue
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        n_test = n - n_train - n_val

        splits = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train+n_val],
            "test": imgs[n_train+n_val:],
        }
        manifest["classes"][cls] = {k: len(v) for k, v in splits.items()}

        for split, files in splits.items():
            out_dir = dst / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for fp in files:
                dst_fp = out_dir / fp.name
                if args.copy or (hasattr(Path, "symlink_to") is False):
                    shutil.copy2(fp, dst_fp)
                else:
                    try:
                        dst_fp.symlink_to(fp.resolve())
                    except Exception:
                        shutil.copy2(fp, dst_fp)

    # 璁板綍涓€娆℃€ф竻鍗曪紝淇濊瘉鍙鐜�
    (dst / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Output -> {dst}")

if __name__ == "__main__":
    main()
