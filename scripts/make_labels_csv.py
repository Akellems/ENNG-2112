# scripts/make_labels_csv.py
import argparse, csv, re
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def infer_label_from_name(name: str) -> str:
    # 规则：取第一个下划线前的部分作为标签，如 glass_001.jpg -> glass
    m = re.match(r"([A-Za-z0-9\-]+)_", name)
    return m.group(1).lower() if m else "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of images")
    ap.add_argument("--out", default="data_raw/labels.csv", help="output CSV path")
    args = ap.parse_args()

    img_root = Path(args.images)
    if not img_root.exists():
        raise FileNotFoundError(f"Images folder not found: {img_root.resolve()}")

    # 检查是否有类别子文件夹
    subdirs = [p for p in img_root.iterdir() if p.is_dir()]
    rows = []

    if subdirs:
        # 目录结构：images/<label>/*.jpg
        for cls_dir in sorted(subdirs):
            label = cls_dir.name
            for f in cls_dir.rglob("*"):
                if f.is_file() and is_image(f):
                    rel = f.relative_to(img_root).as_posix()
                    rows.append((rel, label))
    else:
        # 扁平结构：images/*.jpg —— 用文件名前缀猜标签
        for f in img_root.glob("*"):
            if f.is_file() and is_image(f):
                label = infer_label_from_name(f.name)
                rows.append((f.name, label))

    if not rows:
        raise RuntimeError("No images found to index. Check your --images path.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["filename","label"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out.resolve()}")

if __name__ == "__main__":
    main()
