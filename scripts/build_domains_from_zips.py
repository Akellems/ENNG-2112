# scripts/build_domains_from_zips.py
# Robust builder with verbose logs:
# - 从 WCD zip 自动搜寻包含 /organic/ 与 /recyclable/ 的文件路径
# - 从 trashnet-master.zip 自动定位 data/dataset-resized.zip
# - 全程打印拷贝统计，出问题直接报错退出

import zipfile, io, os, shutil, argparse, sys
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def is_img(name: str) -> bool:
    return name.lower().endswith(IMG_EXTS)

def copy_from_zip(zf: zipfile.ZipFile, member: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst)

def build_wcd(wcd_zip: str, out_root: Path):
    print(f"[*] Open WCD zip: {wcd_zip}")
    with zipfile.ZipFile(wcd_zip) as z:
        names = z.namelist()
        print(f"[*] WCD entries: {len(names)}")
        count_org = count_rec = 0

        for name in names:
            ln = name.lower()
            if not is_img(ln):
                continue
            if "/organic/" in ln:
                copy_from_zip(z, name, out_root/"organic"/os.path.basename(name))
                count_org += 1
            elif "/recyclable/" in ln:
                copy_from_zip(z, name, out_root/"recyclable"/os.path.basename(name))
                count_rec += 1

        if count_org == 0 and count_rec == 0:
            raise SystemExit("!! 没在 WCD zip 里找到含 /organic/ 或 /recyclable/ 的图片路径。请把 zip 的前缀结构再发我。")

        print(f"[OK] WCD copied: organic={count_org}, recyclable={count_rec} -> {out_root}")
        return count_org, count_rec

def build_trashnet(trashnet_zip: str, out_root: Path):
    print(f"[*] Open TrashNet zip: {trashnet_zip}")
    with zipfile.ZipFile(trashnet_zip) as z:
        inner_candidates = [p for p in z.namelist() if p.lower().endswith("dataset-resized.zip")]
        print(f"[*] Inner dataset-resized.zip candidates: {inner_candidates[:3]}")
        if not inner_candidates:
            raise SystemExit("!! 未在 trashnet zip 中找到 dataset-resized.zip")
        inner = inner_candidates[0]

        with zipfile.ZipFile(io.BytesIO(z.read(inner))) as dz:
            names = dz.namelist()
            print(f"[*] TrashNet inner entries: {len(names)}")
            count_rec = count_trash = 0
            for name in names:
                if not is_img(name):
                    continue
                parts = name.split("/")
                if len(parts) < 3:  # dataset-resized/<class>/file.jpg
                    continue
                cls = parts[1].lower()
                if cls in {"cardboard", "glass", "metal", "paper", "plastic"}:
                    copy_from_zip(dz, name, out_root/"recyclable"/os.path.basename(name))
                    count_rec += 1
                elif cls == "trash":
                    copy_from_zip(dz, name, out_root/"trash"/os.path.basename(name))
                    count_trash += 1

            if count_rec == 0 and count_trash == 0:
                raise SystemExit("!! TrashNet 内部未检测到预期类名（cardboard/glass/metal/paper/plastic/trash）。")

            print(f"[OK] TrashNet copied: recyclable={count_rec}, trash={count_trash} -> {out_root}")
            return count_rec, count_trash

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wcd_zip", default="Waste Classification Dataset.zip")
    ap.add_argument("--trashnet_zip", default="trashnet-master.zip")
    ap.add_argument("--out_wcd", default="data_raw_wcd")
    ap.add_argument("--out_trashnet", default="data_raw_trashnet")
    args = ap.parse_args()

    Path(args.out_wcd).mkdir(parents=True, exist_ok=True)
    Path(args.out_trashnet).mkdir(parents=True, exist_ok=True)

    a, b = build_wcd(args.wcd_zip, Path(args.out_wcd))
    c, d = build_trashnet(args.trashnet_zip, Path(args.out_trashnet))

    print(f"[DONE] WCD(organic={a}, recyclable={b}) | TrashNet(recyclable={c}, trash={d})")

if __name__ == "__main__":
    sys.exit(main())
