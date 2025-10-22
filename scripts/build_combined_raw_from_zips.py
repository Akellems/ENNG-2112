# scripts/build_combined_raw_from_zips.py  (v2: auto-detect + verbose)
import zipfile, io, os, shutil, argparse, sys
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def log(*a): print("[build]", *a)

def safe_copy(zf: zipfile.ZipFile, member: str, dest_dir: Path, prefix: str):
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = os.path.basename(member)
    # 避免重名：加上来源前缀
    out = dest_dir / f"{prefix}_{name}"
    with zf.open(member) as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst)

def find_dirs_in_zip(z: zipfile.ZipFile, wanted=("organic","recyclable")):
    """在 zip 中自动寻找包含 wanted 名称的目录前缀，返回 {label: [prefixes]}"""
    found = {w: set() for w in wanted}
    for name in z.namelist():
        low = name.lower().replace("\\", "/")
        parts = low.split("/")
        # 粗暴匹配：路径片段包含 wanted 的目录名
        for w in wanted:
            if f"/{w}/" in low or low.endswith(f"/{w}/"):
                # 拿到从 zip 根到 w 的父目录作为前缀
                idx = parts.index(w)
                prefix = "/".join(parts[:idx+1])  # .../organic
                found[w].add(prefix)
    return {k: sorted(v) for k,v in found.items() if v}

def find_inner_zip(z: zipfile.ZipFile, name_hint="dataset-resized", ext=".zip"):
    for name in z.namelist():
        low = name.lower().replace("\\","/")
        if name_hint in low and low.endswith(ext):
            return name
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wcd_zip", default="Waste Classification Dataset.zip")
    ap.add_argument("--trashnet_zip", default="trashnet-master.zip")
    ap.add_argument("--out", default="data_raw_combined")
    ap.add_argument("--dry_run", action="store_true", help="only print what would be done")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "organic").mkdir(parents=True, exist_ok=True)
    (out / "recyclable").mkdir(parents=True, exist_ok=True)
    (out / "trash").mkdir(parents=True, exist_ok=True)

    # ---- 1) 处理 Waste Classification Dataset.zip ----
    if not Path(args.wcd_zip).exists():
        log("WCD zip NOT FOUND:", args.wcd_zip)
    else:
        log("Opening WCD zip:", args.wcd_zip)
        with zipfile.ZipFile(args.wcd_zip) as z:
            dirs = find_dirs_in_zip(z, wanted=("organic","recyclable"))
            if not dirs:
                log("WARN: Could not auto-detect 'organic/recyclable' folders in WCD zip. Printing first 20 names:")
                for i,name in enumerate(z.namelist()[:20]): print("  ", name)
            else:
                log("Detected folders:", dirs)
                count = {"organic":0, "recyclable":0}
                for name in z.namelist():
                    low = name.lower().replace("\\","/")
                    if not low.endswith(IMG_EXTS): 
                        continue
                    for w in ("organic","recyclable"):
                        if any(low.startswith(pref+"/") for pref in dirs.get(w, [])):
                            if not args.dry_run:
                                safe_copy(z, name, out / w, "wcd")
                            count[w]+=1
                log(f"WCD copied -> organic:{count['organic']} recyclable:{count['recyclable']}")

    # ---- 2) 处理 trashnet-master.zip ----
    if not Path(args.trashnet_zip).exists():
        log("TrashNet zip NOT FOUND:", args.trashnet_zip)
    else:
        log("Opening TrashNet zip:", args.trashnet_zip)
        with zipfile.ZipFile(args.trashnet_zip) as z:
            inner = find_inner_zip(z, "dataset-resized")
            if not inner:
                log("WARN: Cannot find dataset-resized.zip in TrashNet; printing first 20 names:")
                for i,name in enumerate(z.namelist()[:20]): print("  ", name)
            else:
                log("Found inner zip:", inner)
                with zipfile.ZipFile(io.BytesIO(z.read(inner))) as dz:
                    cnt_rec, cnt_trash = 0, 0
                    # 6 类：cardboard/glass/metal/paper/plastic/trash
                    for name in dz.namelist():
                        low = name.lower().replace("\\","/")
                        if not low.endswith(IMG_EXTS): 
                            continue
                        parts = [p for p in low.split("/") if p]
                        if len(parts) < 2: 
                            continue
                        cls = parts[1]  # dataset-resized/<cls>/file
                        if cls in {"cardboard","glass","metal","paper","plastic"}:
                            if not args.dry_run:
                                safe_copy(dz, name, out / "recyclable", f"tn_{cls}")
                            cnt_rec += 1
                        elif cls == "trash":
                            if not args.dry_run:
                                safe_copy(dz, name, out / "trash", "tn_trash")
                            cnt_trash += 1
                    log(f"TrashNet copied -> recyclable:{cnt_rec} trash:{cnt_trash}")

    # ---- 3) labels_mapping.csv ----
    mapping = (
        "fine_label,coarse_label\n"
        "organic,organic\n"
        "recyclable,recyclable\n"
        "cardboard,recyclable\n"
        "glass,recyclable\n"
        "metal,recyclable\n"
        "paper,recyclable\n"
        "plastic,recyclable\n"
        "trash,trash\n"
    )
    if not args.dry_run:
        (out / "labels_mapping.csv").write_text(mapping, encoding="utf-8")
        log("Wrote labels_mapping.csv")
    log("DONE. Output root:", out.resolve())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[build][ERROR]", repr(e))
        sys.exit(1)
