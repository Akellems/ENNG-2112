#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing script for merged waste classification dataset.

This script:
1. Takes the merged_dataset/ folder with 8 unified classes
2. Splits into train (85%), val (10%), test (5%)
3. Copies files to preprocessed_merged_dataset/ with proper structure
4. Ensures stratified sampling for balanced splits

Usage:
    python preprocess_merged_dataset.py
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_PATH = Path("merged_dataset")
OUTPUT_PATH = Path("preprocessed_merged_dataset")

# Split ratios
TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
TEST_RATIO = 0.05  # User requested 5% test

# Reproducibility
RANDOM_SEED = 56

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def set_seed(seed: int = 56):
    """Set random seed for reproducibility"""
    random.seed(seed)

def scan_source_dataset(source_path: Path) -> Dict[str, List[Path]]:
    """
    Scan the merged dataset and organize images by class.
    Returns: {"class_name": [image_path1, image_path2, ...]}
    """
    class_images = defaultdict(list)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")
    
    # Iterate through class directories
    for class_dir in source_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        
        # Find all image files
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            class_images[class_name].extend(class_dir.glob(f"*{ext}"))
    
    return dict(class_images)


def stratified_split(
    class_images: Dict[str, List[Path]], 
    train_ratio: float, 
    val_ratio: float, 
    test_ratio: float,
    seed: int = 56
) -> Tuple[Dict, Dict, Dict]:
    """
    Perform stratified split for each class.
    Returns: (train_dict, val_dict, test_dict)
    """
    random.seed(seed)
    
    train_split = {}
    val_split = {}
    test_split = {}
    
    for class_name, images in class_images.items():
        # Shuffle images
        images_shuffled = images.copy()
        random.shuffle(images_shuffled)
        
        total = len(images_shuffled)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_split[class_name] = images_shuffled[:train_end]
        val_split[class_name] = images_shuffled[train_end:val_end]
        test_split[class_name] = images_shuffled[val_end:]
    
    return train_split, val_split, test_split


def copy_split_to_output(
    split_dict: Dict[str, List[Path]], 
    output_path: Path, 
    split_name: str
) -> int:
    """
    Copy images from a split to the output directory.
    Returns: total number of images copied
    """
    split_path = output_path / split_name
    total_copied = 0
    
    print(f"\n  Copying {split_name} split...")
    
    for class_name, images in split_dict.items():
        class_path = split_path / class_name
        class_path.mkdir(parents=True, exist_ok=True)
        
        for img_path in tqdm(images, desc=f"    {class_name}", leave=False):
            try:
                dest_path = class_path / img_path.name
                
                if not dest_path.exists():
                    shutil.copy2(img_path, dest_path)
                total_copied += 1
            except Exception as e:
                print(f"      ⚠️  Failed to copy {img_path.name}: {e}")
    
    return total_copied


def generate_statistics(
    train_split: Dict, 
    val_split: Dict, 
    test_split: Dict
) -> Dict:
    """Generate detailed statistics about the splits"""
    stats = {
        "classes": {},
        "totals": {
            "train": 0,
            "val": 0,
            "test": 0,
            "total": 0
        }
    }
    
    all_classes = set(train_split.keys()) | set(val_split.keys()) | set(test_split.keys())
    
    for class_name in sorted(all_classes):
        train_count = len(train_split.get(class_name, []))
        val_count = len(val_split.get(class_name, []))
        test_count = len(test_split.get(class_name, []))
        total_count = train_count + val_count + test_count
        
        stats["classes"][class_name] = {
            "train": train_count,
            "val": val_count,
            "test": test_count,
            "total": total_count,
            "train_%": round(train_count / total_count * 100, 1) if total_count > 0 else 0,
            "val_%": round(val_count / total_count * 100, 1) if total_count > 0 else 0,
            "test_%": round(test_count / total_count * 100, 1) if total_count > 0 else 0,
        }
        
        stats["totals"]["train"] += train_count
        stats["totals"]["val"] += val_count
        stats["totals"]["test"] += test_count
        stats["totals"]["total"] += total_count
    
    return stats


# ============================================================
# MAIN PREPROCESSING
# ============================================================

def preprocess_merged_dataset():
    """Main preprocessing function"""
    
    print("="*70)
    print("PREPROCESSING MERGED DATASET")
    print("="*70)
    print(f"Source: {SOURCE_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Split Ratios: Train={TRAIN_RATIO*100}%, Val={VAL_RATIO*100}%, Test={TEST_RATIO*100}%")
    print(f"Random Seed: {RANDOM_SEED}")
    print("="*70)
    
    # Step 1: Set random seed
    set_seed(RANDOM_SEED)
    
    # Step 2: Scan source dataset
    print("\nStep 1: Scanning source dataset...")
    class_images = scan_source_dataset(SOURCE_PATH)
    
    if not class_images:
        print("[X] No images found in source dataset!")
        return
    
    total_images = sum(len(imgs) for imgs in class_images.values())
    print(f"[OK] Found {len(class_images)} classes with {total_images} total images")
    
    for class_name, images in sorted(class_images.items()):
        print(f"  - {class_name}: {len(images)} images")
    
    # Step 3: Perform stratified split
    print("\nStep 2: Performing stratified split...")
    train_split, val_split, test_split = stratified_split(
        class_images, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    print("[OK] Split completed")
    
    # Step 4: Check if output exists
    if OUTPUT_PATH.exists():
        response = input(f"\n[!] Output directory '{OUTPUT_PATH}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(OUTPUT_PATH)
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Step 5: Copy splits to output
    print("\nStep 3: Copying images to output directory...")
    
    train_count = copy_split_to_output(train_split, OUTPUT_PATH, "train")
    val_count = copy_split_to_output(val_split, OUTPUT_PATH, "val")
    test_count = copy_split_to_output(test_split, OUTPUT_PATH, "test")
    
    # Step 6: Generate statistics
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE - STATISTICS")
    print("="*70)
    
    stats = generate_statistics(train_split, val_split, test_split)
    
    print("\nPer-Class Breakdown:")
    print(f"{'Class':<20} {'Train':<12} {'Val':<12} {'Test':<12} {'Total':<12}")
    print("-" * 70)
    
    for class_name in sorted(stats["classes"].keys()):
        class_stats = stats["classes"][class_name]
        print(f"{class_name:<20} "
              f"{class_stats['train']:<6} ({class_stats['train_%']:>4.1f}%)  "
              f"{class_stats['val']:<6} ({class_stats['val_%']:>4.1f}%)  "
              f"{class_stats['test']:<6} ({class_stats['test_%']:>4.1f}%)  "
              f"{class_stats['total']:<12}")
    
    print("-" * 70)
    totals = stats["totals"]
    print(f"{'TOTAL':<20} "
          f"{totals['train']:<12} "
          f"{totals['val']:<12} "
          f"{totals['test']:<12} "
          f"{totals['total']:<12}")
    
    # Calculate actual percentages
    total = totals['total']
    actual_train_pct = totals['train'] / total * 100 if total > 0 else 0
    actual_val_pct = totals['val'] / total * 100 if total > 0 else 0
    actual_test_pct = totals['test'] / total * 100 if total > 0 else 0
    
    print(f"\nActual Split Percentages:")
    print(f"  Train: {actual_train_pct:.2f}%")
    print(f"  Val:   {actual_val_pct:.2f}%")
    print(f"  Test:  {actual_test_pct:.2f}%")
    
    print(f"\n{'='*70}")
    print(f"Output directory: {OUTPUT_PATH.absolute()}")
    print(f"{'='*70}")
    
    # Save statistics to JSON
    import json
    stats_file = OUTPUT_PATH / "split_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n[*] Statistics saved to: {stats_file}")
    
    # Create labels.txt
    labels_file = OUTPUT_PATH / "labels.txt"
    with open(labels_file, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(class_images.keys())))
    print(f"[*] Labels file saved to: {labels_file}")
    
    print("\n[OK] Preprocessing complete! Ready for training.")
    print(f"     Train: {OUTPUT_PATH.absolute() / 'train'}")
    print(f"     Val:   {OUTPUT_PATH.absolute() / 'val'}")
    print(f"     Test:  {OUTPUT_PATH.absolute() / 'test'}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        preprocess_merged_dataset()
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user.")
    except Exception as e:
        print(f"\n\n[X] Error: {e}")
        import traceback
        traceback.print_exc()

