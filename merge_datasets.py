#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Merger: Combine Kaggle, Trashnet, and RealWaste datasets
into a single unified dataset with consistent labels.

This script:
1. Maps classes from 3 datasets to 8 unified categories
2. Copies and organizes images into a new merged_dataset folder
3. Provides detailed statistics on the merged dataset
4. Handles class imbalances and duplicates

Usage:
    python merge_datasets.py
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import json
from typing import Dict, List, Tuple
from tqdm import tqdm

# ============================================================
# UNIFIED CLASS MAPPING
# ============================================================
# Map fine-grained classes to 8 broad, practical categories

UNIFIED_MAPPING = {
    "Cardboard": {
        "kaggle": ["cardboard_boxes", "cardboard_packaging"],
        "trashnet": ["cardboard"],
        "realwaste": ["Cardboard"]
    },
    
    "Glass": {
        "kaggle": ["glass_beverage_bottles", "glass_cosmetic_containers", "glass_food_jars"],
        "trashnet": ["glass"],
        "realwaste": ["Glass"]
    },
    
    "Metal": {
        "kaggle": ["aerosol_cans", "aluminum_food_cans", "aluminum_soda_cans", "steel_food_cans"],
        "trashnet": ["metal"],
        "realwaste": ["Metal"]
    },
    
    "Paper": {
        "kaggle": ["magazines", "newspaper", "office_paper"],
        "trashnet": ["paper"],
        "realwaste": ["Paper"]
    },
    
    "Plastic": {
        "kaggle": [
            "plastic_detergent_bottles", "plastic_food_containers",
            "plastic_soda_bottles", "plastic_water_bottles"
        ],
        "trashnet": ["plastic"],
        "realwaste": ["Plastic"]
    },
    
    "Organic": {
        "kaggle": ["coffee_grounds", "eggshells", "food_waste", "tea_bags"],
        "trashnet": [],  # Trashnet doesn't have organic waste
        "realwaste": ["Food Organics", "Vegetation"]
    },
    
    "Textile": {
        "kaggle": ["clothing", "shoes"],
        "trashnet": [],  # Trashnet doesn't have textiles
        "realwaste": ["Textile Trash"]
    },
    
    "Trash": {
        "kaggle": [
            "disposable_plastic_cutlery", "paper_cups", "plastic_cup_lids",
            "plastic_shopping_bags", "plastic_straws", "plastic_trash_bags",
            "styrofoam_cups", "styrofoam_food_containers"
        ],
        "trashnet": ["trash"],
        "realwaste": ["Miscellaneous Trash"]
    }
}


# ============================================================
# DATASET PATHS
# ============================================================

DATASET_PATHS = {
    "kaggle": Path("C:/Users/hoang/.cache/kagglehub/datasets/alistairking/recyclable-and-household-waste-classification/versions/1/images/images"),
    "trashnet": Path("trashnet/dataset-original"),
    "realwaste": Path("realwaste/realwaste-main/RealWaste")
}

OUTPUT_PATH = Path("merged_dataset")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_reverse_mapping() -> Dict[str, Tuple[str, str]]:
    """
    Create reverse mapping: original_class -> (unified_class, dataset)
    Returns: {"cardboard_boxes": ("Cardboard", "kaggle"), ...}
    """
    reverse_map = {}
    for unified_class, datasets in UNIFIED_MAPPING.items():
        for dataset, original_classes in datasets.items():
            for orig_class in original_classes:
                reverse_map[orig_class] = (unified_class, dataset)
    return reverse_map


def validate_dataset_paths() -> Dict[str, Path]:
    """Check if all dataset paths exist and prepare them"""
    validated_paths = {}
    
    for dataset, raw_path in DATASET_PATHS.items():
        path = raw_path if isinstance(raw_path, Path) else Path(raw_path)

        if not path.exists():
            print(f"[X] {dataset} path not found: {path}")
            continue

        print(f"[OK] {dataset} path found: {path}")
        validated_paths[dataset] = path.resolve()
    
    return validated_paths


def scan_dataset(dataset_name: str, dataset_path: Path) -> Dict[str, List[Path]]:
    """
    Scan a dataset and organize by class.
    Returns: {"class_name": [image_path1, image_path2, ...]}
    """
    class_images = defaultdict(list)
    
    if not dataset_path.exists():
        return dict(class_images)
    
    # Check if it's ImageFolder structure
    for class_dir in dataset_path.iterdir():
        if not class_dir.is_dir():
            continue

        if class_dir.name.startswith('.') or class_dir.name.startswith('__'):
            continue
        
        class_name = class_dir.name
        
        # Find all images (case-insensitive on Windows, so don't scan both cases)
        # Use **/* to search recursively (includes current dir + subdirs like Kaggle's default/real_world)
        all_images = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            all_images.extend(class_dir.glob(f"**/*{ext}"))  # Recursive search (includes current dir)
        
        # Deduplicate on Windows (case-insensitive filesystem may return duplicates)
        # Convert to absolute paths and use set to remove duplicates
        class_images[class_name] = list(set(img.resolve() for img in all_images))
    
    return dict(class_images)


def copy_images_with_progress(source_images: List[Path], 
                               dest_dir: Path,
                               dataset_prefix: str) -> int:
    """
    Copy images to destination with dataset prefix to avoid collisions.
    Returns: number of successfully copied images
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    successful = 0
    
    for img_path in tqdm(source_images, desc=f"  Copying", leave=False):
        try:
            # Include parent directory name to handle subdirs (e.g., Kaggle's default/real_world)
            # This prevents collisions when files have same name in different subdirs
            parent_dir_name = img_path.parent.name
            new_name = f"{dataset_prefix}_{parent_dir_name}_{img_path.name}"
            dest_path = dest_dir / new_name
            
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
                successful += 1
            else:
                successful += 1  # Count existing files too (for re-runs)
        except Exception as e:
            print(f"    ⚠️  Failed to copy {img_path.name}: {e}")
    
    return successful


# ============================================================
# MAIN MERGER
# ============================================================

def merge_datasets():
    """Main function to merge all datasets"""
    
    print("="*70)
    print("DATASET MERGER: Creating Unified Waste Classification Dataset")
    print("="*70)
    
    # Step 1: Validate paths
    print("\nStep 1: Validating dataset paths...")
    validated_paths = validate_dataset_paths()
    if not validated_paths:
        print("\n[X] No dataset paths are available. Cannot proceed.")
        return
    
    print(f"[OK] Found {len(validated_paths)} datasets")
    
    # Step 2: Scan all datasets
    print("\nStep 2: Scanning datasets...")
    all_scans = {}
    for dataset_name, dataset_path in validated_paths.items():
        print(f"\n  Scanning {dataset_name}...")
        scans = scan_dataset(dataset_name, dataset_path)
        all_scans[dataset_name] = scans
        print(f"    Found {len(scans)} classes, {sum(len(imgs) for imgs in scans.values())} images")
    
    # Step 3: Create reverse mapping
    print("\nStep 3: Creating class mappings...")
    reverse_map = create_reverse_mapping()
    
    # Step 4: Merge images
    print("\nStep 4: Merging images into unified dataset...")
    
    if OUTPUT_PATH.exists():
        response = input(f"\n[!] Output directory '{OUTPUT_PATH}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(OUTPUT_PATH)
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = defaultdict(lambda: {"total": 0, "by_dataset": defaultdict(int)})
    
    # Process each unified class
    for unified_class in UNIFIED_MAPPING.keys():
        print(f"\n  Processing: {unified_class}")
        unified_dir = OUTPUT_PATH / unified_class
        
        # Collect images from all datasets
        for dataset_name in validated_paths.keys():
            original_classes = UNIFIED_MAPPING[unified_class].get(dataset_name, [])
            
            for orig_class in original_classes:
                if orig_class in all_scans[dataset_name]:
                    images = all_scans[dataset_name][orig_class]
                    print(f"    - {dataset_name}/{orig_class}: {len(images)} images")
                    
                    # Copy images
                    copied = copy_images_with_progress(
                        images, 
                        unified_dir,
                        dataset_prefix=f"{dataset_name}_{orig_class}"
                    )
                    
                    stats[unified_class]["total"] += copied
                    stats[unified_class]["by_dataset"][dataset_name] += copied
    
    # Step 5: Generate report
    print("\n" + "="*70)
    print("MERGER COMPLETE - STATISTICS")
    print("="*70)
    
    total_images = 0
    for unified_class, class_stats in sorted(stats.items()):
        print(f"\n{unified_class}: {class_stats['total']} images")
        for dataset, count in class_stats['by_dataset'].items():
            print(f"  - {dataset}: {count}")
        total_images += class_stats['total']
    
    print(f"\n{'='*70}")
    print(f"TOTAL IMAGES: {total_images}")
    print(f"Output directory: {OUTPUT_PATH.absolute()}")
    print(f"{'='*70}")
    
    # Save mapping to JSON
    mapping_file = OUTPUT_PATH / "class_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump({
            "unified_classes": list(UNIFIED_MAPPING.keys()),
            "mapping": UNIFIED_MAPPING,
            "statistics": {k: dict(v) for k, v in stats.items()},
            "total_images": total_images
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[*] Class mapping saved to: {mapping_file}")
    
    # Create labels.txt
    labels_file = OUTPUT_PATH / "labels.txt"
    with open(labels_file, "w", encoding="utf-8") as f:
        f.write("\n".join(UNIFIED_MAPPING.keys()))
    print(f"[*] Labels file saved to: {labels_file}")
    
    print("\n[OK] Dataset merger complete! You can now train on the merged dataset.")
    print(f"     Use this path in your training script: {OUTPUT_PATH.absolute()}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        merge_datasets()
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user.")
    except Exception as e:
        print(f"\n\n[X] Error: {e}")
        import traceback
        traceback.print_exc()

