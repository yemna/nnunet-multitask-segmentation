#!/usr/bin/env python3
"""
Data preparation script for nnU-Net multi-task pancreas segmentation and classification.
Converts the subtype-organized dataset to nnU-Net v2 format.

Based on the Jupyter notebook implementation.
"""

import os
import glob
import pathlib
import shutil
import re
import json
import csv
import argparse
import nibabel as nib
import numpy as np
from typing import Dict, List, Tuple

def setup_nnunet_dirs(base_dir: str) -> Tuple[str, str, str]:
    """Setup nnU-Net directory structure."""
    raw_dir = f"{base_dir}/nnUNet_raw"
    prep_dir = f"{base_dir}/nnUNet_preprocessed"
    results_dir = f"{base_dir}/nnUNet_results"
    
    for d in [raw_dir, prep_dir, results_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Set environment variables
    os.environ["nnUNet_raw"] = raw_dir
    os.environ["nnUNet_preprocessed"] = prep_dir
    os.environ["nnUNet_results"] = results_dir
    
    print(f"nnU-Net directories created at: {base_dir}")
    return raw_dir, prep_dir, results_dir

def find_split_folder(src_dir: str, split_name: str) -> str:
    """Find split folder with case-insensitive matching."""
    for candidate in os.listdir(src_dir):
        if candidate.lower() == split_name.lower():
            path = os.path.join(src_dir, candidate)
            if os.path.isdir(path):
                return path
    raise FileNotFoundError(f"Could not find {split_name} folder under {src_dir}")

def extract_case_name(filepath: str) -> str:
    """Extract case name from file path."""
    stem = pathlib.Path(filepath).name.replace(".nii.gz", "")
    return stem.replace("_0000", "")

def ingest_split(split_dir: str, img_dir: str, lbl_dir: str, cls_map: Dict[str, int]) -> int:
    """Process a data split and copy files to nnU-Net format."""
    sub_regex = re.compile(r"subtype\s*([012])", re.IGNORECASE)
    imgs = glob.glob(os.path.join(split_dir, "**", "*_0000.nii.gz"), recursive=True)
    
    processed = 0
    for img_path in sorted(imgs):
        case = extract_case_name(img_path)
        mask_path = img_path.replace("_0000.nii.gz", ".nii.gz")
        
        if not os.path.exists(mask_path):
            print(f"Warning: Missing mask for {img_path}")
            continue
            
        # Copy files
        shutil.copy(img_path, f"{img_dir}/{case}_0000.nii.gz")
        shutil.copy(mask_path, f"{lbl_dir}/{case}.nii.gz")
        processed += 1
        
        # Extract subtype from folder structure
        sub_idx = None
        for part in pathlib.Path(img_path).parts:
            match = sub_regex.search(part)
            if match:
                sub_idx = int(match.group(1))
                break
        
        if sub_idx is not None:
            cls_map[case] = sub_idx
        else:
            print(f"Warning: Could not extract subtype for {case}")
    
    return processed

def fix_label_values(label_dir: str):
    """Fix non-integer label values in masks."""
    print("Checking and fixing label values...")
    
    bad_files = []
    for mask_path in sorted(glob.glob(os.path.join(label_dir, "*.nii.gz"))):
        img = nib.load(mask_path)
        arr = img.get_fdata()
        unique_vals = np.unique(arr)
        
        if not np.all(np.isin(unique_vals, [0, 1, 2])):
            bad_files.append((mask_path, unique_vals))
    
    print(f"Found {len(bad_files)} masks needing fixes")
    
    # Fix problematic files
    for mask_path, unique_vals in bad_files:
        img = nib.load(mask_path)
        arr = img.get_fdata()
        
        # Round and clip to valid label set
        arr = np.rint(arr)
        arr = np.clip(arr, 0, 2)
        arr = arr.astype(np.uint8)
        
        # Update header
        header = img.header.copy()
        header.set_data_dtype(np.uint8)
        header["scl_slope"] = 1
        header["scl_inter"] = 0
        
        # Save fixed image
        fixed_img = nib.Nifti1Image(arr, img.affine, header)
        nib.save(fixed_img, mask_path)
    
    print(f"Fixed {len(bad_files)} label files")

def create_dataset_json(dataset_root: str, img_dir: str, lbl_dir: str, test_dir: str, dataset_id: int):
    """Create dataset.json for nnU-Net v2."""
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "pancreas": 1, "lesion": 2},
        "numTraining": len(glob.glob(os.path.join(lbl_dir, "*.nii.gz"))),
        "file_ending": ".nii.gz"
    }
    
    with open(f"{dataset_root}/dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"Created dataset.json with {dataset_json['numTraining']} training cases")

def create_splits_file(prep_dir: str, dataset_name: str, val_dir: str, img_dir: str):
    """Create splits_final.json with original validation split."""
    # Get validation cases
    val_imgs = glob.glob(os.path.join(val_dir, "**", "*_0000.nii.gz"), recursive=True)
    val_cases = {extract_case_name(p) for p in val_imgs}
    
    # Get all training cases
    all_imgs = glob.glob(os.path.join(img_dir, "*_0000.nii.gz"))
    all_cases = sorted([extract_case_name(p) for p in all_imgs])
    
    # Split into train and validation
    train_cases = [c for c in all_cases if c not in val_cases]
    
    splits_data = [{
        "train": train_cases,
        "val": sorted(list(val_cases))
    }]
    
    splits_dir = f"{prep_dir}/{dataset_name}"
    os.makedirs(splits_dir, exist_ok=True)
    
    with open(f"{splits_dir}/splits_final.json", "w") as f:
        json.dump(splits_data, f, indent=2)
    
    print(f"Created splits: {len(train_cases)} train, {len(val_cases)} validation")

def create_classification_labels(prep_dir: str, dataset_name: str, cls_map: Dict[str, int], all_cases: List[str]):
    """Create classification_labels.csv."""
    csv_path = f"{prep_dir}/{dataset_name}/classification_labels.csv"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for case in all_cases:
            if case in cls_map:
                writer.writerow([case, cls_map[case]])
    
    print(f"Created classification labels for {len(cls_map)} cases")

def main():
    parser = argparse.ArgumentParser(description="Prepare pancreas dataset for nnU-Net v2")
    parser.add_argument("--input_dir", required=True, help="Path to input data directory")
    parser.add_argument("--output_dir", required=True, help="Path to output nnU-Net directory")
    parser.add_argument("--dataset_id", type=int, default=777, help="Dataset ID")
    
    args = parser.parse_args()
    
    # Setup directories
    raw_dir, prep_dir, results_dir = setup_nnunet_dirs(args.output_dir)
    
    dataset_name = f"Dataset{args.dataset_id:03d}_M31Quiz"
    dataset_root = f"{raw_dir}/{dataset_name}"
    
    # Create dataset structure
    img_tr = f"{dataset_root}/imagesTr"
    lbl_tr = f"{dataset_root}/labelsTr"
    img_ts = f"{dataset_root}/imagesTs"
    
    for d in [img_tr, lbl_tr, img_ts]:
        os.makedirs(d, exist_ok=True)
    
    # Find split directories
    train_dir = find_split_folder(args.input_dir, "train")
    val_dir = find_split_folder(args.input_dir, "validation")
    test_dir = find_split_folder(args.input_dir, "test")
    
    print(f"Found directories:")
    print(f"  Train: {train_dir}")
    print(f"  Validation: {val_dir}")
    print(f"  Test: {test_dir}")
    
    # Process splits
    cls_map = {}
    n_train = ingest_split(train_dir, img_tr, lbl_tr, cls_map)
    n_val = ingest_split(val_dir, img_tr, lbl_tr, cls_map)
    
    print(f"Processed {n_train} training cases, {n_val} validation cases")
    
    # Copy test images
    test_imgs = glob.glob(os.path.join(test_dir, "**", "*_0000.nii.gz"), recursive=True)
    for img_path in sorted(test_imgs):
        case = extract_case_name(img_path)
        shutil.copy(img_path, f"{img_ts}/{case}_0000.nii.gz")
    
    print(f"Copied {len(test_imgs)} test images")
    
    # Fix label values
    fix_label_values(lbl_tr)
    
    # Create nnU-Net files
    create_dataset_json(dataset_root, img_tr, lbl_tr, img_ts, args.dataset_id)
    create_splits_file(prep_dir, dataset_name, val_dir, img_tr)
    
    # Get all cases for classification labels
    all_cases = sorted([extract_case_name(p) for p in glob.glob(f"{img_tr}/*_0000.nii.gz")])
    create_classification_labels(prep_dir, dataset_name, cls_map, all_cases)
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"📁 Dataset root: {dataset_root}")
    print(f"📊 Classification distribution:")
    for i in range(3):
        count = sum(1 for v in cls_map.values() if v == i)
        print(f"  Subtype {i}: {count} cases")

if __name__ == "__main__":
    main()
