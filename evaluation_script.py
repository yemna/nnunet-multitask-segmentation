#!/usr/bin/env python3
"""
Evaluation script for multi-task nnU-Net validation performance.
Computes Dice scores for segmentation and macro F1 for classification.

Based on the validation evaluation implementation.
"""

import os
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import nibabel as nib
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Validation cases from the dataset split
VALIDATION_CASES = [
    "quiz_0_168", "quiz_0_171", "quiz_0_174", "quiz_0_184", "quiz_0_187", 
    "quiz_0_189", "quiz_0_244", "quiz_0_253", "quiz_0_254", "quiz_1_090",
    "quiz_1_093", "quiz_1_094", "quiz_1_154", "quiz_1_158", "quiz_1_164",
    "quiz_1_166", "quiz_1_211", "quiz_1_213", "quiz_1_221", "quiz_1_227",
    "quiz_1_231", "quiz_1_242", "quiz_1_331", "quiz_1_335", "quiz_2_074",
    "quiz_2_080", "quiz_2_084", "quiz_2_085", "quiz_2_088", "quiz_2_089",
    "quiz_2_098", "quiz_2_191", "quiz_2_241", "quiz_2_364", "quiz_2_377",
    "quiz_2_379"
]

def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Dice Similarity Coefficient."""
    intersection = np.sum(y_true * y_pred)
    total = np.sum(y_true) + np.sum(y_pred)
    
    if total == 0:
        return 1.0  # Both empty - perfect score
    
    return (2.0 * intersection) / total

def load_segmentation_masks(gt_dir: Path, pred_dir: Path, cases: List[str]) -> Tuple[List[float], List[float]]:
    """Load and compute segmentation metrics for validation cases."""
    pancreas_dices = []
    lesion_dices = []
    
    missing_cases = []
    
    for case in cases:
        gt_file = gt_dir / f"{case}.nii.gz"
        pred_file = pred_dir / f"{case}.nii.gz"
        
        if not gt_file.exists():
            print(f"Warning: Missing ground truth for {case}")
            missing_cases.append(case)
            continue
            
        if not pred_file.exists():
            print(f"Warning: Missing prediction for {case}")
            missing_cases.append(case)
            continue
        
        try:
            # Load masks
            gt = nib.load(gt_file).get_fdata()
            pred = nib.load(pred_file).get_fdata()
            
            # Ensure same shape
            if gt.shape != pred.shape:
                print(f"Warning: Shape mismatch for {case}: GT {gt.shape} vs Pred {pred.shape}")
                continue
            
            # Whole pancreas (label > 0)
            gt_pancreas = (gt > 0).astype(np.uint8)
            pred_pancreas = (pred > 0).astype(np.uint8)
            pancreas_dice = dice_score(gt_pancreas, pred_pancreas)
            pancreas_dices.append(pancreas_dice)
            
            # Lesion only (label == 2)
            gt_lesion = (gt == 2).astype(np.uint8)
            pred_lesion = (pred == 2).astype(np.uint8)
            lesion_dice = dice_score(gt_lesion, pred_lesion)
            lesion_dices.append(lesion_dice)
            
        except Exception as e:
            print(f"Error processing {case}: {e}")
            missing_cases.append(case)
    
    if missing_cases:
        print(f"Skipped {len(missing_cases)} cases due to errors: {missing_cases[:5]}...")
    
    return pancreas_dices, lesion_dices

def load_classification_labels(gt_csv: Path, pred_csv: Path, cases: List[str]) -> Tuple[List[int], List[int]]:
    """Load classification labels for validation cases."""
    # Load ground truth labels
    gt_labels = {}
    if gt_csv.exists():
        with open(gt_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    try:
                        gt_labels[row[0]] = int(row[1])
                    except ValueError:
                        continue
    
    # Load predicted labels
    pred_labels = {}
    if pred_csv.exists():
        with open(pred_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header if present
            for row in reader:
                if len(row) == 2:
                    try:
                        case_name = row[0].replace('.nii.gz', '')  # Remove extension
                        pred_labels[case_name] = int(row[1])
                    except ValueError:
                        continue
    
    # Extract validation labels
    val_gt = []
    val_pred = []
    missing_labels = []
    
    for case in cases:
        if case in gt_labels and case in pred_labels:
            val_gt.append(gt_labels[case])
            val_pred.append(pred_labels[case])
        else:
            missing_labels.append(case)
    
    if missing_labels:
        print(f"Missing classification labels for {len(missing_labels)} cases")
    
    return val_gt, val_pred

def evaluate_model(gt_seg_dir: Path, pred_seg_dir: Path, gt_cls_csv: Path, 
                  pred_cls_csv: Path, level: str = "phd") -> Dict:
    """Evaluate model performance on validation set."""
    
    print("🔍 Starting model evaluation...")
    print(f"📂 GT Segmentation: {gt_seg_dir}")
    print(f"📂 Pred Segmentation: {pred_seg_dir}")
    print(f"📄 GT Classification: {gt_cls_csv}")
    print(f"📄 Pred Classification: {pred_cls_csv}")
    
    # Segmentation evaluation
    print("\n📊 Computing segmentation metrics...")
    pancreas_dices, lesion_dices = load_segmentation_masks(
        gt_seg_dir, pred_seg_dir, VALIDATION_CASES
    )
    
    if not pancreas_dices:
        print("❌ No segmentation results found!")
        return {}
    
    whole_pancreas_dsc = np.mean(pancreas_dices)
    lesion_dsc = np.mean(lesion_dices)
    
    print(f"Processed {len(pancreas_dices)} segmentation cases")
    
    # Classification evaluation
    print("\n🎯 Computing classification metrics...")
    val_gt, val_pred = load_classification_labels(
        gt_cls_csv, pred_cls_csv, VALIDATION_CASES
    )
    
    if val_gt and val_pred:
        macro_f1 = f1_score(val_gt, val_pred, average='macro')
        print(f"Processed {len(val_gt)} classification cases")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(val_gt, val_pred, 
                                  target_names=['Subtype 0', 'Subtype 1', 'Subtype 2']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(val_gt, val_pred)
        print(cm)
    else:
        macro_f1 = 0.0
        print("❌ No classification results found!")
    
    # Results summary
    results = {
        'whole_pancreas_dsc': whole_pancreas_dsc,
        'lesion_dsc': lesion_dsc,
        'macro_f1': macro_f1,
        'n_seg_cases': len(pancreas_dices),
        'n_cls_cases': len(val_gt)
    }
    
    # Performance thresholds
    if level.lower() == "phd":
        thresholds = {'pancreas': 0.91, 'lesion': 0.31, 'f1': 0.70}
        level_name = "PhD"
    else:
        thresholds = {'pancreas': 0.85, 'lesion': 0.27, 'f1': 0.60}
        level_name = "Undergraduate"
    
    print(f"\n🎓 {level_name.upper()} LEVEL REQUIREMENTS:")
    print("=" * 50)
    
    # Check requirements
    pancreas_pass = whole_pancreas_dsc >= thresholds['pancreas']
    lesion_pass = lesion_dsc >= thresholds['lesion']
    f1_pass = macro_f1 >= thresholds['f1']
    
    print(f"Whole Pancreas DSC: {whole_pancreas_dsc:.4f} ≥ {thresholds['pancreas']}: {'✅ PASS' if pancreas_pass else '❌ FAIL'}")
    print(f"Lesion DSC: {lesion_dsc:.4f} ≥ {thresholds['lesion']}: {'✅ PASS' if lesion_pass else '❌ FAIL'}")
    print(f"Macro F1: {macro_f1:.4f} ≥ {thresholds['f1']}: {'✅ PASS' if f1_pass else '❌ FAIL'}")
    
    overall_pass = pancreas_pass and lesion_pass and f1_pass
    print(f"\n🎯 OVERALL: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    
    if overall_pass:
        print("🎉 Congratulations! All requirements met!")
    else:
        print("📈 Areas for improvement:")
        if not pancreas_pass:
            print(f"  • Whole pancreas segmentation needs {thresholds['pancreas'] - whole_pancreas_dsc:.4f} improvement")
        if not lesion_pass:
            print(f"  • Lesion segmentation needs {thresholds['lesion'] - lesion_dsc:.4f} improvement")
        if not f1_pass:
            print(f"  • Classification needs {thresholds['f1'] - macro_f1:.4f} improvement")
    
    # Store results
    results.update({
        'thresholds': thresholds,
        'passes': {
            'pancreas': pancreas_pass,
            'lesion': lesion_pass,
            'f1': f1_pass,
            'overall': overall_pass
        }
    })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-task nnU-Net performance")
    parser.add_argument("--gt_seg_dir", type=Path, required=True,
                       help="Ground truth segmentation directory")
    parser.add_argument("--pred_seg_dir", type=Path, required=True,
                       help="Predicted segmentation directory")
    parser.add_argument("--gt_cls_csv", type=Path, required=True,
                       help="Ground truth classification CSV")
    parser.add_argument("--pred_cls_csv", type=Path, required=True,
                       help="Predicted classification CSV")
    parser.add_argument("--level", choices=["undergraduate", "phd"], default="phd",
                       help="Evaluation level (undergraduate or phd)")
    parser.add_argument("--output", type=Path,
                       help="Optional output file for results")
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model(
        gt_seg_dir=args.gt_seg_dir,
        pred_seg_dir=args.pred_seg_dir,
        gt_cls_csv=args.gt_cls_csv,
        pred_cls_csv=args.pred_cls_csv,
        level=args.level
    )
    
    # Save results if requested
    if args.output and results:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")

if __name__ == "__main__":
    main()