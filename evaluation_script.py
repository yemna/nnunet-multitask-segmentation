# Complete evaluation script for your results
import os
import csv
import numpy as np
import nibabel as nib
from sklearn.metrics import f1_score, classification_report

# Validation cases
validation_cases = [
    "quiz_0_168", "quiz_0_171", "quiz_0_174", "quiz_0_184", "quiz_0_187", 
    "quiz_0_189", "quiz_0_244", "quiz_0_253", "quiz_0_254", "quiz_1_090",
    "quiz_1_093", "quiz_1_094", "quiz_1_154", "quiz_1_158", "quiz_1_164",
    "quiz_1_166", "quiz_1_211", "quiz_1_213", "quiz_1_221", "quiz_1_227",
    "quiz_1_231", "quiz_1_242", "quiz_1_331", "quiz_1_335", "quiz_2_074",
    "quiz_2_080", "quiz_2_084", "quiz_2_085", "quiz_2_088", "quiz_2_089",
    "quiz_2_098", "quiz_2_191", "quiz_2_241", "quiz_2_364", "quiz_2_377",
    "quiz_2_379"
]

# Paths
gt_seg_dir = f"{os.environ['nnUNet_raw']}/Dataset777_M31Quiz/labelsTr"
pred_seg_dir = "D:/nnunet_with_classification/data/predictions_validation"
gt_cls_csv = f"{os.environ['nnUNet_preprocessed']}/Dataset777_M31Quiz/classification_labels.csv"
pred_cls_csv = "D:/nnunet_with_classification/predictions_validation/subtype_results.csv"

def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    total = np.sum(y_true) + np.sum(y_pred)
    return (2.0 * intersection) / total if total > 0 else 1.0

# Check paths
print("🔍 Checking paths...")
print(f"GT segmentation dir: {os.path.exists(gt_seg_dir)}")
print(f"Predicted segmentation dir: {os.path.exists(pred_seg_dir)}")
print(f"GT classification CSV: {os.path.exists(gt_cls_csv)}")
print(f"Predicted classification CSV: {os.path.exists(pred_cls_csv)}")

# Calculate segmentation metrics
pancreas_dices = []
lesion_dices = []
missing_files = []

print("\n📊 Computing segmentation metrics...")
for case in validation_cases:
    gt_file = f"{gt_seg_dir}/{case}.nii.gz"
    pred_file = f"{pred_seg_dir}/{case}.nii.gz"
    
    if os.path.exists(gt_file) and os.path.exists(pred_file):
        try:
            gt = nib.load(gt_file).get_fdata()
            pred = nib.load(pred_file).get_fdata()
            
            # Whole pancreas (label > 0)
            gt_pancreas = (gt > 0).astype(int)
            pred_pancreas = (pred > 0).astype(int)
            pancreas_dice = dice_score(gt_pancreas, pred_pancreas)
            pancreas_dices.append(pancreas_dice)
            
            # Lesion only (label == 2)
            gt_lesion = (gt == 2).astype(int)
            pred_lesion = (pred == 2).astype(int)
            lesion_dice = dice_score(gt_lesion, pred_lesion)
            lesion_dices.append(lesion_dice)
            
            print(f"{case}: Pancreas={pancreas_dice:.3f}, Lesion={lesion_dice:.3f}")
            
        except Exception as e:
            print(f"Error processing {case}: {e}")
            missing_files.append(case)
    else:
        missing_files.append(case)
        print(f"Missing files for {case}")

if pancreas_dices:
    whole_pancreas_dsc = np.mean(pancreas_dices)
    lesion_dsc = np.mean(lesion_dices)
    
    print(f"\n📈 Segmentation Statistics:")
    print(f"Pancreas DSC - Mean: {whole_pancreas_dsc:.4f}, Std: {np.std(pancreas_dices):.4f}")
    print(f"Lesion DSC - Mean: {lesion_dsc:.4f}, Std: {np.std(lesion_dices):.4f}")
else:
    whole_pancreas_dsc = 0.0
    lesion_dsc = 0.0

# Classification metrics
print("\n🎯 Computing classification metrics...")
gt_labels = {}
with open(gt_cls_csv, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) == 2:
            gt_labels[row[0]] = int(row[1])

pred_labels = {}
with open(pred_cls_csv, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if len(row) == 2:
            case_name = row[0].replace('.nii.gz', '')
            pred_labels[case_name] = int(row[1])

# Get validation classification results
val_gt = []
val_pred = []
for case in validation_cases:
    if case in gt_labels and case in pred_labels:
        val_gt.append(gt_labels[case])
        val_pred.append(pred_labels[case])

if val_gt and val_pred:
    macro_f1 = f1_score(val_gt, val_pred, average='macro')
    
    print(f"\n🔍 Classification Analysis:")
    print(f"Ground truth distribution: {np.bincount(val_gt)}")
    print(f"Prediction distribution: {np.bincount(val_pred)}")
    print(f"\nDetailed Classification Report:")
    print(classification_report(val_gt, val_pred, target_names=['Subtype 0', 'Subtype 1', 'Subtype 2']))
else:
    macro_f1 = 0.0

# Final Results
print("\n" + "="*70)
print("🎓 PhD LEVEL REQUIREMENTS EVALUATION")
print("="*70)
print(f"Whole Pancreas DSC: {whole_pancreas_dsc:.4f} ≥ 0.91: {'✅ PASS' if whole_pancreas_dsc >= 0.91 else '❌ FAIL'}")
print(f"Lesion DSC: {lesion_dsc:.4f} ≥ 0.31: {'✅ PASS' if lesion_dsc >= 0.31 else '❌ FAIL'}")
print(f"Macro F1: {macro_f1:.4f} ≥ 0.70: {'✅ PASS' if macro_f1 >= 0.70 else '❌ FAIL'}")

overall_pass = (whole_pancreas_dsc >= 0.91) and (lesion_dsc >= 0.31) and (macro_f1 >= 0.70)
print(f"\n🎯 OVERALL RESULT: {'✅ PASS' if overall_pass else '❌ FAIL'}")

# Summary for your report
print(f"\n📋 SUMMARY FOR REPORT:")
print(f"Processed {len(pancreas_dices)} segmentation cases and {len(val_gt)} classification cases")
print(f"Whole Pancreas DSC: {whole_pancreas_dsc:.4f}")
print(f"Lesion DSC: {lesion_dsc:.4f}")
print(f"Classification Macro F1: {macro_f1:.4f}")

if missing_files:
    print(f"⚠️ Missing files for {len(missing_files)} cases: {missing_files}")
