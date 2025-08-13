#!/usr/bin/env python3
"""
Training script for multi-task nnU-Net pancreas segmentation and classification.
Wrapper around nnUNetv2_train with proper environment setup.
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path

def setup_environment(base_dir: str):
    """Setup nnU-Net environment variables."""
    raw_dir = f"{base_dir}/nnUNet_raw"
    prep_dir = f"{base_dir}/nnUNet_preprocessed" 
    results_dir = f"{base_dir}/nnUNet_results"
    
    os.environ["nnUNet_raw"] = raw_dir
    os.environ["nnUNet_preprocessed"] = prep_dir
    os.environ["nnUNet_results"] = results_dir
    
    print(f"Environment setup:")
    print(f"  nnUNet_raw: {raw_dir}")
    print(f"  nnUNet_preprocessed: {prep_dir}")
    print(f"  nnUNet_results: {results_dir}")

def run_preprocessing(dataset_id: int):
    """Run nnU-Net preprocessing."""
    print(f"\n🔧 Running preprocessing for dataset {dataset_id}...")
    
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "-c", "3d_fullres",
        "--verify_dataset_integrity",
        "-pl", "nnUNetPlannerResEncM"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Preprocessing completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Preprocessing failed with error: {e}")
        return False

def run_training(dataset_id: int, fold: int = 0, trainer: str = "NNUNet", 
                plans: str = "nnUNetResEncUNetMPlans", continue_training: bool = False):
    """Run nnU-Net training with custom trainer."""
    print(f"\n🚀 Starting training for dataset {dataset_id}...")
    print(f"  Fold: {fold}")
    print(f"  Trainer: {trainer}")
    print(f"  Plans: {plans}")
    
    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        "3d_fullres", 
        str(fold),
        "-tr", trainer,
        "-p", plans
    ]
    
    if continue_training:
        cmd.append("-c")
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return False

def find_trainer_file():
    """Check if custom trainer is available."""
    try:
        # Try to import the custom trainer
        from NNUNet import NNUNet
        print("✅ Custom trainer (NNUNet) found and importable")
        return True
    except ImportError:
        print("❌ Custom trainer (NNUNet) not found")
        print("Make sure NNUNet.py is in your Python path or current directory")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train multi-task nnU-Net model")
    parser.add_argument("--base_dir", type=str, required=True,
                       help="Base directory containing nnU-Net folders")
    parser.add_argument("--dataset_id", type=int, default=777,
                       help="Dataset ID (default: 777)")
    parser.add_argument("--fold", type=int, default=0,
                       help="Fold number (default: 0)")
    parser.add_argument("--trainer", type=str, default="NNUNet",
                       help="Trainer class name (default: NNUNet)")
    parser.add_argument("--plans", type=str, default="nnUNetResEncUNetMPlans",
                       help="Plans name (default: nnUNetResEncUNetMPlans)")
    parser.add_argument("--continue_training", action="store_true",
                       help="Continue training from checkpoint")
    parser.add_argument("--skip_preprocessing", action="store_true",
                       help="Skip preprocessing step")
    parser.add_argument("--preprocessing_only", action="store_true",
                       help="Only run preprocessing, skip training")
    
    args = parser.parse_args()
    
    # Validate base directory
    base_path = Path(args.base_dir)
    if not base_path.exists():
        print(f"❌ Base directory does not exist: {args.base_dir}")
        sys.exit(1)
    
    # Setup environment
    setup_environment(args.base_dir)
    
    # Check for custom trainer
    if not find_trainer_file():
        print("\n⚠️  Warning: Custom trainer not found. Training may fail.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Run preprocessing
    if not args.skip_preprocessing:
        success = run_preprocessing(args.dataset_id)
        if not success:
            print("❌ Preprocessing failed. Aborting.")
            sys.exit(1)
    
    # Exit if only preprocessing requested
    if args.preprocessing_only:
        print("✅ Preprocessing completed. Exiting as requested.")
        sys.exit(0)
    
    # Run training
    success = run_training(
        dataset_id=args.dataset_id,
        fold=args.fold,
        trainer=args.trainer,
        plans=args.plans,
        continue_training=args.continue_training
    )
    
    if success:
        print("\n🎉 Training pipeline completed successfully!")
        
        # Show next steps
        model_dir = f"{args.base_dir}/nnUNet_results/Dataset{args.dataset_id:03d}_M31Quiz/{args.trainer}__{args.plans}__3d_fullres"
        print(f"\n📁 Model saved to: {model_dir}")
        print(f"\n🔍 Next steps:")
        print(f"1. Run validation evaluation:")
        print(f"   python evaluation.py --gt_seg_dir <gt_dir> --pred_seg_dir <pred_dir> --gt_cls_csv <gt_csv> --pred_cls_csv <pred_csv>")
        print(f"2. Run inference on test data:")
        print(f"   python inference_classification.py --input_dir <test_dir> --output_file subtype_results.csv --model_dir {model_dir}")
        
    else:
        print("❌ Training pipeline failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
