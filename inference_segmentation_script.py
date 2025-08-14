#!/usr/bin/env python3
"""
Segmentation inference script for multi-task nnU-Net.
Runs segmentation prediction on validation or test data.

Usage:
    python inference_segmentation.py --input_dir /path/to/images --output_dir /path/to/predictions
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path

def create_flat_directory(input_dir: Path, output_dir: Path):
    """
    Create a flat directory structure from subtype-organized validation data.
    nnUNetv2_predict expects all images in a single directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input is already flat (test data) or organized by subtype (validation data)
    image_files = list(input_dir.glob("*_0000.nii.gz"))
    
    if image_files:
        print(f"Input directory is already flat with {len(image_files)} images")
        return input_dir
    
    # If not flat, look for subtype folders
    print("Input directory appears to be organized by subtypes, creating flat structure...")
    
    total_copied = 0
    for subtype_folder in ["subtype0", "subtype1", "subtype2"]:
        subtype_path = input_dir / subtype_folder
        if subtype_path.exists():
            images = list(subtype_path.glob("*_0000.nii.gz"))
            for img in images:
                # Copy to flat directory
                dest = output_dir / img.name
                if not dest.exists():
                    import shutil
                    shutil.copy2(img, dest)
                    total_copied += 1
    
    print(f"Created flat directory with {total_copied} images at: {output_dir}")
    return output_dir

def run_segmentation_inference(input_dir: Path, output_dir: Path, dataset_id: int = 777,
                             fold: int = 0, trainer: str = "NNUNet_tuned", 
                             configuration: str = "3d_fullres", 
                             plans: str = "nnUNetResEncUNetMPlans"):
    """
    Run nnUNet segmentation inference using the trained multi-task model.
    """
    
    print("🔍 Running segmentation inference...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Trainer: {trainer}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build nnUNetv2_predict command
    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(dataset_id),
        "-f", str(fold),
        "-tr", trainer,
        "-c", configuration,
        "-p", plans
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the prediction
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Segmentation inference completed successfully!")
        
        # Count output files
        output_files = list(output_dir.glob("*.nii.gz"))
        print(f"📁 Generated {len(output_files)} segmentation files")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Segmentation inference failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ nnUNetv2_predict command not found. Make sure nnU-Net v2 is installed and in PATH.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run segmentation inference with multi-task nnU-Net")
    parser.add_argument("--input_dir", type=Path, required=True,
                       help="Directory containing input images (*_0000.nii.gz)")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for segmentation predictions")
    parser.add_argument("--dataset_id", type=int, default=777,
                       help="Dataset ID (default: 777)")
    parser.add_argument("--fold", type=int, default=0,
                       help="Fold number (default: 0)")
    parser.add_argument("--trainer", type=str, default="NNUNet_tuned",
                       help="Trainer name (default: NNUNet_tuned)")
    parser.add_argument("--configuration", type=str, default="3d_fullres",
                       help="Configuration (default: 3d_fullres)")
    parser.add_argument("--plans", type=str, default="nnUNetResEncUNetMPlans",
                       help="Plans name (default: nnUNetResEncUNetMPlans)")
    parser.add_argument("--create_flat", action="store_true",
                       help="Create flat directory structure from subtype folders")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        print(f"❌ Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Handle flat directory creation if needed
    if args.create_flat:
        flat_dir = args.output_dir.parent / f"{args.input_dir.name}_flat"
        input_dir = create_flat_directory(args.input_dir, flat_dir)
    else:
        input_dir = args.input_dir
    
    # Check if input directory has images
    image_files = list(input_dir.glob("*_0000.nii.gz"))
    if not image_files:
        print(f"❌ No image files (*_0000.nii.gz) found in: {input_dir}")
        sys.exit(1)
    
    print(f"📁 Found {len(image_files)} images for inference")
    
    # Run segmentation inference
    success = run_segmentation_inference(
        input_dir=input_dir,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        fold=args.fold,
        trainer=args.trainer,
        configuration=args.configuration,
        plans=args.plans
    )
    
    if success:
        print(f"\n🎉 Segmentation inference completed!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"\n📋 Next steps:")
        print(f"1. Run classification inference:")
        print(f"   python inference_classification.py --input_dir {input_dir} --output_file results/subtype_results.csv")
        print(f"2. Evaluate results:")
        print(f"   python evaluation.py --pred_seg_dir {args.output_dir} --pred_cls_csv results/subtype_results.csv")
    else:
        print("❌ Segmentation inference failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
