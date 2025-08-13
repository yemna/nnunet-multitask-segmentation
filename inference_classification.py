#!/usr/bin/env python3
"""
Classification inference script for multi-task nnU-Net.
Extracts features from trained segmentation model and performs classification inference.

Based on the working implementation from the development notebook.
"""

import os
import json
import torch
import csv
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch.nn as nn

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

class GlobalPoolFlatten(nn.Module):
    """Adaptive global average pooling followed by flattening."""
    
    def __init__(self, ndim: int):
        super().__init__()
        if ndim == 2:
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif ndim == 3:
            self.pool = nn.AdaptiveAvgPool3d(1)
        else:
            raise ValueError("ndim must be 2 or 3")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(x), 1)

def setup_predictor(model_dir: Path, checkpoint_name: str, device: torch.device):
    """Initialize nnUNet predictor from trained model."""
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=device,
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(model_dir),
        use_folds=[0],
        checkpoint_name=checkpoint_name,
    )
    
    return predictor

def load_classification_head(checkpoint_path: Path, encoder_channels: int, num_classes: int, 
                           spatial_dims: int, device: torch.device):
    """Load the classification head from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cls_state_dict = checkpoint.get("cls_state_dict", None)
    
    if cls_state_dict is None:
        raise ValueError("No 'cls_state_dict' found in checkpoint. "
                        "Make sure you trained with the multi-task trainer.")
    
    # Create classification components
    global_pool = GlobalPoolFlatten(spatial_dims).to(device)
    classifier = nn.Linear(encoder_channels, num_classes).to(device)
    
    # Load weights
    if "classifier" in cls_state_dict:
        classifier.load_state_dict(cls_state_dict["classifier"], strict=False)
    
    if "global_pool" in cls_state_dict:
        global_pool.load_state_dict(cls_state_dict["global_pool"], strict=False)
    
    return global_pool, classifier

def predict_classification(input_dir: Path, output_file: Path, model_dir: Path, 
                         checkpoint_name: str = "checkpoint_final.pth"):
    """Run classification inference on test images."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Validate paths
    checkpoint_path = model_dir / "fold_all" / checkpoint_name
    if not checkpoint_path.is_file():
        checkpoint_path = model_dir / "fold_0" / checkpoint_name
    
    assert checkpoint_path.is_file(), f"Checkpoint not found: {checkpoint_path}"
    assert input_dir.is_dir(), f"Input directory not found: {input_dir}"
    
    # Setup predictor
    print("Initializing nnUNet predictor...")
    predictor = setup_predictor(model_dir, checkpoint_name, device)
    network = predictor.network
    
    # Load classification head
    print("Loading classification head...")
    encoder_channels = network.encoder.output_channels[-1]  # Deepest layer channels
    spatial_dims = 3  # 3D images
    num_classes = 3   # Subtypes 0, 1, 2
    
    global_pool, classifier = load_classification_head(
        checkpoint_path, encoder_channels, num_classes, spatial_dims, device
    )
    
    # Setup feature capture hook
    enc_features = {"x": None}
    def feature_hook(module, input, output):
        enc_features["x"] = output
    
    hook = network.encoder.stages[-1].register_forward_hook(feature_hook)
    
    # Set models to evaluation mode
    network.eval()
    classifier.eval()
    torch.set_grad_enabled(False)
    
    # Find test images
    case_files = sorted(list(input_dir.glob("*_0000.nii.gz")))
    print(f"Found {len(case_files)} test files")
    
    if len(case_files) == 0:
        print("Warning: No test files found with pattern '*_0000.nii.gz'")
        return
    
    # Process each test case
    results = [("Names", "Subtype")]  # CSV header
    temp_output_dir = output_file.parent / "temp_seg_output"
    temp_output_dir.mkdir(exist_ok=True)
    
    for img_path in case_files:
        case_id = img_path.name.replace("_0000.nii.gz", "") + ".nii.gz"
        print(f"Processing: {img_path.name}")
        
        # Reset captured features
        enc_features["x"] = None
        
        # Run inference (triggers feature capture)
        temp_output = temp_output_dir / f"temp_{case_id}"
        
        try:
            predictor.predict_from_files(
                [[str(img_path)]], 
                [str(temp_output)],
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1
            )
            
            # Clean up temporary segmentation
            if temp_output.exists():
                temp_output.unlink()
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            prediction = 0  # Default fallback
            results.append((case_id, prediction))
            continue
        
        # Perform classification
        if enc_features["x"] is not None:
            features = enc_features["x"].float()  # Ensure float32
            pooled_features = global_pool(features)
            logits = classifier(pooled_features)
            prediction = int(torch.argmax(logits, dim=1).item())
        else:
            print(f"Warning: No features captured for {img_path.name}")
            prediction = 0  # Default fallback
        
        results.append((case_id, prediction))
        print(f"  → Classification: {prediction}")
    
    # Cleanup
    hook.remove()
    temp_output_dir.rmdir()
    
    # Save results to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)
    
    print(f"\n✅ Classification results saved to: {output_file}")
    
    # Show classification distribution
    df = pd.read_csv(output_file)
    print(f"\nClassification distribution:")
    print(df['Subtype'].value_counts().sort_index())
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Run classification inference with multi-task nnU-Net")
    parser.add_argument("--input_dir", type=Path, required=True,
                       help="Directory containing test images (*_0000.nii.gz)")
    parser.add_argument("--output_file", type=Path, required=True,
                       help="Output CSV file for classification results")
    parser.add_argument("--model_dir", type=Path, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_final.pth",
                       help="Checkpoint filename (default: checkpoint_final.pth)")
    
    args = parser.parse_args()
    
    predict_classification(
        input_dir=args.input_dir,
        output_file=args.output_file,
        model_dir=args.model_dir,
        checkpoint_name=args.checkpoint
    )

if __name__ == "__main__":
    main()
