#!/usr/bin/env python3
"""
Classification inference script for multi-task nnU-Net.
Fixed version that handles FP16/FP32 precision mismatch and MultiScaleClassificationHead.

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
from typing import List

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

class MultiScaleClassificationHead(nn.Module):
    """
    Multi-scale feature fusion classification head with attention mechanism
    (Must match exactly what's in your NNUNet.py)
    """
    def __init__(self, encoder_channels: List[int], num_classes: int, dim: int = 3, 
                 target_channels: int = 256, spatial_reduction: int = 4):
        super().__init__()
        self.num_scales = 3  # Use last 3 encoder stages
        self.dim = dim
        
        # Multi-scale feature adapters
        self.feature_adapters = nn.ModuleList()
        
        for channels in encoder_channels[-self.num_scales:]:
            if dim == 3:
                adapter = nn.Sequential(
                    nn.AdaptiveAvgPool3d((spatial_reduction, spatial_reduction, spatial_reduction)),
                    nn.Conv3d(channels, target_channels, kernel_size=1, bias=False),
                    nn.BatchNorm3d(target_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout3d(0.1)
                )
            else:
                adapter = nn.Sequential(
                    nn.AdaptiveAvgPool2d((spatial_reduction, spatial_reduction)),
                    nn.Conv2d(channels, target_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(target_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1)
                )
            self.feature_adapters.append(adapter)
        
        # Global pooling for each scale
        if dim == 3:
            self.global_pool = nn.AdaptiveAvgPool3d(1)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Attention mechanism to weight different scales
        self.attention = nn.Sequential(
            nn.Linear(target_channels * self.num_scales, target_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(target_channels, self.num_scales),
            nn.Softmax(dim=1)
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(target_channels, target_channels),
            nn.BatchNorm1d(target_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(target_channels, target_channels // 2),
            nn.BatchNorm1d(target_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(target_channels // 2, target_channels // 4),
            nn.BatchNorm1d(target_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(target_channels // 4, num_classes)
        )
    
    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List of feature tensors from encoder stages
        Returns:
            Classification logits [B, num_classes]
        """
        if len(encoder_features) < self.num_scales:
            raise ValueError(f"Expected at least {self.num_scales} encoder features, "
                           f"got {len(encoder_features)}")
        
        # Process multi-scale features
        multi_scale_features = []
        
        for feat, adapter in zip(encoder_features[-self.num_scales:], self.feature_adapters):
            # Adapt features to common channel size and spatial resolution
            adapted = adapter(feat)
            # Global pooling to get feature vector
            pooled = self.global_pool(adapted).flatten(1)
            multi_scale_features.append(pooled)
        
        # Stack all scale features
        stacked_features = torch.stack(multi_scale_features, dim=1)  # [B, num_scales, target_channels]
        
        # Concatenate for attention computation
        concat_features = torch.cat(multi_scale_features, dim=1)  # [B, num_scales * target_channels]
        
        # Compute attention weights for different scales
        attention_weights = self.attention(concat_features)  # [B, num_scales]
        
        # Apply attention weights to aggregate multi-scale features
        attention_weights = attention_weights.unsqueeze(-1)  # [B, num_scales, 1]
        weighted_features = (stacked_features * attention_weights).sum(dim=1)  # [B, target_channels]
        
        # Feature fusion
        fused_features = self.feature_fusion(weighted_features)
        
        # Final classification
        logits = self.classifier(fused_features)
        
        return logits

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

def load_classification_head(checkpoint_path: Path, encoder_channels: List[int], device: torch.device):
    """Load the MultiScaleClassificationHead from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cls_state_dict = checkpoint.get("cls_state_dict", None)
    
    if cls_state_dict is None:
        raise ValueError("No 'cls_state_dict' found in checkpoint. "
                        "Make sure you trained with the multi-task trainer.")
    
    # Create the full MultiScaleClassificationHead
    classifier = MultiScaleClassificationHead(
        encoder_channels=encoder_channels,
        num_classes=3,
        dim=3,
        target_channels=256,
        spatial_reduction=4
    ).to(device)
    
    # Load the saved weights
    try:
        classifier.load_state_dict(cls_state_dict, strict=True)
        print("✅ Successfully loaded MultiScaleClassificationHead weights")
    except Exception as e:
        raise ValueError(f"Error loading classification head: {e}")
    
    return classifier

def predict_classification(input_dir: Path, output_file: Path, model_dir: Path, 
                         checkpoint_name: str = "checkpoint_best.pth"):
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
    encoder_channels = network.encoder.output_channels
    classifier = load_classification_head(checkpoint_path, encoder_channels, device)
    
    # Setup feature capture hooks for all encoder stages
    encoder_features = []

    def create_hook(stage_idx):
        def hook_fn(module, input, output):
            while len(encoder_features) <= stage_idx:
                encoder_features.append(None)
            encoder_features[stage_idx] = output
        return hook_fn

    # Register hooks for all encoder stages
    hooks = []
    for i, stage in enumerate(network.encoder.stages):
        hook = stage.register_forward_hook(create_hook(i))
        hooks.append(hook)
    
    # Set models to evaluation mode
    network.eval()
    classifier.eval()
    torch.set_grad_enabled(False)
    
    # Find test images (handle both flat directory and subfolders)
    case_files = []
    
    # Check for subfolder structure (validation case)
    for subfolder in ["subtype0", "subtype1", "subtype2"]:
        subfolder_path = input_dir / subfolder
        if subfolder_path.exists():
            files = list(subfolder_path.glob("*_0000.nii.gz"))
            case_files.extend(files)
    
    # If no subfolders, check flat directory (test case)
    if not case_files:
        case_files = list(input_dir.glob("*_0000.nii.gz"))
    
    case_files = sorted(case_files)
    print(f"Found {len(case_files)} files for classification")
    
    if len(case_files) == 0:
        print("Warning: No files found with pattern '*_0000.nii.gz'")
        return
    
    # Process each case
    results = [("Names", "Subtype")]  # CSV header
    temp_output_dir = output_file.parent / "temp_seg_output"
    temp_output_dir.mkdir(exist_ok=True)
    
    for img_path in case_files:
        case_id = img_path.name.replace("_0000.nii.gz", "") + ".nii.gz"
        print(f"Processing: {img_path.name}")
        
        # Reset captured features
        encoder_features.clear()
        
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
        
        # Perform classification using MultiScaleClassificationHead
        if len(encoder_features) >= 3:
            try:
                # Filter out None values
                valid_features = [f for f in encoder_features if f is not None]
                
                if len(valid_features) >= 3:
                    # 🔧 FIX: Convert features to float32 to match classifier weights
                    valid_features_float = [f.float() for f in valid_features]
                    logits = classifier(valid_features_float)
                    prediction = int(torch.argmax(logits, dim=1).item())
                else:
                    print(f"Warning: Insufficient features captured for {img_path.name}")
                    prediction = 0
                    
            except Exception as e:
                print(f"Classification error for {img_path.name}: {e}")
                prediction = 0
        else:
            print(f"Warning: No encoder features captured for {img_path.name}")
            prediction = 0
        
        results.append((case_id, prediction))
        print(f"  ✅ Classification: {prediction}")
    
    # Cleanup hooks
    for hook in hooks:
        hook.remove()
    
    # Clean up temp directory
    if temp_output_dir.exists():
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
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pth",
                       help="Checkpoint filename (default: checkpoint_best.pth)")
    
    args = parser.parse_args()
    
    predict_classification(
        input_dir=args.input_dir,
        output_file=args.output_file,
        model_dir=args.model_dir,
        checkpoint_name=args.checkpoint
    )

if __name__ == "__main__":
    main()
