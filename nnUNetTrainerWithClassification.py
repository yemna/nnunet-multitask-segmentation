"""
This file inherits from nnUNetTrainer and adds a classification head to the nnUNet architecture.
so that we can train a nnUNet model for both segmentation and classification tasks.
The nnUNetTrainerWithClassification class builds upon the nnUNetTrainer class and modifies
the network architecture to include a classification head.
It also overrides the train, and validation step to compute both segmentation and classification losses.
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.network_initialization import InitWeights_He
import torch
import torch.nn as nn
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.collate_outputs import collate_outputs
import numpy as np
from torch import distributed as dist
from typing import Tuple, Union, List
import os
import pandas as pd
from sklearn.metrics import f1_score
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from sklearn.utils.class_weight import compute_class_weight
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler


class ClassificationHead(nn.Module):
    def __init__(self, encoder_channels, num_classes=3, hidden_dim=256, dropout_p=0.3, use_all_features=False):
        super().__init__()
        
        # =================== TOGGLE HERE ===================
        # Change use_all_features to True for all features
        # Change use_all_features to False for last 3 features
        self.use_all_features = use_all_features
        
        if self.use_all_features:
            # USE ALL FEATURES
            selected_channels = encoder_channels
            print(f"Using ALL {len(selected_channels)} encoder features: {selected_channels}")
        else:
            # USE LAST 3 FEATURES (original)
            if len(encoder_channels) < 3:
                raise ValueError("Encoder must have at least 3 feature maps")
            selected_channels = encoder_channels[-3:]
            print(f"Using LAST 3 encoder features: {selected_channels}")
        
        # Feature compression (adapts to actual channel dims)
        self.feature_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, 64, kernel_size=1, bias=False),  # Reduce to 64 channels
                nn.BatchNorm3d(64),
                nn.GELU(),
                nn.AdaptiveAvgPool3d((4, 4, 4))  # Downsample to (4,4,4)
            ) for ch in selected_channels
        ])
        
        # Enhanced fusion - DYNAMIC based on number of features
        fusion_input_channels = 64 * len(selected_channels)
        self.fusion = nn.Sequential(
            nn.Conv3d(fusion_input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, encoder_features):
        if self.use_all_features:
            # Process ALL features in order
            if len(encoder_features) < 1:
                raise ValueError(f"Expected ≥1 feature maps, got {len(encoder_features)}")
            
            adapted = []
            for i, adapter in enumerate(self.feature_adapters):
                feat = encoder_features[i]  # Use ALL features
                adapted.append(adapter(feat))
        else:
            # Process LAST 3 features (original)
            if len(encoder_features) < 3:
                raise ValueError(f"Expected ≥3 feature maps, got {len(encoder_features)}")
            
            adapted = []
            for i, adapter in enumerate(self.feature_adapters):
                feat = encoder_features[-(3 - i)]  # Get features from last to -3
                adapted.append(adapter(feat))
        
        # Concatenate and fuse
        x = torch.cat(adapted, dim=1)
        x = self.fusion(x).flatten(1)
        
        return self.classifier(x)


class nnUNetTrainerWithClassification(nnUNetTrainer):

    def initialize(self, *args, **kwargs):
        # Call the base class's initialize method first
        super().initialize(*args, **kwargs)

       
        
        # Try BOTH paths to see which one exists:
        possible_paths = [
            'D:/nnunet_with_classification/data/nnUNet_preprocessed/Dataset777_M31Quiz/classification_labels_named.csv'
            
        ]
        
        subtype_file = None
        for path in possible_paths:
            if os.path.isfile(path):
                subtype_file = path
                print(f"Found subtype file: {subtype_file}")
                break
        
        if subtype_file is None:
            # List what's actually in the directory
            for dataset_path in ['nnUNet_preprocessed/Dataset777_M31Quiz/', 'nnUNet_preprocessed/Dataset777_PancreasMultiTask/']:
                if os.path.exists(dataset_path):
                    files = os.listdir(dataset_path)
                    print(f"Files in {dataset_path}:")
                    for f in files:
                        if f.endswith('.csv'):
                            print(f"  - {f}")
            
            raise FileNotFoundError(f"Could not find subtype file in any of: {possible_paths}")

        df = pd.read_csv(subtype_file)
        self.subtype_dict = {
            row['Name'].replace(".nii.gz", ""): int(row['Subtype'])
            for _, row in df.iterrows()
        }
        
        # Debug: Print class distribution
        class_counts = df['Subtype'].value_counts().sort_index()
        print(f"Class distribution: {dict(class_counts)}")
        print(f"Total samples: {len(df)}")
    
    def build_network_architecture(self, architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels, enable_deep_supervision=True):
        network = super().build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, enable_deep_supervision
        )

        encoder_output_channels = network.encoder.output_channels
        print(f"Encoder channels: {encoder_output_channels}")
        
        # =================== TOGGLE HERE ===================
        # Change use_all_features=True for ALL features
        # Change use_all_features=False for LAST 3 features
        network.ClassificationHead = ClassificationHead(
            encoder_output_channels,
            num_classes=3,
            use_all_features=True  # <-- CHANGE THIS TO True FOR ALL FEATURES
        ).to(self.device)
        return network

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']

        # get the subtype from the batch
        case_ids = batch['keys']
        subtype = torch.tensor(
            [self.subtype_dict[k] for k in case_ids],
            dtype=torch.long,
            device=self.device
        )

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        if subtype is not None:
            subtype = subtype.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_output = self.network(data)
            enc_features = self.network.encoder(data)
            class_logits = self.network.ClassificationHead(enc_features)

            seg_loss = self.loss(seg_output, target)

            if subtype is not None:
                class_loss = nn.CrossEntropyLoss()(class_logits, subtype.long())   
                total_loss = seg_loss + 1.0 * class_loss  # Weighted combo
            else:
                total_loss = seg_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': total_loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']

        # get the subtype from the batch
        case_ids = batch['keys']
        subtype = torch.tensor(
            [self.subtype_dict[k] for k in case_ids],
            dtype=torch.long,
            device=self.device
        )

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        if subtype is not None:
            subtype = subtype.to(self.device, non_blocking=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_output = self.network(data)
            enc_features = self.network.encoder(data)
            class_logits = self.network.ClassificationHead(enc_features)

            seg_loss = self.loss(seg_output, target)

            if subtype is not None:
                class_loss = nn.CrossEntropyLoss()(class_logits, subtype.long())
                total_loss = seg_loss + 0.3 * class_loss
                preds = torch.argmax(class_logits, dim=1)
                f1 = f1_score(subtype.cpu().numpy(), preds.cpu().numpy(), average='macro')
            else:
                total_loss = seg_loss
                f1 = 0

        result = {
            'loss': total_loss.detach().cpu().numpy(),
            'classification_f1': f1
        }

        # Keep original segmentation dice computation
        if self.enable_deep_supervision:
            seg_output = seg_output[0]
            target = target[0]

        axes = [0] + list(range(2, seg_output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(seg_output) > 0.5).long()
        else:
            output_seg = seg_output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(seg_output.shape, device=seg_output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:] if target.dtype != torch.bool else ~target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
        result.update({
            'tp_hard': tp.detach().cpu().numpy(),
            'fp_hard': fp.detach().cpu().numpy(),
            'fn_hard': fn.detach().cpu().numpy()
        })

        return result
    
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)

        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            # Gather segmentation stats
            for name, var in zip(['tp', 'fp', 'fn'], [tp, fp, fn]):
                gathered = [None for _ in range(world_size)]
                dist.all_gather_object(gathered, var)
                locals()[name] = np.vstack([i[None] for i in gathered]).sum(0)

            # Gather losses
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()

            # f1
            f1s = [None for _ in range(world_size)]
            dist.all_gather_object(f1s, outputs_collated['classification_f1'])
            f1 = np.mean(f1s)
        else:
            loss_here = np.mean(outputs_collated['loss'])
            f1 = np.mean(outputs_collated['classification_f1'])

        global_dc_per_class = [2 * i / (2 * i + j + k) if (2 * i + j + k) > 0 else 0 for i, j, k in zip(tp, fp, fn)]
        mean_fg_dice = np.nanmean(global_dc_per_class)

        # Store segmentation metrics
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

        # Store F1 values
        if 'classification_f1' not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging['classification_f1'] = []

        self.logger.my_fantastic_logging['classification_f1'].append(f1)

    def on_epoch_end(self):
        super().on_epoch_end()
        # Log classification F1
        if 'classification_f1' in self.logger.my_fantastic_logging:
            f1 = self.logger.my_fantastic_logging['classification_f1'][-1]
            self.print_to_log_file(f"Classification Macro F1: {np.round(f1, 4)}")
        else:
            f1 = 0.0

        # Get segmentation score
        dice_per_class = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
        whole_pancreas_dsc = np.mean(dice_per_class[1:])  # label 1 and 2

        # Create a combined score metric (customizable)
        combined_score = (whole_pancreas_dsc + f1) / 2

        # Initialize if not yet done
        if not hasattr(self, '_best_combined_score'):
            self._best_combined_score = -np.inf

        # Save model only if improved - RELAXED THRESHOLDS
        if combined_score > self._best_combined_score and whole_pancreas_dsc > 0.5 and f1 > 0.3:
            self._best_combined_score = combined_score
            self.print_to_log_file(
                f"New best model: whole_dsc={whole_pancreas_dsc:.4f}, f1={f1:.4f}, combined={combined_score:.4f}"
            )
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best_combined.pth'))