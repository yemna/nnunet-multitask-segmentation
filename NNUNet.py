import os
from typing import List, Dict, Union, Tuple
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import torch.amp

from torch.cuda.amp import autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

# ------------------ Classification Debugger ------------------

class ClassificationDebugger:
    """Debug tool to identify which class is not being predicted"""
    
    def __init__(self):
        self.all_predictions = []
        self.all_targets = []
        self.all_logits = []
    
    def log_batch(self, logits, targets):
        """Call this in train_step and validation_step"""
        with torch.no_grad():
            predictions = logits.argmax(1).cpu().numpy()
            targets_np = targets.cpu().numpy()
            logits_np = logits.cpu().numpy()
            
            self.all_predictions.extend(predictions)
            self.all_targets.extend(targets_np)
            self.all_logits.extend(logits_np)
    
    def analyze_epoch(self, epoch_name=""):
        """Call this at end of each epoch"""
        if not self.all_predictions:
            return
            
        pred_counts = Counter(self.all_predictions)
        target_counts = Counter(self.all_targets)
        
        print(f"\n=== {epoch_name} CLASSIFICATION ANALYSIS ===")
        print("Target distribution:", dict(target_counts))
        print("Prediction distribution:", dict(pred_counts))
        
        # Check which classes are never predicted
        all_classes = set(range(3))  # 0, 1, 2
        predicted_classes = set(self.all_predictions)
        missing_classes = all_classes - predicted_classes
        
        if missing_classes:
            print(f"NEVER PREDICTED CLASSES: {missing_classes}")
        
        # Per-class accuracy
        targets_np = np.array(self.all_targets)
        preds_np = np.array(self.all_predictions)
        
        for cls in range(3):
            mask = (targets_np == cls)
            if mask.sum() > 0:
                correct = ((preds_np == targets_np) & mask).sum()
                total = mask.sum()
                acc = correct / total
                print(f"Class {cls}: {correct}/{total} = {acc:.3f}")
        
        # Check logits distribution
        logits_np = np.array(self.all_logits)
        mean_logits = logits_np.mean(axis=0)
        std_logits = logits_np.std(axis=0)
        
        print("Mean logits per class:", [f"{x:.3f}" for x in mean_logits])
        print("Std logits per class:", [f"{x:.3f}" for x in std_logits])
        
        # Check if one class has much lower logits (bias issue)
        if len(mean_logits) == 3:
            if max(mean_logits) - min(mean_logits) > 2.0:
                print("LARGE LOGIT IMBALANCE - possible bias issue!")
        
        # Clear for next epoch
        self.all_predictions = []
        self.all_targets = []
        self.all_logits = []

# ------------------ Utils: pooling & losses ------------------

def compute_cls_accuracy(logits, targets):
    """
    logits: torch.Tensor, shape [B, num_classes] 
    targets: torch.Tensor, shape [B]
    """
    if logits.ndim > 1 and logits.shape[1] > 1:
        preds = torch.argmax(logits, dim=1)
    else:
        preds = (logits > 0.5).long()  # binary case
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0

class GeM(nn.Module):
    """Generalized Mean Pooling (works better than plain GAP sometimes)."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6, dim: int = 3):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        if dim == 3:
            self.pool = nn.AdaptiveAvgPool3d(1)
        elif dim == 2:
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError("dim must be 2 or 3")

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = self.pool(x)
        return x.pow(1.0 / self.p)

class FocalLoss(nn.Module):
    """
    Multi-class focal loss (softmax).
    gamma>0 focuses on hard samples. alpha can rebalance classes (len==C or scalar).
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer('alpha', torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        # logits: (B,C), target: (B,)
        logpt = F.log_softmax(logits, dim=1)
        pt = torch.exp(logpt)
        loss = - (1 - pt).pow(self.gamma) * logpt
        loss = loss.gather(1, target[:, None]).squeeze(1)  # (B,)
        if hasattr(self, 'alpha') and self.alpha is not None:
            a = self.alpha[target] if self.alpha.ndim > 0 else self.alpha
            loss = a * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# ------------------ Multi-Scale Classification Head ------------------

class MultiScaleClassificationHead(nn.Module):
    """
    Multi-scale feature fusion classification head with attention mechanism
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
        
        # Final classifier with multiple layers for better representation
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
        
        # Proper initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for the final classification layer
        final_layer = self.classifier[-1]
        nn.init.normal_(final_layer.weight, 0, 0.01)
        nn.init.constant_(final_layer.bias, 0)
    
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

# ------------------ Enhanced Trainer ------------------

class NNUNet(nnUNetTrainer):
    """
    Enhanced multitask trainer with multi-scale classification head and improved debugging
    """

    # Configuration
    num_epochs = 200
    cls_loss_weight: float = 2.5  # Slightly increased for better balance
    num_classes_cls: int = 3
    labels_csv_name: str = "classification_labels.csv"

    # Classification loss choice
    use_focal: bool = False
    focal_gamma: float = 2.0
    focal_alpha: Union[float, List[float], None] = None

    # Early stopping
    early_stop_patience: int = 20  # Increased patience
    early_stop_min_delta: float = 1e-3

    # Training iterations
    num_iterations_per_epoch = 120
    num_val_iterations_per_epoch = 24

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        # Initialize parent class
        ok = False
        for kwargs in (
            dict(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, device=device),
            dict(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json),
        ):
            try:
                super().__init__(**kwargs)
                ok = True
                break
            except TypeError:
                continue
        if not ok:
            raise

        # Classification state
        self.classifier: nn.Module = None
        self._enc_feat_hooks: List = []
        self._enc_features: List[torch.Tensor] = []
        self._spatial_dims: int = None
        self._labels_map: Dict[str, int] = None

        # Initialize debugger
        self.cls_debugger = ClassificationDebugger()

        # Class weights based on your distribution
        # Class 0: 71 samples (24.7%), Class 1: 121 samples (42.0%), Class 2: 96 samples (33.3%)
        total_samples = 71 + 121 + 96
        class_weights = torch.tensor([
            total_samples / (3 * 71),   # 1.35 for class 0
            total_samples / (3 * 121),  # 0.79 for class 1  
            total_samples / (3 * 96)    # 1.00 for class 2
        ])

        # Initialize loss function
        if self.use_focal:
            self.cls_loss_fn = FocalLoss(gamma=self.focal_gamma, alpha=class_weights.tolist())
        else:
            self.cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Early stopping state
        self._best_val = float('inf')
        self._es_wait = 0
        self._es_stop = False
        
        # Classification tracking
        self.train_cls_accuracies = []
        self.train_cls_losses = []
        self.val_cls_accuracies = []
        self.val_cls_losses = []

    # ---------- Labels ----------
    def _find_labels_csv(self) -> str:
        cands = []
        if getattr(self, 'preprocessed_dataset_folder_base', None):
            cands.append(os.path.join(self.preprocessed_dataset_folder_base, self.labels_csv_name))
        if getattr(self, 'preprocessed_dataset_folder', None):
            cands.append(os.path.join(os.path.dirname(self.preprocessed_dataset_folder), self.labels_csv_name))
        env_prep = os.environ.get('nnUNet_preprocessed', '')
        if env_prep:
            cands.append(os.path.join(env_prep, self.plans_manager.dataset_name, self.labels_csv_name))
        if getattr(self, 'output_folder_base', None):
            cands.append(os.path.join(self.output_folder_base, self.labels_csv_name))
        for p in cands:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"Could not find {self.labels_csv_name} in: " + " | ".join(cands))

    def _load_classification_labels(self):
        path = self._find_labels_csv()
        labels = {}
        class_counts = {0: 0, 1: 0, 2: 0}
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                line = line.replace('\t', ',')
                parts = [x.strip() for x in line.split(',')]
                if len(parts) < 2: continue
                k, v = parts[0], int(parts[1])
                labels[k] = v
                if v in class_counts:
                    class_counts[v] += 1
        
        self._labels_map = labels
        self.print_to_log_file(f"Loaded {len(labels)} classification labels from {path}")
        self.print_to_log_file(f"Label distribution: {class_counts}")

    def _fetch_cls_targets(self, keys: List[str]) -> torch.Tensor:
        """Enhanced version with debugging"""
        if self._labels_map is None:
            self._load_classification_labels()

        ys = []
        miss = 0
        class_counts = {0: 0, 1: 0, 2: 0}
        
        for k in keys:
            name = k
            if name.endswith(".npz"): name = name[:-4]
            if name.endswith(".nii.gz"): name = name[:-7]
            if name.endswith("_0000"): name = name[:-5]
            
            if name in self._labels_map:
                label = self._labels_map[name]
                ys.append(label)
                if label in class_counts:
                    class_counts[label] += 1
            else:
                miss += 1
                ys.append(1)  # Default to class 1
                class_counts[1] += 1
        
        # Debug print every few batches
        if hasattr(self, '_debug_batch_count'):
            self._debug_batch_count += 1
        else:
            self._debug_batch_count = 0
        
        if self._debug_batch_count % 50 == 0:
            self.print_to_log_file(f"Batch targets - Class 0: {class_counts[0]}, Class 1: {class_counts[1]}, Class 2: {class_counts[2]}")
        
        if miss:
            self.print_to_log_file(f"[WARN] {miss} keys missing in labels CSV (using default class=1).")
        
        return torch.as_tensor(ys, dtype=torch.long, device=self.device)

    # ---------- Lifecycle ----------
    def on_train_start(self):
        super().on_train_start()

        # Spatial dims from patch size
        self._spatial_dims = len(self.configuration_manager.patch_size)

        # Get encoder channels from the network
        net = self.network.module if self.is_ddp else self.network
        encoder_channels = net.encoder.output_channels
        
        self.print_to_log_file(f"Encoder channels: {encoder_channels}")
        self.print_to_log_file(f"Spatial dimensions: {self._spatial_dims}")

        # Build multi-scale classification head
        self.classifier = MultiScaleClassificationHead(
            encoder_channels=encoder_channels,
            num_classes=self.num_classes_cls,
            dim=self._spatial_dims,
            target_channels=256,
            spatial_reduction=4
        ).to(self.device)

        # Move class weights to device
        if hasattr(self.cls_loss_fn, 'weight') and self.cls_loss_fn.weight is not None:
            self.cls_loss_fn.weight = self.cls_loss_fn.weight.to(self.device)

        # Set up hooks to capture multi-scale encoder features
        self._setup_feature_hooks()

        # Add classifier parameters to optimizer with appropriate learning rate
        self.optimizer.add_param_group({
            'params': self.classifier.parameters(),
            'lr': self.initial_lr * 1.5,  # Slightly higher LR for classification head
            'weight_decay': 1e-4
        })

        total_params = sum(p.numel() for p in self.classifier.parameters())
        trainable_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        self.print_to_log_file(f"Multi-scale classification head - Total params: {total_params}, Trainable: {trainable_params}")

    def _setup_feature_hooks(self):
        """Set up hooks to capture encoder features from multiple scales"""
        net = self.network.module if self.is_ddp else self.network
        
        # Clear existing hooks
        for hook in self._enc_feat_hooks:
            hook.remove()
        self._enc_feat_hooks = []
        
        def create_hook(stage_idx):
            def hook_fn(module, input, output):
                # Ensure we have enough space in the list
                while len(self._enc_features) <= stage_idx:
                    self._enc_features.append(None)
                self._enc_features[stage_idx] = output
            return hook_fn
        
        # Register hooks for all encoder stages
        for i, stage in enumerate(net.encoder.stages):
            hook = stage.register_forward_hook(create_hook(i))
            self._enc_feat_hooks.append(hook)

    def on_train_end(self):
        # Clean up hooks
        for hook in self._enc_feat_hooks:
            hook.remove()
        self._enc_feat_hooks = []
        return super().on_train_end()

    # ---------- Forward helpers ----------
    def _classification_forward(self) -> torch.Tensor:
        """Forward pass through the multi-scale classification head"""
        if not self._enc_features:
            raise RuntimeError("No encoder features captured. Check if hooks are properly set up.")
        
        # Filter out None values and ensure we have valid features
        valid_features = [f for f in self._enc_features if f is not None]
        if len(valid_features) < 3:
            raise RuntimeError(f"Insufficient encoder features captured: {len(valid_features)} < 3")
        
        return self.classifier(valid_features)

    # ---------- Training steps ----------
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        keys: Union[List[str], None] = batch.get('keys', None)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=(self.device.type=='cuda')):
            # Clear previous features
            self._enc_features = []
            
            # Forward pass through segmentation network (triggers feature capture via hooks)
            seg_logits = self.network(data)
            seg_loss = self.loss(seg_logits, target)

            if keys is not None:
                cls_targets = self._fetch_cls_targets(keys)
                cls_logits = self._classification_forward()
                
                # DEBUG: Log predictions
                self.cls_debugger.log_batch(cls_logits, cls_targets)
                
                cls_loss = self.cls_loss_fn(cls_logits, cls_targets)
                cls_acc = (cls_logits.argmax(1) == cls_targets).float().mean()
            else:
                cls_loss = torch.zeros((), device=self.device)
                cls_acc = torch.zeros((), device=self.device)

            total_loss = seg_loss + self.cls_loss_weight * cls_loss

        # Gradient computation and optimization
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            # Gradient clipping for both networks
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 12)
            self.optimizer.step()

        result = {'loss': float(total_loss.detach().cpu())}
        
        if keys is not None and cls_loss.item() > 0:
            result['cls_loss_scalar'] = float(cls_loss.item())
            result['cls_acc_scalar'] = float(cls_acc.item())
            result['batch_size'] = len(keys)
        
        return result

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        keys: Union[List[str], None] = batch.get('keys', None)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(enabled=(self.device.type=='cuda')):
            # Clear previous features
            self._enc_features = []
            
            # Forward pass
            seg_logits = self.network(data)
            seg_loss = self.loss(seg_logits, target)

            # For pseudo-dice metrics
            out_top = seg_logits[0] if isinstance(seg_logits, (list, tuple)) else seg_logits
            tgt_top = target[0] if isinstance(target, list) else target

            if keys is not None:
                cls_targets = self._fetch_cls_targets(keys)
                cls_logits = self._classification_forward()
                
                # DEBUG: Log predictions  
                self.cls_debugger.log_batch(cls_logits, cls_targets)
                
                cls_loss = self.cls_loss_fn(cls_logits, cls_targets)
                cls_acc = (cls_logits.argmax(1) == cls_targets).float().mean()
            else:
                cls_loss = torch.zeros((), device=self.device)
                cls_acc = torch.zeros((), device=self.device)

            total_loss = seg_loss + self.cls_loss_weight * cls_loss

        # Pseudo-dice calculation (unchanged from original)
        axes = [0] + list(range(2, out_top.ndim))
        if self.label_manager.has_regions:
            pred_1hot = (torch.sigmoid(out_top) > 0.5).long()
        else:
            seg = out_top.argmax(1)[:, None]
            pred_1hot = torch.zeros_like(out_top, dtype=torch.float32)
            pred_1hot.scatter_(1, seg, 1)

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (tgt_top != self.label_manager.ignore_label).float()
                tgt_top = tgt_top.clone()
                tgt_top[tgt_top == self.label_manager.ignore_label] = 0
            else:
                if tgt_top.dtype == torch.bool:
                    mask = ~tgt_top[:, -1:]
                else:
                    mask = 1 - tgt_top[:, -1:]
                tgt_top = tgt_top[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(pred_1hot, tgt_top, axes=axes, mask=mask)
        tp = tp.detach().cpu().numpy(); fp = fp.detach().cpu().numpy(); fn = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp, fp, fn = tp[1:], fp[1:], fn[1:]

        result = {'loss': float(total_loss.detach().cpu()),
                  'tp_hard': tp, 'fp_hard': fp, 'fn_hard': fn}
        
        if keys is not None and cls_loss.item() > 0:
            result['cls_loss_scalar'] = float(cls_loss.item())
            result['cls_acc_scalar'] = float(cls_acc.item())
            result['batch_size'] = len(keys)
        
        return result

    def on_train_epoch_end(self, train_outputs: List[dict]):
        """Called after each training epoch"""
        super().on_train_epoch_end(train_outputs)
        
        # Compute average training classification metrics
        cls_losses = [out['cls_loss_scalar'] for out in train_outputs if 'cls_loss_scalar' in out]
        cls_accs = [out['cls_acc_scalar'] for out in train_outputs if 'cls_acc_scalar' in out]
        
        if cls_losses:
            train_cls_loss = sum(cls_losses) / len(cls_losses)
            train_cls_acc = sum(cls_accs) / len(cls_accs)
            
            self.train_cls_accuracies.append(train_cls_acc)
            self.train_cls_losses.append(train_cls_loss)
            
            self.print_to_log_file(f"Epoch {self.current_epoch}: Train cls_acc={train_cls_acc:.4f}, "
                                 f"cls_loss={train_cls_loss:.4f}")
        
        # Debug analysis
        self.cls_debugger.analyze_epoch(f"TRAIN Epoch {self.current_epoch}")

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """Enhanced validation end with classification metrics"""
        from nnunetv2.utilities.collate_outputs import collate_outputs
        outputs = collate_outputs(val_outputs)
        tp = np.sum(outputs['tp_hard'], 0)
        fp = np.sum(outputs['fp_hard'], 0)
        fn = np.sum(outputs['fn_hard'], 0)
        loss_here = np.mean(outputs['loss'])

        global_dc_per_class = [2 * i / (2 * i + j + k) if (2 * i + j + k) > 0 else np.nan
                               for i, j, k in zip(tp, fp, fn)]
        mean_fg_dice = np.nanmean(global_dc_per_class)

        self.logger.log('mean_fg_dice', float(mean_fg_dice), self.current_epoch)
        self.logger.log('dice_per_class_or_region', [float(x) for x in global_dc_per_class], self.current_epoch)
        self.logger.log('val_losses', float(loss_here), self.current_epoch)
        
        # Compute classification metrics
        cls_losses = [out['cls_loss_scalar'] for out in val_outputs if 'cls_loss_scalar' in out]
        cls_accs = [out['cls_acc_scalar'] for out in val_outputs if 'cls_acc_scalar' in out]
        
        if cls_losses:
            val_cls_loss = sum(cls_losses) / len(cls_losses)
            val_cls_acc = sum(cls_accs) / len(cls_accs)
            
            self.val_cls_accuracies.append(val_cls_acc)
            self.val_cls_losses.append(val_cls_loss)
            
            self.print_to_log_file(f"Epoch {self.current_epoch}: Val cls_acc={val_cls_acc:.4f}, "
                                 f"cls_loss={val_cls_loss:.4f}")
            
            # Log classification metrics to logger
            #self.logger.log('val_cls_accuracy', float(val_cls_acc), self.current_epoch)
            #self.logger.log('val_cls_loss', float(val_cls_loss), self.current_epoch)

        # Debug analysis
        self.cls_debugger.analyze_epoch(f"VAL Epoch {self.current_epoch}")

        # Early stopping logic
        improved = (self._best_val - loss_here) > self.early_stop_min_delta
        if improved:
            self._best_val = loss_here
            self._es_wait = 0
        else:
            self._es_wait += 1

        self.print_to_log_file(f"[EarlyStopping-VAL] val_total={loss_here:.6f} "
                               f"best={self._best_val:.6f} wait={self._es_wait}/{self.early_stop_patience}")
        if self._es_wait >= self.early_stop_patience:
            self.print_to_log_file("[EarlyStopping-VAL] Patience reached. Stopping after this epoch.")
            self._es_stop = True
            self.terminate_after_epoch = True

    # Explicit training loop for early stopping
    def run_training(self):
        self.on_train_start()
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for _ in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for _ in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
            if self._es_stop or getattr(self, "terminate_after_epoch", False):
                break
        self.on_train_end()

    # ---------- Save/Load with classification state ----------
    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        if self.local_rank == 0:
            ckpt = torch.load(filename, map_location=self.device)
            ckpt['cls_state_dict'] = self.classifier.state_dict() if self.classifier is not None else None
            ckpt['cls_metrics'] = {
                'train_cls_accuracies': self.train_cls_accuracies,
                'train_cls_losses': self.train_cls_losses,
                'val_cls_accuracies': self.val_cls_accuracies,
                'val_cls_losses': self.val_cls_losses,
            }
            # Save classification configuration for reproducibility
            ckpt['cls_config'] = {
                'cls_loss_weight': self.cls_loss_weight,
                'num_classes_cls': self.num_classes_cls,
                'use_focal': self.use_focal,
                'focal_gamma': self.focal_gamma,
                'focal_alpha': self.focal_alpha,
                'early_stop_patience': self.early_stop_patience,
                'spatial_dims': self._spatial_dims,
            }
            torch.save(ckpt, filename)

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        super().load_checkpoint(filename_or_checkpoint)
        if isinstance(filename_or_checkpoint, str):
            ckpt = torch.load(filename_or_checkpoint, map_location=self.device)
        else:
            ckpt = filename_or_checkpoint
            
        # Load classification state
        cls_sd = ckpt.get('cls_state_dict', None)
        if cls_sd is not None and self.classifier is not None:
            try:
                self.classifier.load_state_dict(cls_sd, strict=False)
                self.print_to_log_file("Loaded classification head state dict")
            except Exception as e:
                self.print_to_log_file(f"Warning: Could not load classification state dict: {e}")
        
        # Load classification metrics
        if 'cls_metrics' in ckpt:
            metrics = ckpt['cls_metrics']
            self.train_cls_accuracies = metrics.get('train_cls_accuracies', [])
            self.train_cls_losses = metrics.get('train_cls_losses', [])
            self.val_cls_accuracies = metrics.get('val_cls_accuracies', [])
            self.val_cls_losses = metrics.get('val_cls_losses', [])
            self.print_to_log_file("Loaded classification metrics")
        
        # Load classification configuration
        if 'cls_config' in ckpt:
            cls_config = ckpt['cls_config']
            self.print_to_log_file(f"Loaded classification config: {cls_config}")
    
    # ---------- Additional utility methods ----------
    def get_classification_summary(self) -> Dict:
        """Get a summary of classification performance"""
        summary = {
            'train_cls_accuracies': self.train_cls_accuracies,
            'train_cls_losses': self.train_cls_losses,
            'val_cls_accuracies': self.val_cls_accuracies,
            'val_cls_losses': self.val_cls_losses,
            'best_val_cls_acc': max(self.val_cls_accuracies) if self.val_cls_accuracies else 0.0,
            'best_train_cls_acc': max(self.train_cls_accuracies) if self.train_cls_accuracies else 0.0,
            'current_epoch': self.current_epoch,
            'cls_loss_weight': self.cls_loss_weight,
        }
        return summary
    
    def print_classification_summary(self):
        """Print a summary of classification training progress"""
        summary = self.get_classification_summary()
        
        self.print_to_log_file("\n" + "="*60)
        self.print_to_log_file("CLASSIFICATION TRAINING SUMMARY")
        self.print_to_log_file("="*60)
        self.print_to_log_file(f"Current Epoch: {summary['current_epoch']}")
        self.print_to_log_file(f"Classification Loss Weight: {summary['cls_loss_weight']}")
        self.print_to_log_file(f"Best Validation Accuracy: {summary['best_val_cls_acc']:.4f}")
        self.print_to_log_file(f"Best Training Accuracy: {summary['best_train_cls_acc']:.4f}")
        
        if self.val_cls_accuracies:
            recent_val_acc = np.mean(self.val_cls_accuracies[-5:])  # Last 5 epochs
            self.print_to_log_file(f"Recent Validation Accuracy (last 5 epochs): {recent_val_acc:.4f}")
        
        if self.train_cls_accuracies:
            recent_train_acc = np.mean(self.train_cls_accuracies[-5:])  # Last 5 epochs
            self.print_to_log_file(f"Recent Training Accuracy (last 5 epochs): {recent_train_acc:.4f}")
        
        self.print_to_log_file("="*60 + "\n")

    def on_epoch_end(self):
        """Enhanced epoch end with classification summary"""
        super().on_epoch_end()
        
        # Print classification summary every 10 epochs
        if (self.current_epoch + 1) % 10 == 0:
            self.print_classification_summary()
    
    # ---------- Inference methods ----------
    def predict_single_case(self, data: torch.Tensor, return_classification: bool = True):
        """
        Predict both segmentation and classification for a single case
        
        Args:
            data: Input tensor [1, C, D, H, W] or [1, C, H, W]
            return_classification: Whether to return classification prediction
            
        Returns:
            tuple: (segmentation_prediction, classification_prediction) or just segmentation_prediction
        """
        self.network.eval()
        if self.classifier is not None:
            self.classifier.eval()
        
        with torch.no_grad():
            # Clear features
            self._enc_features = []
            
            # Segmentation prediction
            seg_logits = self.network(data)
            
            if isinstance(seg_logits, (list, tuple)):
                seg_pred = seg_logits[0].argmax(1)
            else:
                seg_pred = seg_logits.argmax(1)
            
            if return_classification and self.classifier is not None:
                try:
                    cls_logits = self._classification_forward()
                    cls_pred = cls_logits.argmax(1)
                    cls_probs = F.softmax(cls_logits, dim=1)
                    
                    return seg_pred, {
                        'prediction': cls_pred,
                        'probabilities': cls_probs,
                        'logits': cls_logits
                    }
                except Exception as e:
                    self.print_to_log_file(f"Warning: Classification prediction failed: {e}")
                    return seg_pred, None
            else:
                return seg_pred