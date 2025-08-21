"""
Simple inference (segmentation + classification) for your nnU-Net trainer.

- Safe static patch so init works on your nnU-Net build.
- Classification via encoder forward-hook during nnU-Net sliding-window,
  using ALL encoder features; per-case probs are averaged over tiles.

Outputs:
  - *.nii.gz in --output_dir
  - classification_logits.json
  - classification_results.csv
"""

import os
import json
import csv
import argparse
import numpy as np
import torch
import multiprocessing
from time import sleep
from collections import OrderedDict

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.helpers import empty_cache

# Import the ClassificationHead (try main file, else backup)
try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerWithClassification import ClassificationHead
except ImportError:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerWithClassification_backup import ClassificationHead


class PredictorWithClassification(nnUNetPredictor):
    def predict_from_data_iterator(
        self,
        data_iterator,
        save_probabilities: bool = False,
        num_processes_segmentation_export: int = default_num_processes
    ):
        """
        Each item from data_iterator: dict with keys 'data', 'ofile', 'data_properties'.
        Produces segmentation and per-case classification probs.
        """
        classification_outputs: OrderedDict[str, list] = OrderedDict()
        first_output_dir = None

        # Keep pools small (Windows-friendly)
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            pending = []

            for pre in data_iterator:
                data = pre['data']
                if isinstance(data, str):
                    tmp = data
                    arr = np.ascontiguousarray(np.load(tmp))
                    data = torch.from_numpy(arr)
                    os.remove(tmp)
                elif isinstance(data, np.ndarray):
                    data = torch.from_numpy(np.ascontiguousarray(data))

                # ---- IMPORTANT SHAPE HANDLING FOR YOUR NNUNET BUILD ----
                # This nnU-Net expects 4D [C, Z, Y, X] for 3D sliding window.
                if not data.is_floating_point():
                    data = data.float()
                if data.ndim == 5:  # [B, C, Z, Y, X] -> squeeze B if 1
                    if data.shape[0] != 1:
                        raise ValueError(f"Got 5D with batch {data.shape[0]}; expected batch=1. Shape: {tuple(data.shape)}")
                    data = data[0]
                elif data.ndim != 4:
                    raise ValueError(f"Expected 4D (C,Z,Y,X) or 5D (1,C,Z,Y,X); got {tuple(data.shape)}")
                # -------------------------------------------------------

                ofile = pre['ofile']
                props = pre['data_properties']

                case_id = (os.path.basename(ofile).replace('.nii.gz', '')
                           if ofile else f"case_{len(classification_outputs)}")

                if ofile is not None:
                    print(f"\nPredicting {os.path.basename(ofile)}")
                else:
                    print(f"\nPredicting array with shape {tuple(data.shape)}")

                # throttle background export queue
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, pending, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, pending, allowed_num_queued=2)

                # --- Collect classification votes via encoder forward-hook ---
                cls_votes = []

                def enc_hook(module, inputs, output):
                    feats = list(output) if isinstance(output, (list, tuple)) else [output]
                    with torch.no_grad():
                        logits = self.network.ClassificationHead(feats)  # [B, num_classes], uses ALL features
                        probs = torch.softmax(logits, dim=1)             # [B, num_classes]
                        cls_votes.append(probs.detach().cpu())

                hook_handle = self.network.encoder.register_forward_hook(enc_hook)

                # --- Segmentation (nnU-Net's own sliding-window path) ---
                with torch.no_grad():
                    prediction = self.predict_logits_from_preprocessed_data(data).cpu()

                # remove hook & aggregate classification
                hook_handle.remove()
                if len(cls_votes) == 0:
                    print(f"Warning: no classification votes captured for {case_id}.")
                    num_classes = getattr(self.network.ClassificationHead, 'num_classes', 3)
                    cls_prob = [0.0] * num_classes
                else:
                    cls_prob = torch.mean(torch.cat(cls_votes, dim=0), dim=0).squeeze().tolist()

                classification_outputs[case_id] = cls_prob

                # --- Queue export for segmentation ---
                if ofile is not None:
                    if first_output_dir is None:
                        first_output_dir = os.path.dirname(ofile)
                    pending.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, props, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    pending.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape,
                            ((prediction, self.plans_manager, self.configuration_manager,
                               self.label_manager, props, save_probabilities),)
                        )
                    )

            # Gather export results
            ret = [h.get()[0] for h in pending]

        # --- Save classification outputs (JSON + CSV) ---
        out_dir = first_output_dir if first_output_dir is not None else os.getcwd()
        maybe_mkdir_p(out_dir)

        json_path = join(out_dir, "classification_logits.json")
        with open(json_path, "w") as f:
            json.dump(classification_outputs, f, indent=2)

        csv_path = join(out_dir, "classification_results.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            if len(classification_outputs) == 0:
                w.writerow(["case", "pred_class"])  # nothing predicted
            else:
                first_probs = next(iter(classification_outputs.values()))
                header = ["case", "pred_class"] + [f"p{i}" for i in range(len(first_probs))]
                w.writerow(header)
                for k, probs in classification_outputs.items():
                    pred = int(np.argmax(probs))
                    w.writerow([k, pred] + [float(p) for p in probs])

        # tidy up
        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()
        compute_gaussian.cache_clear()
        empty_cache(self.device)

        print(f"\nSaved classification to:\n  {json_path}\n  {csv_path}")
        return ret


def ensure_head_and_reload(predictor, num_classes=3):
    """
    Ensure the net has ClassificationHead (consuming ALL encoder features).
    Do NOT manually reload the checkpoint here; the predictor already did.
    """
    if not hasattr(predictor.network, "ClassificationHead"):
        enc_ch = predictor.network.encoder.output_channels
        try:
            head = ClassificationHead(enc_ch, num_classes=num_classes, use_all_features=True)
        except TypeError:
            head = ClassificationHead(enc_ch, num_classes=num_classes)
        predictor.network.ClassificationHead = head.to(predictor.device)
        print("Attached ClassificationHead at inference (was missing).")
    predictor.network.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help="trainer folder (contains fold_0, plans.json, dataset.json)")
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default='checkpoint_best_combined.pth')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--device', type=str, default=None)  # e.g. 'cuda:0' or 'cpu'
    parser.add_argument('--tta', dest='tta', action='store_true')
    parser.add_argument('--no-tta', dest='tta', action='store_false')
    parser.set_defaults(tta=False)
    parser.add_argument('--preproc_workers', type=int, default=1)  # must be >= 1
    parser.add_argument('--export_workers', type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device is not None else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    predictor = PredictorWithClassification(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=args.tta,           # TTA for seg; cls votes come from same forwards
        perform_everything_on_device=False,
        device=device,
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    # --- Patch: static, signature-safe builder that attaches your head ---
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerWithClassification import nnUNetTrainerWithClassification
    except ImportError:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerWithClassification_backup import nnUNetTrainerWithClassification
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    def _safe_build_network_architecture(*args_, **kwargs_):
        net = nnUNetTrainer.build_network_architecture(*args_, **kwargs_)
        if not hasattr(net, "ClassificationHead"):
            enc_ch = net.encoder.output_channels
            try:
                head = ClassificationHead(enc_ch, num_classes=args.num_classes, use_all_features=True)
            except TypeError:
                head = ClassificationHead(enc_ch, num_classes=args.num_classes)
            net.ClassificationHead = head.to(device)
        return net

    nnUNetTrainerWithClassification.build_network_architecture = staticmethod(_safe_build_network_architecture)
    print("Patched nnUNetTrainerWithClassification.build_network_architecture (static, signature-safe).")
    # --------------------------------------------------------------------

    # Initialize (positional args for cross-version compatibility)
    predictor.initialize_from_trained_model_folder(
        args.model_dir,
        use_folds=(args.fold,),
        checkpoint_name=args.checkpoint
    )

    # Safety net: ensure head exists
    ensure_head_and_reload(predictor, num_classes=args.num_classes)

    # Run inference
    predictor.predict_from_files(
        list_of_lists_or_source_folder=args.input_dir,
        output_folder_or_list_of_truncated_output_files=args.output_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=max(1, args.preproc_workers),
        num_processes_segmentation_export=args.export_workers
    )

    print(f"\nDone. Results in: {args.output_dir}")
