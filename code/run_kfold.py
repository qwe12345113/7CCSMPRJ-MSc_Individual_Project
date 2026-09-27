import os
import io
import copy
import csv
import json
import time
import random
import platform
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold

from dataloader import (
    read_samples_from_txt,
    get_segmentation_dataloader,
    get_segmentation_dataloader_from_txt,
)
from prediction_result import (
    test_one_epoch_with_overlay,
    plot_test_metrics,
    plot_test_summary,
    select_ranked_cases,
    save_ranked_case_visualizations,
    save_selected_cases_csv,
    remove_padding_from_tensor,
    binary_segmentation_metrics_from_cropped,
)
from model import UNet
import gc


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_timestamped_save_root(base_name="kfold_results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"


def save_json(data, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_environment_info():
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
    }
    gpu_names = []
    for i in range(torch.cuda.device_count()):
        gpu_names.append(torch.cuda.get_device_name(i))
    info["gpu_names"] = gpu_names
    return info


def save_environment_info(save_path):
    info = get_environment_info()
    with open(save_path, "w", encoding="utf-8") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")


def save_experiment_notes(save_path, notes=""):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(notes.strip() + "\n")


def save_samples_list(samples, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for image_path, mask_path in samples:
            f.write(f"{image_path} {mask_path}\n")


def save_epoch_log_csv(history_rows, save_path):
    if not history_rows:
        return
    fieldnames = list(history_rows[0].keys())
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)


def get_model_summary_text(model):
    buffer = io.StringIO()
    buffer.write(str(model))
    buffer.write("\n")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffer.write(f"\nTotal parameters: {total_params}\n")
    buffer.write(f"Trainable parameters: {trainable_params}\n")
    return buffer.getvalue(), total_params, trainable_params


def get_effective_pixel_count(mask_tensor):
    if mask_tensor.ndim == 3:
        return int(mask_tensor.shape[-2] * mask_tensor.shape[-1])
    if mask_tensor.ndim == 2:
        return int(mask_tensor.shape[0] * mask_tensor.shape[1])
    raise ValueError(f"Unsupported mask shape: {mask_tensor.shape}")


def get_normalize_transform(normalize_mode="none"):
    if normalize_mode is None or normalize_mode == "none":
        return None
    if normalize_mode == "fixed_05":
        return T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")


def compute_batch_loss(
        outputs,
        masks,
        paddings,
        criterion,
        crop_padding_for_loss=False,
        loss_reduction_mode="sample_mean"
):
    if loss_reduction_mode not in ["sample_mean", "pixel_weighted"]:
        raise ValueError(f"Unsupported loss_reduction_mode: {loss_reduction_mode}")

    if not crop_padding_for_loss:
        loss = criterion(outputs, masks)
        pixel_count = int(masks.shape[-2] * masks.shape[-1])
        stats = {
            "effective_pixels_mean": float(pixel_count),
            "effective_pixels_min": int(pixel_count),
            "effective_pixels_max": int(pixel_count),
        }
        return loss, stats

    per_sample_losses = []
    per_sample_pixels = []

    batch_size = outputs.size(0)
    for i in range(batch_size):
        padding = paddings[i]
        logit_crop = remove_padding_from_tensor(outputs[i], padding)
        mask_crop = remove_padding_from_tensor(masks[i], padding)

        loss_i = criterion(
            logit_crop.unsqueeze(0),
            mask_crop.unsqueeze(0)
        )
        pixels_i = get_effective_pixel_count(mask_crop)

        per_sample_losses.append(loss_i)
        per_sample_pixels.append(pixels_i)

    if loss_reduction_mode == "sample_mean":
        loss = torch.stack(per_sample_losses).mean()
    else:
        weights = torch.tensor(
            per_sample_pixels,
            dtype=per_sample_losses[0].dtype,
            device=per_sample_losses[0].device
        )
        losses = torch.stack(per_sample_losses)
        loss = (losses * weights).sum() / weights.sum()

    stats = {
        "effective_pixels_mean": float(np.mean(per_sample_pixels)),
        "effective_pixels_min": int(np.min(per_sample_pixels)),
        "effective_pixels_max": int(np.max(per_sample_pixels)),
    }
    return loss, stats


def get_monitor_mode(monitor_name):
    if monitor_name in ["train_loss", "val_loss"]:
        return "min"
    if monitor_name == "val_dice":
        return "max"
    raise ValueError(f"Unsupported monitor_name: {monitor_name}")


def get_monitor_value(monitor_name, train_loss, val_loss, val_dice):
    if monitor_name == "train_loss":
        return train_loss
    if monitor_name == "val_loss":
        return val_loss
    if monitor_name == "val_dice":
        return val_dice
    raise ValueError(f"Unsupported monitor_name: {monitor_name}")


def train_one_epoch(
        model,
        dataloader,
        optimizer,
        criterion,
        device,
        use_amp=True,
        crop_padding_for_train_loss=False,
        loss_reduction_mode="sample_mean"
):
    model.train()
    running_loss = 0.0
    batch_pixel_means = []
    batch_pixel_mins = []
    batch_pixel_maxs = []

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        paddings = batch["padding"]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            outputs = model(images)
            loss, loss_stats = compute_batch_loss(
                outputs=outputs,
                masks=masks,
                paddings=paddings,
                criterion=criterion,
                crop_padding_for_loss=crop_padding_for_train_loss,
                loss_reduction_mode=loss_reduction_mode
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        batch_pixel_means.append(loss_stats["effective_pixels_mean"])
        batch_pixel_mins.append(loss_stats["effective_pixels_min"])
        batch_pixel_maxs.append(loss_stats["effective_pixels_max"])

    epoch_stats = {
        "effective_pixels_mean": float(np.mean(batch_pixel_means)) if batch_pixel_means else 0.0,
        "effective_pixels_min": int(np.min(batch_pixel_mins)) if batch_pixel_mins else 0,
        "effective_pixels_max": int(np.max(batch_pixel_maxs)) if batch_pixel_maxs else 0,
    }

    return running_loss / len(dataloader), epoch_stats


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, num_classes=1, use_amp=True):
    model.eval()

    total_loss = 0.0
    sample_count = 0
    dice_scores = []
    iou_scores = []

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        paddings = batch["padding"]

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            outputs = model(images)

        probs = torch.sigmoid(outputs) if num_classes == 1 else None
        preds = (probs > 0.5).float() if num_classes == 1 else None

        batch_size = images.size(0)
        for i in range(batch_size):
            padding = paddings[i]

            if num_classes == 1:
                logit_crop = remove_padding_from_tensor(outputs[i], padding)
                mask_crop = remove_padding_from_tensor(masks[i], padding)
                pred_crop = remove_padding_from_tensor(preds[i], padding)

                loss_i = criterion(
                    logit_crop.unsqueeze(0),
                    mask_crop.unsqueeze(0)
                )
                total_loss += loss_i.item()

                metrics = binary_segmentation_metrics_from_cropped(pred_crop, mask_crop)
                dice_scores.append(metrics["dice"])
                iou_scores.append(metrics["iou"])
            else:
                logit_crop = remove_padding_from_tensor(outputs[i], padding)
                mask_crop = remove_padding_from_tensor(masks[i], padding)

                loss_i = criterion(
                    logit_crop.unsqueeze(0),
                    mask_crop.unsqueeze(0)
                )
                total_loss += loss_i.item()

            sample_count += 1

    avg_loss = total_loss / max(sample_count, 1)
    avg_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    avg_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    return avg_loss, avg_dice, avg_iou


class EarlyStopping:
    def __init__(self, mode="max", patience=10, min_delta=1e-4):
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.num_bad_epochs = 0

    def step(self, current):
        if self.best is None:
            self.best = current
            return False

        if self.mode == "max":
            improved = current > self.best + self.min_delta
        else:
            improved = current < self.best - self.min_delta

        if improved:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs >= self.patience


class SoftDiceLoss(nn.Module):
    """
    Differentiable Dice loss for binary segmentation.

    Inputs:
        logits : [B, 1, H, W]
        targets: [B, 1, H, W], values in {0, 1}
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        dims = (1, 2, 3)
        intersection = torch.sum(probs * targets, dim=dims)
        denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class SoftIoULoss(nn.Module):
    """
    Differentiable IoU/Jaccard loss for binary segmentation.

    Inputs:
        logits : [B, 1, H, W]
        targets: [B, 1, H, W], values in {0, 1}
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        dims = (1, 2, 3)
        intersection = torch.sum(probs * targets, dim=dims)
        total = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
        union = total - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou.mean()


class TverskyLoss(nn.Module):
    """
    Differentiable Tversky loss for binary segmentation.

    alpha controls false positive penalty.
    beta controls false negative penalty.

    Common settings:
        alpha=0.3, beta=0.7 -> penalize false negatives more, improve recall
        alpha=0.7, beta=0.3 -> penalize false positives more, improve precision
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        dims = (1, 2, 3)
        tp = torch.sum(probs * targets, dim=dims)
        fp = torch.sum(probs * (1.0 - targets), dim=dims)
        fn = torch.sum((1.0 - probs) * targets, dim=dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky.mean()


class MultiLoss(nn.Module):
    """
    Weighted multi-loss for binary segmentation.

    Supported components:
        BCEWithLogitsLoss, SoftDiceLoss, SoftIoULoss, TverskyLoss

    Example:
        0.5 * BCE + 0.5 * Dice
        0.4 * BCE + 0.4 * Dice + 0.2 * IoU
    """
    def __init__(
        self,
        binary_pos_weight=None,
        bce_weight=0.5,
        dice_weight=0.5,
        iou_weight=0.0,
        tversky_weight=0.0,
        tversky_alpha=0.5,
        tversky_beta=0.5,
        smooth=1.0,
        device="cpu"
    ):
        super().__init__()

        if binary_pos_weight is not None:
            pos_weight = torch.tensor(
                [binary_pos_weight],
                dtype=torch.float32,
                device=device
            )
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()

        self.dice = SoftDiceLoss(smooth=smooth)
        self.iou = SoftIoULoss(smooth=smooth)
        self.tversky = TverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            smooth=smooth
        )

        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.iou_weight = float(iou_weight)
        self.tversky_weight = float(tversky_weight)

        total_weight = (
            self.bce_weight
            + self.dice_weight
            + self.iou_weight
            + self.tversky_weight
        )
        if total_weight <= 0:
            raise ValueError("At least one loss weight must be > 0.")

    def forward(self, logits, targets):
        targets = targets.float()
        loss = logits.new_tensor(0.0)

        if self.bce_weight > 0:
            loss = loss + self.bce_weight * self.bce(logits, targets)

        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, targets)

        if self.iou_weight > 0:
            loss = loss + self.iou_weight * self.iou(logits, targets)

        if self.tversky_weight > 0:
            loss = loss + self.tversky_weight * self.tversky(logits, targets)

        return loss


def build_optimizer(model, optimizer_name="Adam", learning_rate=1e-4, weight_decay=0.0):
    """
    Build optimizer from config.

    optimizer_name:
        "Adam"  -> torch.optim.Adam
        "AdamW" -> torch.optim.AdamW with decoupled weight decay
    """
    optimizer_name = str(optimizer_name).lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")


def build_loss_function(
    num_classes=1,
    binary_pos_weight=None,
    multiclass_weights=None,
    device="cpu",
    loss_type="bce",
    bce_weight=1.0,
    dice_weight=0.0,
    iou_weight=0.0,
    tversky_weight=0.0,
    tversky_alpha=0.5,
    tversky_beta=0.5,
    loss_smooth=1.0,
):
    """
    Build loss function.

    Binary segmentation loss_type options:
        "bce"
        "bce_dice"
        "bce_dice_iou"
        "bce_tversky"
        "multi"

    Notes:
        - Binary logits should be raw logits, not sigmoid outputs.
        - targets should be [B, 1, H, W] and values should be 0/1.
    """
    if num_classes == 1:
        loss_type = str(loss_type).lower()

        if loss_type == "bce":
            if binary_pos_weight is not None:
                pos_weight = torch.tensor(
                    [binary_pos_weight],
                    dtype=torch.float32,
                    device=device
                )
                return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            return nn.BCEWithLogitsLoss()

        if loss_type == "bce_dice":
            # Safe default if config forgot to set dice_weight.
            if dice_weight <= 0 and iou_weight <= 0 and tversky_weight <= 0:
                bce_weight = 0.5
                dice_weight = 0.5

        elif loss_type == "bce_dice_iou":
            # Safe default if config forgot to set dice/iou weights.
            if dice_weight <= 0 and iou_weight <= 0 and tversky_weight <= 0:
                bce_weight = 0.4
                dice_weight = 0.4
                iou_weight = 0.2

        elif loss_type == "bce_tversky":
            # Safe default if config forgot to set tversky_weight.
            if tversky_weight <= 0 and dice_weight <= 0 and iou_weight <= 0:
                bce_weight = 0.5
                tversky_weight = 0.5

        elif loss_type == "multi":
            pass

        else:
            raise ValueError(f"Unsupported binary loss_type: {loss_type}")

        return MultiLoss(
            binary_pos_weight=binary_pos_weight,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            iou_weight=iou_weight,
            tversky_weight=tversky_weight,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            smooth=loss_smooth,
            device=device,
        )

    if multiclass_weights is not None:
        class_weights = torch.tensor(multiclass_weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=class_weights)

    return nn.CrossEntropyLoss()


def plot_training_history(history, save_dir="results", prefix="fold"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["val_dice"], label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Validation Dice")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_dice_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["val_iou"], label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Validation IoU")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_iou_curve.png"))
    plt.close()


def save_kfold_summary_csv(results, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold",
            "best_epoch",
            "best_val_dice",
            "best_val_iou",
            "best_val_loss",
            "train_time_sec",
            "train_time_min",
            "early_stopped",
            "stop_epoch",
            "gpu_peak_memory_bytes",
            "gpu_peak_memory_mb",
        ])
        for row in results:
            writer.writerow([
                row["fold"],
                row["best_epoch"],
                row["best_val_dice"],
                row["best_val_iou"],
                row["best_val_loss"],
                row["train_time_sec"],
                row["train_time_min"],
                row["early_stopped"],
                row["stop_epoch"],
                row["gpu_peak_memory_bytes"],
                row["gpu_peak_memory_mb"],
            ])


def save_all_folds_test_results_csv(all_fold_test_results, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold",
            "test_loss",
            "dice",
            "iou",
            "precision",
            "recall",
            "accuracy",
            "normalized_surface_distance",
            "normalized_surface_dice",
        ])
        for row in all_fold_test_results:
            writer.writerow([
                row["fold"],
                row["test_loss"],
                row["dice"],
                row["iou"],
                row["precision"],
                row["recall"],
                row["accuracy"],
                row["normalized_surface_distance"],
                row["normalized_surface_dice"]
            ])


def save_all_folds_test_summary_csv(all_fold_test_results, save_path):
    metrics = [
        "test_loss",
        "dice",
        "iou",
        "precision",
        "recall",
        "accuracy",
        "normalized_surface_distance",
        "normalized_surface_dice",
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])
        for metric in metrics:
            values = [row[metric] for row in all_fold_test_results]
            writer.writerow([metric, float(np.mean(values)), float(np.std(values))])


def plot_all_folds_test_metrics(all_fold_test_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    metrics = ["dice", "iou", "precision", "recall", "accuracy"]
    means = [np.mean([row[m] for row in all_fold_test_results]) for m in metrics]
    stds = [np.std([row[m] for row in all_fold_test_results]) for m in metrics]

    plt.figure(figsize=(9, 6))
    plt.bar(metrics, means, yerr=stds, capsize=5)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("All Folds Test Metrics (Mean ± Std)")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_folds_test_metrics.png"), dpi=150)
    plt.close()


def run_single_fold(
        fold_id,
        train_samples,
        val_samples,
        device,
        save_root,
        batch_size=8,
        target_size=(672, 928),
        num_classes=1,
        learning_rate=1e-4,
        optimizer_name="Adam",
        weight_decay=0.0,
        loss_type="bce",
        bce_weight=1.0,
        dice_weight=0.0,
        iou_weight=0.0,
        tversky_weight=0.0,
        tversky_alpha=0.5,
        tversky_beta=0.5,
        loss_smooth=1.0,
        num_workers=8,
        max_epochs=200,
        patience=15,
        min_delta=1e-4,
        binary_pos_weight=None,
        multiclass_weights=None,
        use_amp=True,
        crop_padding_for_train_loss=False,
        loss_reduction_mode="sample_mean",
        early_stop_monitor="val_dice",
        scheduler_monitor="val_dice",
        augment_train=False,
        normalize_mode="none",
        aug_prob=0.5,
        hflip_prob=0.5,
        vflip_prob=0.0,
        rotation_degree=10,
        rotation_prob=0.5,
        use_color_jitter=True,
        color_jitter_prob=0.5,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.02
):
    fold_dir = os.path.join(save_root, f"fold_{fold_id}")
    os.makedirs(fold_dir, exist_ok=True)

    save_samples_list(train_samples, os.path.join(fold_dir, "train_samples.txt"))
    save_samples_list(val_samples, os.path.join(fold_dir, "val_samples.txt"))

    normalize = get_normalize_transform(normalize_mode)

    train_loader = get_segmentation_dataloader(
        samples=train_samples,
        batch_size=batch_size,
        target_size=target_size,
        num_classes=num_classes,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        augment=augment_train,
        normalize=normalize,
        aug_prob=aug_prob,
        hflip_prob=hflip_prob,
        vflip_prob=vflip_prob,
        rotation_degree=rotation_degree,
        rotation_prob=rotation_prob,
        use_color_jitter=use_color_jitter,
        color_jitter_prob=color_jitter_prob,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue
    )

    val_loader = get_segmentation_dataloader(
        samples=val_samples,
        batch_size=batch_size,
        target_size=target_size,
        num_classes=num_classes,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        augment=False,
        normalize=normalize
    )

    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    criterion = build_loss_function(
        num_classes=num_classes,
        binary_pos_weight=binary_pos_weight,
        multiclass_weights=multiclass_weights,
        device=device,
        loss_type=loss_type,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        iou_weight=iou_weight,
        tversky_weight=tversky_weight,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        loss_smooth=loss_smooth,
    )

    optimizer = build_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=get_monitor_mode(scheduler_monitor),
        factor=0.5,
        patience=5,
        verbose=True
    )

    early_stopper = EarlyStopping(
        mode=get_monitor_mode(early_stop_monitor),
        patience=patience,
        min_delta=min_delta
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": []
    }
    epoch_log_rows = []

    best_epoch = 0
    best_val_dice = -1.0
    best_val_iou = 0.0
    best_val_loss = float("inf")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_start_time = time.time()
    early_stopped = False
    stop_epoch = 0

    for epoch in range(1, max_epochs + 1):
        epoch_start_time = time.time()

        train_loss, train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            crop_padding_for_train_loss=crop_padding_for_train_loss,
            loss_reduction_mode=loss_reduction_mode
        )

        val_loss, val_dice, val_iou = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            use_amp=use_amp
        )

        monitor_value_for_scheduler = get_monitor_value(
            scheduler_monitor, train_loss, val_loss, val_dice
        )
        monitor_value_for_early_stop = get_monitor_value(
            early_stop_monitor, train_loss, val_loss, val_dice
        )

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time_sec = time.time() - epoch_start_time

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        epoch_log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "learning_rate": current_lr,
            "epoch_time_sec": epoch_time_sec,
            "effective_pixels_mean": train_stats["effective_pixels_mean"],
            "effective_pixels_min": train_stats["effective_pixels_min"],
            "effective_pixels_max": train_stats["effective_pixels_max"],
            "scheduler_monitor_value": monitor_value_for_scheduler,
            "early_stop_monitor_value": monitor_value_for_early_stop,
        })

        print(
            f"[Fold {fold_id}] Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"LR: {current_lr:.6g} | "
            f"Epoch Time: {epoch_time_sec:.2f}s | "
            f"Pixels(mean/min/max): "
            f"{train_stats['effective_pixels_mean']:.1f}/"
            f"{train_stats['effective_pixels_min']}/"
            f"{train_stats['effective_pixels_max']} | "
            f"ES monitor({early_stop_monitor})={monitor_value_for_early_stop:.6f}"
        )

        scheduler.step(monitor_value_for_scheduler)

        if val_dice > best_val_dice:
            best_epoch = epoch
            best_val_dice = val_dice
            best_val_iou = val_iou
            best_val_loss = val_loss
            torch.save(copy.deepcopy(model.state_dict()), os.path.join(fold_dir, "best_model.pth"))

        if early_stopper.step(monitor_value_for_early_stop):
            early_stopped = True
            stop_epoch = epoch
            print(f"[Fold {fold_id}] Early stopping at epoch {epoch}")
            break

    if not early_stopped:
        stop_epoch = len(epoch_log_rows)

    train_end_time = time.time()
    train_time_sec = train_end_time - train_start_time
    train_time_min = train_time_sec / 60.0

    if device.type == "cuda":
        gpu_peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        gpu_peak_memory_mb = gpu_peak_memory_bytes / (1024 ** 2)
    else:
        gpu_peak_memory_bytes = 0
        gpu_peak_memory_mb = 0.0

    print(f"[Fold {fold_id}] Training time: {train_time_sec:.2f} sec ({train_time_min:.2f} min)")
    print(f"[Fold {fold_id}] GPU peak memory: {gpu_peak_memory_mb:.2f} MB")

    torch.save(model.state_dict(), os.path.join(fold_dir, "last_model.pth"))
    save_epoch_log_csv(epoch_log_rows, os.path.join(fold_dir, "epoch_log.csv"))
    plot_training_history(history, save_dir=fold_dir, prefix=f"fold_{fold_id}")

    return {
        "fold": fold_id,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "best_val_iou": best_val_iou,
        "best_val_loss": best_val_loss,
        "best_model_path": os.path.join(fold_dir, "best_model.pth"),
        "train_time_sec": train_time_sec,
        "train_time_min": train_time_min,
        "early_stopped": early_stopped,
        "stop_epoch": stop_epoch,
        "gpu_peak_memory_bytes": gpu_peak_memory_bytes,
        "gpu_peak_memory_mb": gpu_peak_memory_mb,
    }


def run_kfold_training(
        all_trainval_txt,
        test_txt=None,
        n_splits=5,
        seed=42,
        batch_size=8,
        target_size=(672, 928),
        num_classes=1,
        learning_rate=1e-4,
        optimizer_name="Adam",
        weight_decay=0.0,
        loss_type="bce",
        bce_weight=1.0,
        dice_weight=0.0,
        iou_weight=0.0,
        tversky_weight=0.0,
        tversky_alpha=0.5,
        tversky_beta=0.5,
        loss_smooth=1.0,
        num_workers=8,
        max_epochs=200,
        patience=15,
        min_delta=1e-4,
        binary_pos_weight=None,
        multiclass_weights=None,
        save_root="kfold_results",
        use_amp=True,
        n_case_samples=5,
        ranking_metric="dice",
        crop_padding_for_train_loss=False,
        loss_reduction_mode="sample_mean",
        early_stop_monitor="val_dice",
        scheduler_monitor="val_dice",
        augment_train=False,
        normalize_mode="none",
        aug_prob=0.5,
        hflip_prob=0.5,
        vflip_prob=0.0,
        rotation_degree=10,
        rotation_prob=0.5,
        use_color_jitter=True,
        color_jitter_prob=0.5,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.02
):
    os.makedirs(save_root, exist_ok=True)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"K-fold training is enabled | n_splits = {n_splits}")
    print(f"crop_padding_for_train_loss = {crop_padding_for_train_loss}")
    print(f"loss_reduction_mode = {loss_reduction_mode}")
    print(f"early_stop_monitor = {early_stop_monitor}")
    print(f"scheduler_monitor = {scheduler_monitor}")
    print(f"optimizer_name = {optimizer_name}")
    print(f"weight_decay = {weight_decay}")
    print(f"loss_type = {loss_type}")
    print(
        "loss_weights = "
        f"bce:{bce_weight}, dice:{dice_weight}, iou:{iou_weight}, "
        f"tversky:{tversky_weight}"
    )

    all_samples = read_samples_from_txt(all_trainval_txt)
    print(f"Total train+val samples: {len(all_samples)}")

    normalize = get_normalize_transform(normalize_mode)

    test_loader = None
    if test_txt is not None:
        test_loader = get_segmentation_dataloader_from_txt(
            txt_file=test_txt,
            batch_size=batch_size,
            target_size=target_size,
            num_classes=num_classes,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            augment=False,
            normalize=normalize
        )
    fold_results = []
    all_fold_test_results = []

    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2 for K-fold training, got {n_splits}")
    if n_splits > len(all_samples):
        raise ValueError(
            f"n_splits must be <= number of train+val samples. "
            f"got n_splits={n_splits}, samples={len(all_samples)}"
        )

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    split_list = list(splitter.split(all_samples))
    total_runs = n_splits
    print(f"Using K-Fold cross validation | n_splits = {n_splits}")

    for fold_id, (train_idx, val_idx) in enumerate(split_list, start=1):
        train_samples = [all_samples[i] for i in train_idx]
        val_samples = [all_samples[i] for i in val_idx]

        print(f"\n===== Fold {fold_id}/{total_runs} =====")
        print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

        result = run_single_fold(
            fold_id=fold_id,
            train_samples=train_samples,
            val_samples=val_samples,
            device=device,
            save_root=save_root,
            batch_size=batch_size,
            target_size=target_size,
            num_classes=num_classes,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            loss_type=loss_type,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            iou_weight=iou_weight,
            tversky_weight=tversky_weight,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            loss_smooth=loss_smooth,
            num_workers=num_workers,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            binary_pos_weight=binary_pos_weight,
            multiclass_weights=multiclass_weights,
            use_amp=use_amp,
            crop_padding_for_train_loss=crop_padding_for_train_loss,
            loss_reduction_mode=loss_reduction_mode,
            early_stop_monitor=early_stop_monitor,
            scheduler_monitor=scheduler_monitor,
            augment_train=augment_train,
            normalize_mode=normalize_mode,
            aug_prob=aug_prob,
            hflip_prob=hflip_prob,
            vflip_prob=vflip_prob,
            rotation_degree=rotation_degree,
            rotation_prob=rotation_prob,
            use_color_jitter=use_color_jitter,
            color_jitter_prob=color_jitter_prob,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        fold_results.append(result)

        if test_loader is not None:
            fold_dir = os.path.join(save_root, f"fold_{fold_id}")
            test_save_dir = os.path.join(fold_dir, "test_results")
            os.makedirs(test_save_dir, exist_ok=True)

            model = UNet(in_channels=3, num_classes=num_classes).to(device)
            model.load_state_dict(torch.load(result["best_model_path"], map_location=device))

            criterion = build_loss_function(
                num_classes=num_classes,
                binary_pos_weight=binary_pos_weight,
                multiclass_weights=multiclass_weights,
                device=device,
                loss_type=loss_type,
                bce_weight=bce_weight,
                dice_weight=dice_weight,
                iou_weight=iou_weight,
                tversky_weight=tversky_weight,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
                loss_smooth=loss_smooth,
            )

            fold_test_result, all_sample_rows = test_one_epoch_with_overlay(
                model=model,
                dataloader=test_loader,
                criterion=criterion,
                device=device,
                save_dir=test_save_dir,
                save_overlay_samples=0,
                use_amp=use_amp,
                normalize_mode=normalize_mode
            )

            fold_test_result["fold"] = fold_id
            all_fold_test_results.append(fold_test_result)

            print(
                f"[Fold {fold_id}] Test Dice: {fold_test_result['dice']:.4f} | "
                f"Test IoU: {fold_test_result['iou']:.4f}"
            )

            plot_test_metrics(fold_test_result, save_dir=test_save_dir)
            plot_test_summary(fold_test_result, save_dir=test_save_dir)

            case_groups = select_ranked_cases(
                all_sample_rows,
                metric=ranking_metric,
                n_per_group=n_case_samples
            )

            save_selected_cases_csv(
                case_groups,
                os.path.join(test_save_dir, "selected_cases.csv")
            )

            save_ranked_case_visualizations(
                model=model,
                dataset=test_loader.dataset,
                case_groups=case_groups,
                device=device,
                save_dir=os.path.join(test_save_dir, "ranked_cases"),
                use_amp=use_amp,
                normalize_mode=normalize_mode
            )
            # ---- fold cleanup ----
            try:
                del model
            except:
                pass

            try:
                del criterion
            except:
                pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_kfold_summary_csv(fold_results, os.path.join(save_root, "kfold_summary.csv"))

    mean_dice = np.mean([r["best_val_dice"] for r in fold_results])
    mean_iou = np.mean([r["best_val_iou"] for r in fold_results])
    mean_train_time_sec = np.mean([r["train_time_sec"] for r in fold_results])
    mean_train_time_min = np.mean([r["train_time_min"] for r in fold_results])
    mean_gpu_peak_memory_mb = np.mean([r["gpu_peak_memory_mb"] for r in fold_results])
    print(f"\nK-Fold mean Val Dice: {mean_dice:.4f}")
    print(f"K-Fold mean Val IoU : {mean_iou:.4f}")
    print(f"K-Fold mean training time: {mean_train_time_sec:.2f} sec ({mean_train_time_min:.2f} min)")
    print(f"K-Fold mean GPU peak memory: {mean_gpu_peak_memory_mb:.2f} MB")

    if all_fold_test_results:
        save_all_folds_test_results_csv(
            all_fold_test_results,
            os.path.join(save_root, "all_folds_test_results.csv")
        )

        save_all_folds_test_summary_csv(
            all_fold_test_results,
            os.path.join(save_root, "all_folds_test_summary.csv")
        )

        plot_all_folds_test_metrics(
            all_fold_test_results,
            save_dir=save_root
        )

        print("\n===== All Folds Test Summary =====")
        for metric in ["test_loss", "dice", "iou", "precision", "recall", "accuracy", "normalized_surface_distance", "normalized_surface_dice"]:
            values = [row[metric] for row in all_fold_test_results]
            print(f"{metric}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")