import os
import csv
from typing import List, Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_learning_curve_summary(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v is None:
                    parsed[k] = v
                    continue

                vv = v.strip()
                try:
                    if "." in vv:
                        parsed[k] = float(vv)
                    else:
                        parsed[k] = int(vv)
                except ValueError:
                    parsed[k] = vv
            rows.append(parsed)

    rows = sorted(rows, key=lambda x: x["fraction"])
    return rows


def _get_x_values(rows: List[Dict[str, Any]], x_mode: str = "fraction"):
    if x_mode == "fraction":
        x = [row["fraction"] for row in rows]
        xlabel = "Training Data Fraction"
    elif x_mode == "num_samples":
        x = [row["num_samples"] for row in rows]
        xlabel = "Number of Training Samples"
    else:
        raise ValueError(f"Unsupported x_mode: {x_mode}")

    return x, xlabel


def plot_metric_with_errorbar(
    rows: List[Dict[str, Any]],
    mean_key: str,
    std_key: str,
    title: str,
    ylabel: str,
    save_path: str,
    x_mode: str = "fraction"
):
    x, xlabel = _get_x_values(rows, x_mode=x_mode)
    y = [row[mean_key] for row in rows]
    yerr = [row[std_key] for row in rows]

    plt.figure(figsize=(8, 6))
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_metric_line(
    rows: List[Dict[str, Any]],
    value_key: str,
    title: str,
    ylabel: str,
    save_path: str,
    x_mode: str = "fraction"
):
    x, xlabel = _get_x_values(rows, x_mode=x_mode)
    y = [row[value_key] for row in rows]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_learning_curve(summary_csv: str, save_dir: str = None, x_mode: str = "fraction"):
    rows = read_learning_curve_summary(summary_csv)

    if save_dir is None:
        save_dir = os.path.dirname(summary_csv)
    os.makedirs(save_dir, exist_ok=True)

    # Validation metrics
    plot_metric_with_errorbar(
        rows=rows,
        mean_key="val_dice_mean",
        std_key="val_dice_std",
        title="Validation Dice vs Training Data Size",
        ylabel="Validation Dice",
        save_path=os.path.join(save_dir, "val_dice_vs_fraction.png"),
        x_mode=x_mode
    )

    plot_metric_with_errorbar(
        rows=rows,
        mean_key="val_iou_mean",
        std_key="val_iou_std",
        title="Validation IoU vs Training Data Size",
        ylabel="Validation IoU",
        save_path=os.path.join(save_dir, "val_iou_vs_fraction.png"),
        x_mode=x_mode
    )

    # Optional test metrics
    if "test_dice_mean" in rows[0] and "test_dice_std" in rows[0]:
        plot_metric_with_errorbar(
            rows=rows,
            mean_key="test_dice_mean",
            std_key="test_dice_std",
            title="Test Dice vs Training Data Size",
            ylabel="Test Dice",
            save_path=os.path.join(save_dir, "test_dice_vs_fraction.png"),
            x_mode=x_mode
        )

    if "test_iou_mean" in rows[0] and "test_iou_std" in rows[0]:
        plot_metric_with_errorbar(
            rows=rows,
            mean_key="test_iou_mean",
            std_key="test_iou_std",
            title="Test IoU vs Training Data Size",
            ylabel="Test IoU",
            save_path=os.path.join(save_dir, "test_iou_vs_fraction.png"),
            x_mode=x_mode
        )

    if "test_normalized_surface_distance_mean" in rows[0] and "test_normalized_surface_distance_std" in rows[0]:
        plot_metric_with_errorbar(
            rows=rows,
            mean_key="test_normalized_surface_distance_mean",
            std_key="test_normalized_surface_distance_std",
            title="Test Normalized Surface Distance vs Training Data Size",
            ylabel="Test Normalized Surface Distance",
            save_path=os.path.join(save_dir, "test_normalized_surface_distance_vs_fraction.png"),
            x_mode=x_mode
        )

    if "test_normalized_surface_dice_mean" in rows[0] and "test_normalized_surface_dice_std" in rows[0]:
        plot_metric_with_errorbar(
            rows=rows,
            mean_key="test_normalized_surface_dice_mean",
            std_key="test_normalized_surface_dice_std",
            title="Test Normalized Surface Dice vs Training Data Size",
            ylabel="Test Normalized Surface Dice",
            save_path=os.path.join(save_dir, "test_normalized_surface_dice_vs_fraction.png"),
            x_mode=x_mode
        )

    # Training efficiency
    plot_metric_line(
        rows=rows,
        value_key="train_time_mean_sec",
        title="Training Time vs Training Data Size",
        ylabel="Mean Training Time (sec)",
        save_path=os.path.join(save_dir, "train_time_vs_fraction.png"),
        x_mode=x_mode
    )

    plot_metric_line(
        rows=rows,
        value_key="gpu_peak_memory_mean_mb",
        title="GPU Peak Memory vs Training Data Size",
        ylabel="Mean GPU Peak Memory (MB)",
        save_path=os.path.join(save_dir, "gpu_peak_memory_vs_fraction.png"),
        x_mode=x_mode
    )

    plot_metric_line(
        rows=rows,
        value_key="best_epoch_mean",
        title="Best Epoch vs Training Data Size",
        ylabel="Mean Best Epoch",
        save_path=os.path.join(save_dir, "best_epoch_vs_fraction.png"),
        x_mode=x_mode
    )

    print(f"Learning curve plots saved to: {save_dir}")
