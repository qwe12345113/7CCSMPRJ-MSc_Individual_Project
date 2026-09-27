import os
import random
from typing import List, Tuple

from dataloader import read_samples_from_txt


Sample = Tuple[str, str]


def load_trainval_samples(trainval_txt: str) -> List[Sample]:
    """
    Load all samples from trainval txt.

    Returns:
        List of (image_path, mask_path)
    """
    return read_samples_from_txt(trainval_txt)

def save_subset_txt(samples: List[Sample], save_path: str) -> None:
    """
    Save a subset sample list to txt in:
        image_path mask_path
    format.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for image_path, mask_path in samples:
            f.write(f"{image_path} {mask_path}\n")


def fraction_to_tag(fraction: float) -> str:
    """
    Convert fraction 0.25 -> '25'
    """
    return str(int(round(fraction * 100)))
