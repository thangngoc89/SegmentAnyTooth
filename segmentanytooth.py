# SPDX-License-Identifier: MIT
# ============================================================================
# SegmentAnyTooth
#
# Copyright (c) 2025 Khoa D. Nguyen
#
# This file is part of SegmentAnyTooth and is licensed under the MIT License.
# See LICENSE file in the repository root for full license information.
#
# Note: Pretrained model weights provided separately are under a Non-Commercial License.
# Refer to the WEIGHTS_LICENSE.txt for terms and conditions regarding model usage.
# ============================================================================
import os
from typing import Literal, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import LOGGER

from sam import sam_load, sam_predict
from utils import suppress_stdout

# Set Ultralytics logger to error-only
LOGGER.setLevel("ERROR")


# Define class names for left lateral view (flipped horizontally)
LEFT_CLASSES = [
    "le28", "le27", "le26", "le25", "le24", "le23", "le22", "le21",
    "le38", "le37", "le36", "le35", "le34", "le33", "le32", "le31",
    "le11", "le12", "le13", "le14", "le41", "le42", "le43", "le44",
]


def predict(
    image_path: str,
    view: Literal["upper", "lower", "left", "right", "front"],
    weight_dir: Optional[str] = "./weight",
    sam_batch_size: Optional[int] = 10,
) -> np.ndarray:
    """Predicts a semantic segmentation mask for teeth in the given image.

    Args:
        image_path (str): Path to the input image.
        view (str): View type ("upper", "lower", "left", "right", "front").
        weight_dir (str, optional): Directory containing model weights.
        sam_batch_size (int, optional): Batch size for SAM prediction.

    Returns:
        np.ndarray: Segmentation mask with FDI tooth labels.
    """
    weight_dir = os.path.normpath(weight_dir)
    should_flip = view == "left"
    image = cv2.imread(image_path)

    if should_flip:
        image = cv2.flip(image, 1)

    # Load models and run detection while suppressing noisy outputs
    with suppress_stdout():
        sam = sam_load(get_model_path("sam", weight_dir))
        yolo = YOLO(model=get_model_path(view, weight_dir))
        r = yolo.predict(
            image,
            save=False,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            project=None,
        )[0]

    # Early exit if no detections
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    # Get YOLO output
    names = r.names if not should_flip else LEFT_CLASSES
    boxes = r.boxes.xyxy.squeeze(0).cpu().numpy()
    clss = r.boxes.cls.squeeze(0).cpu().numpy().astype(np.int32)

    # Sort by class id to ensure consistent label ordering
    sort_ids = np.argsort(clss)
    clss = clss[sort_ids]
    boxes = boxes[sort_ids]

    if should_flip:
        # Unflip image and adjust box coordinates
        image_width = image.shape[1]
        image = cv2.flip(image, 1)

        flipped_boxes = boxes.copy()
        flipped_boxes[:, [0, 2]] = image_width - flipped_boxes[:, [2, 0]]
        boxes = flipped_boxes

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Predict masks using SAM
    sam_masks = sam_predict(
        sam=sam,
        boxes_xyxy=boxes,
        image=image,
        batch_size=sam_batch_size,
    )

    # Build the segmentation mask
    predict_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for cls_id, current_mask in zip(clss, sam_masks):
        fdi_tooth_name = int(names[cls_id][-2:])
        predict_mask[current_mask == 1] = fdi_tooth_name

    return predict_mask


def get_model_path(
    model: Literal["upper", "lower", "left", "right", "front", "sam"],
    weight_dir: Optional[str] = "./weight",
) -> str:
    """Returns the file path to the model weights."""
    if model == "left":
        model = "right"

    if model == "sam":
        name = "vit_tiny.pt"
    else:
        name = f"yolo11_{model}.pt"

    return os.path.join(weight_dir, f"segmentanytooth_{name}")


if __name__ == "__main__":
    mask = predict(
        image_path="examples/upper.jpg",
        view="upper",
        weight_dir="./weight",
    )
    cv2.imwrite("predicted_mask.png")