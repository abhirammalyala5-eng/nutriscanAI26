# -*- coding: utf-8 -*-
"""
vision/segmentation.py
======================
DeepLabV3+ semantic segmentation inference with overlapping native-resolution tiles
and OpenCV Hough circle plate rim scale calibration.
"""

import json
import os
from pathlib import Path
from typing import Any, Tuple, List, Dict

import cv2
import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp


def load_segmenter(model_path: Path, labels_path: Path) -> Tuple[torch.nn.Module, Dict[int, str], torch.device]:
    """Load PyTorch DeepLabV3+ MobileNetV2 model and 104-class label dictionary."""
    if not model_path.is_file():
        raise RuntimeError(f"FoodSeg103 weights not found: {model_path}")
    if not labels_path.is_file():
        raise RuntimeError(f"FoodSeg103 class labels not found: {labels_path}")

    with labels_path.open(encoding="utf-8") as f:
        _label_config = json.load(f)
    class_names = {int(key): value for key, value in _label_config["id2label"].items()}
    if len(class_names) != 104:
        raise RuntimeError(f"Expected 104 FoodSeg103 classes, found {len(class_names)}")

    segmenter = smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=104
    )
    segmenter.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmenter.to(device).eval()
    return segmenter, class_names, device


def segmentation_summary_letterbox(
    image: Image.Image,
    segmenter: torch.nn.Module,
    device: torch.device,
    class_names: Dict[int, str],
    target_size: int = 512,
    confidence_thresh: float = 0.25,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Strategy 3 (Production): Aspect-Ratio-Preserving Letterbox Padding to 512x512.
    
    1. Resizes input image so its longest dimension is 512 pixels, preserving aspect ratio.
    2. Places the scaled image onto a 512x512 canvas with neutral padding.
    3. Executes a single forward pass on DeepLabV3+ MobileNetV2.
    4. Removes padding from the prediction mask and confidence map.
    5. Resamples unpadded masks back to original image dimensions (W, H).
    """
    orig_w, orig_h = image.size

    # 1. Scale preserving aspect ratio so max dimension is target_size (512)
    scale = target_size / max(orig_w, orig_h)
    new_w = max(1, min(target_size, int(round(orig_w * scale))))
    new_h = max(1, min(target_size, int(round(orig_h * scale))))

    resized_img = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # 2. Place on 512x512 canvas with neutral padding
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas.paste(resized_img, (pad_x, pad_y))

    # 3. ImageNet normalization
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    normalized = (pixels - mean) / std
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # 4. DeepLabV3+ inference
    with torch.inference_mode():
        logits = segmenter(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, prediction = probabilities.max(dim=0)

    pred_canvas = prediction.cpu().numpy().astype(np.uint8)
    conf_canvas = confidence.cpu().numpy().astype(np.float32)

    # 5. Crop out padding: isolate active food photo region
    pred_unpad = pred_canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    conf_unpad = conf_canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w]

    # 6. Map unpadded mask and confidence map back to original image dimensions
    pred_orig = np.asarray(
        Image.fromarray(pred_unpad).resize((orig_w, orig_h), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    conf_orig = np.asarray(
        Image.fromarray(conf_unpad, mode="F").resize((orig_w, orig_h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )

    # 7. Extract class areas, confidences, and total food pixels
    areas: List[Dict[str, Any]] = []
    detections: List[Dict[str, Any]] = []
    total_pixels = orig_w * orig_h
    food_pixels = 0

    for class_id in np.unique(pred_orig):
        class_id = int(class_id)
        if class_id == 0 or class_id not in class_names:
            continue
        mask = pred_orig == class_id
        count = int(mask.sum())
        fraction = count / total_pixels
        mean_conf = float(conf_orig[mask].mean()) if count > 0 else 0.0
        if fraction < 0.001 or mean_conf < confidence_thresh:
            continue
        class_name = class_names[class_id]
        food_pixels += count
        areas.append({"class": class_name, "area_percent": round(fraction * 100, 2)})
        detections.append({"class": class_name, "confidence": round(mean_conf, 4)})

    areas.sort(key=lambda item: item["area_percent"], reverse=True)
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return areas, detections, food_pixels


def segmentation_summary_tiled(
    image: Image.Image,
    segmenter: torch.nn.Module,
    device: torch.device,
    class_names: Dict[int, str],
    tile_size: int = 512,
    overlap: int = 64,
    confidence_thresh: float = 0.25,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Segment original-resolution image pixels using overlapping tiles (Rollback Strategy)."""
    width, height = image.size
    pixels = np.asarray(image, dtype=np.uint8)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def tile_starts(length: int) -> List[int]:
        if length <= tile_size:
            return [0]
        stride = tile_size - overlap
        starts = list(range(0, length - tile_size + 1, stride))
        final_start = length - tile_size
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    def blend_weights(start: int, valid_length: int, full_length: int) -> np.ndarray:
        weights = np.ones(valid_length, dtype=np.float32)
        ramp_length = min(overlap, valid_length)
        if ramp_length and start > 0:
            weights[:ramp_length] = np.linspace(0.05, 1.0, ramp_length, dtype=np.float32)
        if ramp_length and start + valid_length < full_length:
            weights[-ramp_length:] = np.minimum(
                weights[-ramp_length:],
                np.linspace(1.0, 0.05, ramp_length, dtype=np.float32),
            )
        return weights

    pred = np.zeros((height, width), dtype=np.uint8)
    confidence_map = np.zeros((height, width), dtype=np.float32)
    best_score = np.full((height, width), -1.0, dtype=np.float32)

    for y in tile_starts(height):
        valid_height = min(tile_size, height - y)
        input_height = max(32, ((valid_height + 31) // 32) * 32)
        y_weights = blend_weights(y, valid_height, height)

        for x in tile_starts(width):
            valid_width = min(tile_size, width - x)
            input_width = max(32, ((valid_width + 31) // 32) * 32)
            x_weights = blend_weights(x, valid_width, width)

            tile = pixels[y:y + valid_height, x:x + valid_width].astype(np.float32) / 255.0
            normalized = (tile - mean) / std
            tensor_array = np.zeros((1, 3, input_height, input_width), dtype=np.float32)
            tensor_array[0, :, :valid_height, :valid_width] = normalized.transpose(2, 0, 1)
            tensor = torch.from_numpy(tensor_array).to(device)

            with torch.inference_mode():
                logits = segmenter(tensor)
                probabilities = torch.softmax(logits, dim=1)[0, :, :valid_height, :valid_width]
                confidence, prediction = probabilities.max(dim=0)

            tile_pred = prediction.cpu().numpy().astype(np.uint8)
            tile_conf = confidence.cpu().numpy().astype(np.float32)
            tile_weight = y_weights[:, None] * x_weights[None, :]
            score = tile_conf * tile_weight
            score_region = best_score[y:y + valid_height, x:x + valid_width]
            replace = score > score_region
            pred_region = pred[y:y + valid_height, x:x + valid_width]
            confidence_region = confidence_map[y:y + valid_height, x:x + valid_width]
            pred_region[replace] = tile_pred[replace]
            confidence_region[replace] = tile_conf[replace]
            score_region[replace] = score[replace]

    areas: List[Dict[str, Any]] = []
    detections: List[Dict[str, Any]] = []
    total_pixels = width * height
    food_pixels = 0
    for class_id in np.unique(pred):
        class_id = int(class_id)
        if class_id == 0 or class_id not in class_names:
            continue
        mask = pred == class_id
        count = int(mask.sum())
        fraction = count / total_pixels
        mean_conf = float(confidence_map[mask].mean())
        if fraction < 0.001 or mean_conf < confidence_thresh:
            continue
        class_name = class_names[class_id]
        food_pixels += count
        areas.append({"class": class_name, "area_percent": round(fraction * 100, 2)})
        detections.append({"class": class_name, "confidence": round(mean_conf, 4)})
    areas.sort(key=lambda item: item["area_percent"], reverse=True)
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return areas, detections, food_pixels


def segmentation_summary(
    image: Image.Image,
    segmenter: torch.nn.Module,
    device: torch.device,
    class_names: Dict[int, str],
    tile_size: int = 512,
    overlap: int = 64,
    confidence_thresh: float = 0.25,
    strategy: str = "letterbox",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Run 104-class semantic segmentation on DeepLabV3+ MobileNetV2.

    Strategies:
      - 'letterbox' (Default, Phase 2): Aspect-ratio-preserving resize with padding to 512x512.
        Preserves global plate context, eliminates seam-induced meat misclassifications, and runs in ~0.3s.
      - 'tiling' (Rollback Baseline): Overlapping 512x512 tiles with boundary blending.
    """
    selected_strategy = os.getenv("FOODSEG_STRATEGY", strategy).lower().strip()
    if selected_strategy in ("tiling", "tiles", "overlap"):
        return segmentation_summary_tiled(
            image, segmenter, device, class_names,
            tile_size=tile_size, overlap=overlap, confidence_thresh=confidence_thresh,
        )
    return segmentation_summary_letterbox(
        image, segmenter, device, class_names,
        target_size=tile_size, confidence_thresh=confidence_thresh,
    )


def detect_top_plate_scale(image: Image.Image, plate_diameter_cm: float, food_pixels: int) -> Dict[str, Any]:
    """Calibrate top-view pixel scale from fully visible plate rim using Hough circles."""
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    short_side = min(image.size)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=short_side * 0.45,
        param1=100, param2=30, minRadius=int(short_side * 0.28),
        maxRadius=int(short_side * 0.58),
    )
    if circles is None:
        return {"detected": False, "confidence": "low", "reason": "plate rim not detected"}

    h, w = gray.shape
    cx0, cy0 = w / 2, h / 2
    candidates = []
    for x, y, radius in circles[0]:
        if x - radius < 2 or y - radius < 2 or x + radius > w - 2 or y + radius > h - 2:
            continue
        center_offset = np.hypot((x - cx0) / w, (y - cy0) / h)
        candidates.append((center_offset, -radius, float(x), float(y), float(radius)))
    if not candidates:
        return {"detected": False, "confidence": "low", "reason": "plate appears cropped"}

    offset, _, cx, cy, radius = min(candidates)
    diameter_px = radius * 2
    pixels_per_cm = diameter_px / plate_diameter_cm
    plate_area_px = np.pi * radius * radius
    food_to_plate_ratio = food_pixels / plate_area_px
    if food_to_plate_ratio > 0.95:
        return {"detected": False, "confidence": "low", "reason": "candidate plate circle is inconsistent with food-mask area"}
    footprint_cm2 = food_to_plate_ratio * (np.pi * (plate_diameter_cm / 2) ** 2)
    confidence = "medium" if offset < 0.12 else "low"
    return {
        "detected": True,
        "confidence": confidence,
        "plate_diameter_px": round(diameter_px, 1),
        "pixels_per_cm": round(pixels_per_cm, 2),
        "food_footprint_cm2": round(footprint_cm2, 1),
        "method": "Hough circle estimate from top-view plate rim",
    }

