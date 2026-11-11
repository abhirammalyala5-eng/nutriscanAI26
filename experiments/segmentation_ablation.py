# -*- coding: utf-8 -*-
"""
experiments/segmentation_ablation.py
====================================
Phase 2: FoodSeg103 Segmentation Inference Strategy Ablation Study

Compares three preprocessing and inference strategies on DeepLabV3+ MobileNetV2:
1. Overlapping 512x512 Native-Resolution Tiling (Current Production Baseline)
2. Whole-Image Direct Resize to 512x512 (Standard DeepLabV3+ Whole-Plate Evaluation)
3. Aspect-Ratio-Preserving Resize with Letterbox Padding to 512x512

Outputs:
- Quantitative metrics per strategy: Inference time (s), detected classes, area percentages, confidence.
- Qualitative visual comparison panels saved as PNG images:
  [Original Image] | [Strategy 1: Tiling] | [Strategy 2: Direct Resize] | [Strategy 3: Letterbox Pad]
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

from vision.segmentation import load_segmenter

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

# Color palette for masks (distinct RGB colors)
PALETTE = [
    [0, 0, 0],         # 0: background
    [230, 25, 75],     # 1: red
    [60, 180, 75],     # 2: green
    [255, 225, 25],    # 3: yellow
    [0, 130, 200],     # 4: blue
    [245, 130, 48],    # 5: orange
    [145, 30, 180],    # 6: purple
    [70, 240, 240],    # 7: cyan
    [240, 50, 230],    # 8: magenta
    [210, 245, 60],    # 9: lime
    [250, 190, 212],   # 10: pink
    [0, 128, 128],     # 11: teal
    [220, 190, 255],   # 12: lavender
    [170, 110, 40],    # 13: brown
    [255, 250, 200],   # 14: beige
    [128, 0, 0],       # 15: maroon
    [170, 255, 195],   # 16: mint
    [128, 128, 0],     # 17: olive
    [255, 215, 180],   # 18: apricot
    [0, 0, 128],       # 19: navy
    [128, 128, 128],   # 20: grey
]


def extract_areas_and_detections(
    pred: np.ndarray,
    conf_map: np.ndarray,
    class_names: Dict[int, str],
    conf_thresh: float = 0.25,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    height, width = pred.shape
    total_pixels = height * width
    areas = []
    detections = []
    food_pixels = 0

    for class_id in np.unique(pred):
        class_id = int(class_id)
        if class_id == 0 or class_id not in class_names:
            continue
        mask = pred == class_id
        count = int(mask.sum())
        fraction = count / total_pixels
        mean_conf = float(conf_map[mask].mean()) if count > 0 else 0.0
        if fraction < 0.001 or mean_conf < conf_thresh:
            continue
        class_name = class_names[class_id]
        food_pixels += count
        areas.append({"class": class_name, "class_id": class_id, "area_percent": round(fraction * 100, 2)})
        detections.append({"class": class_name, "class_id": class_id, "confidence": round(mean_conf, 4)})

    areas.sort(key=lambda item: item["area_percent"], reverse=True)
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return areas, detections, food_pixels


# ─────────────────────────────────────────────────────────────
# Strategy 1: Current Overlapping Native-Resolution Tiling
# ─────────────────────────────────────────────────────────────
def run_tiling_inference(
    image: Image.Image,
    segmenter: torch.nn.Module,
    device: torch.device,
    class_names: Dict[int, str],
    tile_size: int = 512,
    overlap: int = 64,
    conf_thresh: float = 0.25,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    width, height = image.size
    pixels = np.asarray(image, dtype=np.uint8)

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
    tile_count = 0

    y_starts = tile_starts(height)
    x_starts = tile_starts(width)

    for y in y_starts:
        valid_height = min(tile_size, height - y)
        input_height = max(32, ((valid_height + 31) // 32) * 32)
        y_weights = blend_weights(y, valid_height, height)

        for x in x_starts:
            valid_width = min(tile_size, width - x)
            input_width = max(32, ((valid_width + 31) // 32) * 32)
            x_weights = blend_weights(x, valid_width, width)
            tile_count += 1

            tile = pixels[y:y + valid_height, x:x + valid_width].astype(np.float32) / 255.0
            normalized = (tile - MEAN) / STD
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

    areas, detections, food_pixels = extract_areas_and_detections(pred, confidence_map, class_names, conf_thresh)
    latency = time.perf_counter() - t0

    return {
        "strategy": "Overlapping Tiling (512x512)",
        "pred_mask": pred,
        "confidence_map": confidence_map,
        "areas": areas,
        "detections": detections,
        "food_pixels": food_pixels,
        "tile_count": tile_count,
        "latency_sec": round(latency, 3),
    }


# ─────────────────────────────────────────────────────────────
# Strategy 2: Whole-Image Direct Resize to 512x512
# ─────────────────────────────────────────────────────────────
def run_direct_resize_inference(
    image: Image.Image,
    segmenter: torch.nn.Module,
    device: torch.device,
    class_names: Dict[int, str],
    target_size: int = 512,
    conf_thresh: float = 0.25,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    orig_w, orig_h = image.size

    # Bilinear resize full plate to 512x512
    resized_img = image.resize((target_size, target_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized_img, dtype=np.float32) / 255.0
    normalized = (pixels - MEAN) / STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = segmenter(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, prediction = probabilities.max(dim=0)

    pred_512 = prediction.cpu().numpy().astype(np.uint8)
    conf_512 = confidence.cpu().numpy().astype(np.float32)

    # Nearest-neighbor resize mask back to original resolution
    pred_orig = np.asarray(
        Image.fromarray(pred_512).resize((orig_w, orig_h), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    conf_orig = np.asarray(
        Image.fromarray(conf_512, mode="F").resize((orig_w, orig_h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )

    areas, detections, food_pixels = extract_areas_and_detections(pred_orig, conf_orig, class_names, conf_thresh)
    latency = time.perf_counter() - t0

    return {
        "strategy": "Direct Whole-Image Resize (512x512)",
        "pred_mask": pred_orig,
        "confidence_map": conf_orig,
        "areas": areas,
        "detections": detections,
        "food_pixels": food_pixels,
        "tile_count": 1,
        "latency_sec": round(latency, 3),
    }


# ─────────────────────────────────────────────────────────────
# Strategy 3: Aspect-Ratio Preserving Resize with Letterbox Padding
# ─────────────────────────────────────────────────────────────
def run_letterbox_pad_inference(
    image: Image.Image,
    segmenter: torch.nn.Module,
    device: torch.device,
    class_names: Dict[int, str],
    target_size: int = 512,
    conf_thresh: float = 0.25,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    orig_w, orig_h = image.size

    # Calculate scale factor to fit inside 512x512 preserving aspect ratio
    scale = target_size / max(orig_w, orig_h)
    new_w = max(1, min(target_size, int(round(orig_w * scale))))
    new_h = max(1, min(target_size, int(round(orig_h * scale))))

    resized_img = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Create 512x512 canvas with neutral zero padding
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas.paste(resized_img, (pad_x, pad_y))

    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    normalized = (pixels - MEAN) / STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = segmenter(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, prediction = probabilities.max(dim=0)

    pred_canvas = prediction.cpu().numpy().astype(np.uint8)
    conf_canvas = confidence.cpu().numpy().astype(np.float32)

    # Unpad: crop out the active food area
    pred_unpad = pred_canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    conf_unpad = conf_canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w]

    # Resize back to original resolution
    pred_orig = np.asarray(
        Image.fromarray(pred_unpad).resize((orig_w, orig_h), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    conf_orig = np.asarray(
        Image.fromarray(conf_unpad, mode="F").resize((orig_w, orig_h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )

    areas, detections, food_pixels = extract_areas_and_detections(pred_orig, conf_orig, class_names, conf_thresh)
    latency = time.perf_counter() - t0

    return {
        "strategy": "Letterbox Pad (512x512 Aspect-Preserving)",
        "pred_mask": pred_orig,
        "confidence_map": conf_orig,
        "areas": areas,
        "detections": detections,
        "food_pixels": food_pixels,
        "tile_count": 1,
        "latency_sec": round(latency, 3),
    }


# ─────────────────────────────────────────────────────────────
# Visual Comparison Renderer
# ─────────────────────────────────────────────────────────────
def render_visual_comparison(
    image: Image.Image,
    results: List[Dict[str, Any]],
    class_names: Dict[int, str],
    output_path: Path,
    image_title: str,
):
    orig_np = np.asarray(image)
    h, w, _ = orig_np.shape

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f"Segmentation Preprocessing Ablation: {image_title} ({w}x{h})", fontsize=16, fontweight="bold")

    # Panel 0: Original Photo
    axes[0].imshow(orig_np)
    axes[0].set_title("Input Meal Photo", fontsize=12, fontweight="semibold")
    axes[0].axis("off")

    # Panels 1-3: The 3 strategies
    used_classes = set()
    for idx, res in enumerate(results):
        ax = axes[idx + 1]
        pred_mask = res["pred_mask"]
        latency = res["latency_sec"]
        strat = res["strategy"]
        tiles = res.get("tile_count", 1)

        # Create color overlay
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        unique_ids = np.unique(pred_mask)

        for c_id in unique_ids:
            if c_id == 0:
                continue
            used_classes.add(int(c_id))
            color = PALETTE[int(c_id) % len(PALETTE)]
            color_mask[pred_mask == c_id] = color

        # Alpha blend overlay on original image
        overlay = orig_np.copy()
        mask_binary = pred_mask > 0
        overlay[mask_binary] = (orig_np[mask_binary] * 0.45 + color_mask[mask_binary] * 0.55).astype(np.uint8)

        ax.imshow(overlay)

        # Summary label
        top_classes = [f"{a['class']} ({a['area_percent']}%)" for a in res["areas"][:3]]
        class_str = ", ".join(top_classes) if top_classes else "No food detected"
        ax.set_title(f"{strat}\nTime: {latency}s | Tiles: {tiles}\n{class_str}", fontsize=10)
        ax.axis("off")

    # Legend at bottom
    patches = []
    for c_id in sorted(used_classes):
        c_name = class_names.get(c_id, f"Class {c_id}")
        color = [c / 255.0 for c in PALETTE[c_id % len(PALETTE)]]
        patches.append(mpatches.Patch(color=color, label=f"{c_name} (ID {c_id})"))

    if patches:
        fig.legend(handles=patches, loc="lower center", ncol=min(6, len(patches)), bbox_to_anchor=(0.5, 0.0), fontsize=9)

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visual comparison: {output_path}")


def main():
    model_path = ROOT / "vision" / "models" / "FoodSeg103" / "best_model.pth"
    labels_path = ROOT / "vision" / "models" / "FoodSeg103" / "labels_config.json"

    print("================================================================")
    print("  NUTRI-SCAN-AI: PHASE 2 SEGMENTATION STRATEGY ABLATION STUDY   ")
    print("================================================================")
    print(f"Loading FoodSeg103 DeepLabV3+ MobileNetV2 from: {model_path}")
    segmenter, class_names, device = load_segmenter(model_path, labels_path)
    print(f"Model loaded successfully on: {device}\n")

    # Candidate evaluation images
    test_images: List[Tuple[str, Path]] = []

    # 1. Sample meal top (noodles and chicken)
    p_sample = ROOT / "FrontEnd" / "images" / "sample_meal_top.jpg"
    if p_sample.is_file():
        test_images.append(("Sample Meal (Top View)", p_sample))

    # 2. User uploaded test images
    user_up = Path(r"C:\Users\abhiram\.gemini\antigravity\brain\646b060f-3532-41f7-b070-d2153b69c57c\.user_uploaded")
    img_candidates = [
        ("Indian Meal - Rice, Curry, Chicken", user_up / "media_1794333290174_811312a7.jpg"),
        ("Plated Rice & Curry Meal", user_up / "media_1794335734564_d0f40611.jpg"),
        ("Overhead Food Plate", user_up / "media_1794338750993_1520fdb2.jpg"),
        ("Dual-View Meal Top View", user_up / "media_1793870701374_81eb298c.jpg"),
    ]
    for label, p in img_candidates:
        if p.is_file():
            test_images.append((label, p))

    if not test_images:
        print("ERROR: No test images found!")
        return

    results_dir = ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []

    for img_idx, (label, img_path) in enumerate(test_images, 1):
        print(f"\n--- [{img_idx}/{len(test_images)}] Evaluating: {label} ---")
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        print(f"Image Resolution: {w}x{h} px ({img_path.name})")

        # 1. Overlapping Tiling
        res_tile = run_tiling_inference(img, segmenter, device, class_names)
        print(f"  [1] Tiling (512px): {res_tile['latency_sec']}s ({res_tile['tile_count']} tiles) | Classes: {[a['class'] for a in res_tile['areas'][:4]]}")

        # 2. Direct Resize
        res_direct = run_direct_resize_inference(img, segmenter, device, class_names)
        print(f"  [2] Direct Resize (512x512): {res_direct['latency_sec']}s (1 pass) | Classes: {[a['class'] for a in res_direct['areas'][:4]]}")

        # 3. Letterbox Pad
        res_pad = run_letterbox_pad_inference(img, segmenter, device, class_names)
        print(f"  [3] Letterbox Pad (512x512): {res_pad['latency_sec']}s (1 pass) | Classes: {[a['class'] for a in res_pad['areas'][:4]]}")

        results_list = [res_tile, res_direct, res_pad]

        # Render visual comparison
        out_vis = results_dir / f"ablation_eval_img_{img_idx}.png"
        render_visual_comparison(img, results_list, class_names, out_vis, label)

        summary_records.append({
            "image_index": img_idx,
            "label": label,
            "filename": img_path.name,
            "resolution": f"{w}x{h}",
            "tiling": {
                "latency_sec": res_tile["latency_sec"],
                "tiles": res_tile["tile_count"],
                "classes": res_tile["areas"],
                "food_pixels": res_tile["food_pixels"],
            },
            "direct_resize": {
                "latency_sec": res_direct["latency_sec"],
                "tiles": 1,
                "classes": res_direct["areas"],
                "food_pixels": res_direct["food_pixels"],
            },
            "letterbox_pad": {
                "latency_sec": res_pad["latency_sec"],
                "tiles": 1,
                "classes": res_pad["areas"],
                "food_pixels": res_pad["food_pixels"],
            },
        })

    # Save summary JSON
    json_path = results_dir / "ablation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)
    print(f"\nSaved quantitative summary to: {json_path}")
    print("\nAblation study execution complete!")


if __name__ == "__main__":
    main()

