# -*- coding: utf-8 -*-
"""
experiments/compare_production_strategies.py
============================================
Compares Strategy 1 (Tiling - Rollback Baseline) vs Strategy 3 (Letterbox Padding - Production)
across the 5 benchmark images evaluated in Phase 2.

Checks:
- Mask dimensions
- Class mapping & area fractions
- Plate calibration (Hough circles, px_per_cm, footprint cm²)
- Weight estimation outputs (with mock/parsed components)
- Inference latency (seconds)
"""

import os
import sys
import time
import json
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

from vision.segmentation import (
    load_segmenter,
    segmentation_summary_letterbox,
    segmentation_summary_tiled,
    detect_top_plate_scale,
)
from vision.estimation import build_weight_estimates

DENSITY_TABLE = json.loads((ROOT / "vision" / "food_density.json").read_text(encoding="utf-8"))


def main():
    model_path = ROOT / "vision" / "models" / "FoodSeg103" / "best_model.pth"
    labels_path = ROOT / "vision" / "models" / "FoodSeg103" / "labels_config.json"

    print("Loading segmentation model...")
    segmenter, class_names, device = load_segmenter(model_path, labels_path)
    print(f"Loaded on {device}")

    # The 5 evaluation images
    user_up = Path(r"C:\Users\abhiram\.gemini\antigravity\brain\646b060f-3532-41f7-b070-d2153b69c57c\.user_uploaded")
    image_specs = [
        ("Image 1: Sample Meal (Noodles & Chicken)", ROOT / "FrontEnd" / "images" / "sample_meal_top.jpg", 25.0),
        ("Image 2: Indian Meal (Rice, Curry, Chicken)", user_up / "media_1794333290174_811312a7.jpg", 26.0),
        ("Image 3: Plated Rice & Curry", user_up / "media_1794335734564_d0f40611.jpg", 25.0),
        ("Image 4: Overhead Food Plate", user_up / "media_1794338750993_1520fdb2.jpg", 26.0),
        ("Image 5: Dual-View Top View", user_up / "media_1793870701374_81eb298c.jpg", 25.0),
    ]

    results = []

    print("\n" + "="*80)
    print("  PRODUCTION COMPARISON: LETTERBOX PADDING (PROD) vs OVERLAPPING TILING (BASELINE)")
    print("="*80)

    for label, img_path, plate_dia in image_specs:
        if not img_path.is_file():
            print(f"Skipping {label} (not found: {img_path})")
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # 1. Tiling
        t0 = time.perf_counter()
        t_areas, t_dets, t_px = segmentation_summary_tiled(img, segmenter, device, class_names)
        t_lat = time.perf_counter() - t0
        t_scale = detect_top_plate_scale(img, plate_dia, t_px)

        # 2. Letterbox (Production)
        t0 = time.perf_counter()
        l_areas, l_dets, l_px = segmentation_summary_letterbox(img, segmenter, device, class_names)
        l_lat = time.perf_counter() - t0
        l_scale = detect_top_plate_scale(img, plate_dia, l_px)

        # Build mock components from top areas for weight estimation comparison
        # Use top 2 areas as components
        def make_components(areas):
            comps = []
            for a in areas[:3]:
                comps.append({
                    "name": a["class"],
                    "food_area_fraction": round(a["area_percent"] / 100.0, 3),
                    "elevation": "medium",
                    "side_view_used": True,
                    "height_cm_min": 1.0,
                    "height_cm_max": 2.5,
                    "shape_factor": 0.7,
                    "volume_ml_min": 50.0,
                    "volume_ml_max": 150.0,
                    "confidence": "high",
                })
            return comps

        t_comps = make_components(t_areas)
        l_comps = make_components(l_areas)
        t_weights = build_weight_estimates(t_comps, t_scale, DENSITY_TABLE)
        l_weights = build_weight_estimates(l_comps, l_scale, DENSITY_TABLE)

        speedup = t_lat / l_lat if l_lat > 0 else 1.0

        rec = {
            "label": label,
            "filename": img_path.name,
            "dimensions": f"{w}x{h}",
            "tiling": {
                "latency_s": round(t_lat, 3),
                "food_pixels": t_px,
                "classes": [f"{a['class']} ({a['area_percent']}%)" for a in t_areas[:3]],
                "plate_detected": t_scale.get("detected", False),
                "plate_footprint_cm2": t_scale.get("food_footprint_cm2", 0),
                "weights": [f"{w_['name']}: {w_['weight_g']['estimate']}g" for w_ in t_weights[:2]],
            },
            "letterbox": {
                "latency_s": round(l_lat, 3),
                "food_pixels": l_px,
                "classes": [f"{a['class']} ({a['area_percent']}%)" for a in l_areas[:3]],
                "plate_detected": l_scale.get("detected", False),
                "plate_footprint_cm2": l_scale.get("food_footprint_cm2", 0),
                "weights": [f"{w_['name']}: {w_['weight_g']['estimate']}g" for w_ in l_weights[:2]],
            },
            "speedup": round(speedup, 1),
        }
        results.append(rec)

        print(f"\n--- {label} ({w}x{h} px) ---")
        print(f"  [Tiling]    Latency: {t_lat:.3f}s | Classes: {rec['tiling']['classes']} | Plate: {rec['tiling']['plate_detected']} ({rec['tiling']['plate_footprint_cm2']} cm²)")
        print(f"              Weights: {rec['tiling']['weights']}")
        print(f"  [Letterbox] Latency: {l_lat:.3f}s ({speedup:.1f}x faster) | Classes: {rec['letterbox']['classes']} | Plate: {rec['letterbox']['plate_detected']} ({rec['letterbox']['plate_footprint_cm2']} cm²)")
        print(f"              Weights: {rec['letterbox']['weights']}")

    out_file = ROOT / "experiments" / "results" / "production_comparison_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed comparison to: {out_file}")


if __name__ == "__main__":
    main()
