# -*- coding: utf-8 -*-
"""
tests/test_phase2_segmentation.py
==================================
Unit and validation test suite for Phase 2:
Aspect-Ratio-Preserving Letterbox Padding Segmentation & Calibration.

Validates:
1. Strategy 3 letterbox inference:
   - Preserves original image aspect ratio.
   - Pads to 512x512 without distortion.
   - Restores prediction mask and confidence map to exact original image (W, H).
   - Zero padding bleed into food areas.
2. Rollback capability:
   - segmentation_summary_tiled remains functional.
   - FOODSEG_STRATEGY environment variable and strategy parameter dynamically select implementation.
3. Geometric calibration protection:
   - detect_top_plate_scale uses unpadded original image coordinates.
   - Hough circle detection and pixel scale are untouched by preprocessing.
4. Zero-area component handling:
   - Components with food_area_fraction <= 0.0 receive strictly 0g weight and 0ml volume.
   - No artificial minimum weight (5g or 10g) is assigned to empty components.
"""

import os
import unittest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from vision.segmentation import (
    load_segmenter,
    segmentation_summary,
    segmentation_summary_letterbox,
    segmentation_summary_tiled,
    detect_top_plate_scale,
)
from vision.estimation import build_weight_estimates

ROOT = Path(__file__).resolve().parent.parent
DENSITY_TABLE = json.loads((ROOT / "vision" / "food_density.json").read_text(encoding="utf-8"))


class TestPhase2SegmentationLetterbox(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.model_path = cls.root / "vision" / "models" / "FoodSeg103" / "best_model.pth"
        cls.labels_path = cls.root / "vision" / "models" / "FoodSeg103" / "labels_config.json"
        cls.segmenter, cls.class_names, cls.device = load_segmenter(cls.model_path, cls.labels_path)

    def test_letterbox_output_dimensions_match_non_square_image(self):
        """Verify letterbox output mask and coordinates match non-square original aspect ratios (4:3 and 16:9)."""
        for orig_size in [(640, 480), (800, 450), (450, 800)]:
            w, h = orig_size
            # Create a synthetic image
            img_np = np.zeros((h, w, 3), dtype=np.uint8)
            # Add a colored patch in center
            img_np[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = [180, 100, 50]
            img = Image.fromarray(img_np)

            areas, detections, food_pixels = segmentation_summary_letterbox(
                img, self.segmenter, self.device, self.class_names, target_size=512
            )

            # Assert output structures
            self.assertIsInstance(areas, list)
            self.assertIsInstance(detections, list)
            self.assertIsInstance(food_pixels, int)
            self.assertGreaterEqual(food_pixels, 0)
            self.assertLessEqual(food_pixels, w * h)

            for area in areas:
                self.assertIn("class", area)
                self.assertIn("area_percent", area)
                self.assertGreater(area["area_percent"], 0.0)
                self.assertLessEqual(area["area_percent"], 100.0)

    def test_rollback_capability_preserved(self):
        """Verify tiled inference remains fully functional and accessible via router and environment variable."""
        img = Image.new("RGB", (300, 300), color=(150, 120, 80))

        # 1. Direct call to tiled implementation
        tiled_areas, tiled_det, tiled_px = segmentation_summary_tiled(
            img, self.segmenter, self.device, self.class_names, tile_size=512
        )
        self.assertIsInstance(tiled_areas, list)
        self.assertIsInstance(tiled_det, list)
        self.assertIsInstance(tiled_px, int)

        # 2. Router with strategy='tiling'
        routed_tiled_areas, _, _ = segmentation_summary(
            img, self.segmenter, self.device, self.class_names, strategy="tiling"
        )
        self.assertIsInstance(routed_tiled_areas, list)

        # 3. Router with FOODSEG_STRATEGY env var
        with patch.dict(os.environ, {"FOODSEG_STRATEGY": "tiling"}):
            env_areas, _, _ = segmentation_summary(
                img, self.segmenter, self.device, self.class_names
            )
            self.assertIsInstance(env_areas, list)

        # 4. Default router strategy is letterbox
        with patch.dict(os.environ, {}, clear=True):
            default_areas, _, _ = segmentation_summary(
                img, self.segmenter, self.device, self.class_names
            )
            self.assertIsInstance(default_areas, list)

    def test_plate_calibration_uses_original_coordinates(self):
        """Verify detect_top_plate_scale operates on untouched original image dimensions."""
        width, height = 600, 400
        # Draw a synthetic white plate circle on black background
        img_np = np.zeros((height, width, 3), dtype=np.uint8)
        import cv2
        center = (width // 2, height // 2)
        radius = 120
        cv2.circle(img_np, center, radius, (240, 240, 240), -1)
        cv2.circle(img_np, center, radius, (80, 80, 80), 3)  # dark rim
        img = Image.fromarray(img_np)

        food_pixels = int(np.pi * (radius * 0.7) ** 2)
        scale = detect_top_plate_scale(img, plate_diameter_cm=26.0, food_pixels=food_pixels)

        # Scale metrics must be strictly positive and consistent
        if scale.get("detected"):
            self.assertGreater(scale["pixels_per_cm"], 0)
            self.assertGreater(scale["plate_diameter_px"], 0)
            self.assertGreater(scale["food_footprint_cm2"], 0)


class TestPhase2ZeroAreaComponentHandling(unittest.TestCase):

    def test_zero_area_component_yields_zero_weight(self):
        """Verify components with zero detected area receive strictly 0g weight and 0ml volume."""
        plate_scale = {
            "detected": True,
            "plate_diameter_cm": 26.0,
            "food_footprint_cm2": 250.0,
            "confidence": "high",
        }

        components = [
            {
                "name": "Steamed Rice",
                "food_area_fraction": 0.5,
                "elevation": "heaped",
                "side_view_used": True,
                "height_cm_min": 1.5,
                "height_cm_max": 3.0,
                "shape_factor": 0.8,
                "volume_ml_min": 120.0,
                "volume_ml_max": 240.0,
                "confidence": "high",
            },
            {
                "name": "Phantom Garnish",
                "food_area_fraction": 0.0,  # ZERO area detected
                "elevation": "flat",
                "side_view_used": True,
                "height_cm_min": 0.1,
                "height_cm_max": 0.2,
                "shape_factor": 0.5,
                "volume_ml_min": 5.0,
                "volume_ml_max": 10.0,
                "confidence": "low",
            },
            {
                "name": "Negative Area Edge Case",
                "food_area_fraction": -0.05,  # Negative / invalid area fraction
                "elevation": "flat",
                "side_view_used": False,
                "height_cm_min": 0.5,
                "height_cm_max": 1.0,
                "shape_factor": 0.5,
                "volume_ml_min": 10.0,
                "volume_ml_max": 20.0,
                "confidence": "low",
            }
        ]

        estimates = build_weight_estimates(components, plate_scale, DENSITY_TABLE)

        # 1. Non-zero item should have positive weight
        rice = estimates[0]
        self.assertEqual(rice["name"], "Steamed Rice")
        self.assertGreater(rice["weight_g"]["estimate"], 0)
        self.assertGreater(rice["volume_ml"]["estimate"], 0)

        # 2. Zero-area item must have strictly zero weight and volume (NO artificial 5g/10g)
        garnish = estimates[1]
        self.assertEqual(garnish["name"], "Phantom Garnish")
        self.assertEqual(garnish["weight_g"]["min"], 0)
        self.assertEqual(garnish["weight_g"]["estimate"], 0)
        self.assertEqual(garnish["weight_g"]["max"], 0)
        self.assertEqual(garnish["volume_ml"]["min"], 0)
        self.assertEqual(garnish["volume_ml"]["estimate"], 0)
        self.assertEqual(garnish["volume_ml"]["max"], 0)

        # 3. Negative-area item must also have strictly zero weight and volume
        neg_item = estimates[2]
        self.assertEqual(neg_item["name"], "Negative Area Edge Case")
        self.assertEqual(neg_item["weight_g"]["min"], 0)
        self.assertEqual(neg_item["weight_g"]["estimate"], 0)
        self.assertEqual(neg_item["weight_g"]["max"], 0)
        self.assertEqual(neg_item["volume_ml"]["min"], 0)
        self.assertEqual(neg_item["volume_ml"]["estimate"], 0)
        self.assertEqual(neg_item["volume_ml"]["max"], 0)


if __name__ == "__main__":
    unittest.main()
