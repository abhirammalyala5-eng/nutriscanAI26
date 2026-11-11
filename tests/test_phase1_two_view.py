# -*- coding: utf-8 -*-
"""
tests/test_phase1_two_view.py
=============================
Unit and integration test suite validating Phase 1 improvements:
1. Both top and side images are accepted and processed.
2. Both images reach the VLM when provided, with explicit view prompts.
3. Side-view elevation modulates height, volume, and gram weights.
4. Missing, invalid, or duplicate side images are handled safely.
5. Plate calibration failure triggers truthful, uncalibrated fallback reporting.
6. Misleading "Qwen two-view volume" claims are completely eliminated.
7. Weight estimates remain finite, non-negative, and physically bounded.

All VLM calls are mocked. Tests do NOT require an active Ollama server or network services.
"""

import io
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
from PIL import Image

# Ensure project roots are in sys.path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "vision") not in sys.path:
    sys.path.insert(0, str(ROOT / "vision"))

from vision.vlm import (
    query_qwen_vlm,
    parse_component_response,
    extract_universal_components,
    get_vlm_thumbnail_b64,
    ELEVATION_FACTORS,
)
from vision.estimation import density_category, build_weight_estimates
from vision.app import app
from starlette.testclient import TestClient


DENSITY_TABLE = {
    "cooked_grain": {"min": 0.70, "max": 0.90},
    "legume_or_curry": {"min": 0.90, "max": 1.15},
    "cooked_meat": {"min": 0.85, "max": 1.10},
    "bread": {"min": 0.35, "max": 0.55},
    "papad": {"min": 0.20, "max": 0.40},
    "mixed_food": {"min": 0.60, "max": 1.10},
}


def _make_dummy_jpeg(color=(200, 100, 50), size=(100, 100)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestPhase1TwoViewProcessing(unittest.IsolatedAsyncioTestCase):

    async def test_vlm_prompt_dual_images(self):
        """Verify both top and side images reach Ollama with explicit two-view prompting."""
        top_b64 = "TOP_IMAGE_B64_MOCK"
        side_b64 = "SIDE_IMAGE_B64_MOCK"

        mock_reply = {
            "message": {
                "content": (
                    '{\n'
                    '  "dish": "Chicken Biryani with Raita",\n'
                    '  "side_view_observation": "Biryani is piled into a tall heaped mound; raita is flat in a bowl.",\n'
                    '  "components": [\n'
                    '    {"name": "Biryani", "fraction": 0.7, "elevation": "heaped"},\n'
                    '    {"name": "Raita", "fraction": 0.3, "elevation": "flat"}\n'
                    '  ]\n'
                    '}'
                )
            }
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_reply
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            result = await query_qwen_vlm(
                top_thumb_b64=top_b64,
                plate_diameter_cm=25.0,
                ollama_url="http://127.0.0.1:11434",
                qwen_model="qwen3-vl:2b",
                side_thumb_b64=side_b64,
            )

            # 1. Assert both images were sent in the payload
            self.assertEqual(mock_post.call_count, 1)
            call_kwargs = mock_post.call_args[1]
            payload = call_kwargs["json"]
            sent_images = payload["messages"][0]["images"]
            self.assertEqual(len(sent_images), 2, "Expected exactly 2 images sent to Ollama")
            self.assertEqual(sent_images[0], top_b64)
            self.assertEqual(sent_images[1], side_b64)

            # 2. Assert prompt explicitly labels Image 1 as Overhead and Image 2 as Horizontal Side
            prompt = payload["messages"][0]["content"]
            self.assertIn("OVERHEAD (TOP-DOWN)", prompt)
            self.assertIn("HORIZONTAL (SIDE-LEVEL)", prompt)
            self.assertIn("elevation", prompt)

            # 3. Assert parsed result retains side-view observations and elevation
            self.assertEqual(result["dish"].lower(), "chicken biryani with raita")
            self.assertIn("tall heaped mound", result["side_view_observation"])
            self.assertEqual(len(result["components"]), 2)

            comp_biryani = result["components"][0]
            self.assertEqual(comp_biryani["name"], "Biryani")
            self.assertEqual(comp_biryani["elevation"], "heaped")
            self.assertTrue(comp_biryani["side_view_used"])

            comp_raita = result["components"][1]
            self.assertEqual(comp_raita["name"], "Raita")
            self.assertEqual(comp_raita["elevation"], "flat")
            self.assertTrue(comp_raita["side_view_used"])

    async def test_vlm_prompt_single_image(self):
        """Verify single image fallback sends only top view and flags side_view_used as False."""
        top_b64 = "TOP_ONLY_B64_MOCK"

        mock_reply = {
            "message": {
                "content": (
                    '{\n'
                    '  "dish": "Steamed Rice",\n'
                    '  "components": [\n'
                    '    {"name": "Rice", "fraction": 1.0, "elevation": "medium"}\n'
                    '  ]\n'
                    '}'
                )
            }
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_reply
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            result = await query_qwen_vlm(
                top_thumb_b64=top_b64,
                plate_diameter_cm=25.0,
                ollama_url="http://127.0.0.1:11434",
                qwen_model="qwen3-vl:2b",
                side_thumb_b64=None,
            )

            call_kwargs = mock_post.call_args[1]
            payload = call_kwargs["json"]
            sent_images = payload["messages"][0]["images"]
            self.assertEqual(len(sent_images), 1, "Only 1 image should be sent when side view is None")
            self.assertEqual(sent_images[0], top_b64)

            self.assertFalse(result["components"][0]["side_view_used"])

    def test_side_elevation_modulates_volume_and_weight(self):
        """Verify side-view elevation ('flat' vs 'heaped') produces measurably different volume and weight."""
        plate_scale = {
            "detected": True,
            "confidence": "medium",
            "pixels_per_cm": 20.0,
            "food_footprint_cm2": 250.0,
        }

        comp_flat = [{
            "name": "Rice",
            "food_area_fraction": 0.5,
            "elevation": "flat",
            "side_view_used": True,
            "height_cm_min": 0.06 * 0.55 * 25.0,
            "height_cm_max": 0.15 * 0.55 * 25.0,
            "shape_factor": 0.55,
            "volume_ml_min": 50.0,
            "volume_ml_max": 100.0,
            "confidence": "high",
        }]

        comp_heaped = [{
            "name": "Rice",
            "food_area_fraction": 0.5,
            "elevation": "heaped",
            "side_view_used": True,
            "height_cm_min": 0.06 * 1.35 * 25.0,
            "height_cm_max": 0.15 * 1.35 * 25.0,
            "shape_factor": 0.85,
            "volume_ml_min": 150.0,
            "volume_ml_max": 300.0,
            "confidence": "high",
        }]

        est_flat = build_weight_estimates(comp_flat, plate_scale, DENSITY_TABLE)[0]
        est_heaped = build_weight_estimates(comp_heaped, plate_scale, DENSITY_TABLE)[0]

        # 1. Heaped mound must produce higher volume and weight than flat spread
        self.assertGreater(
            est_heaped["volume_ml"]["estimate"],
            est_flat["volume_ml"]["estimate"],
            "Heaped mound must have greater volume than flat spread on identical surface area",
        )
        self.assertGreater(
            est_heaped["weight_g"]["estimate"],
            est_flat["weight_g"]["estimate"],
            "Heaped mound must yield greater gram weight than flat spread",
        )

        # 2. Verify accurate method string naming the elevation profile
        self.assertIn("side-view elevation profile ('flat')", est_flat["method"])
        self.assertIn("side-view elevation profile ('heaped')", est_heaped["method"])
        self.assertIn("Calibrated top footprint", est_flat["method"])

    def test_plate_calibration_failure_reporting(self):
        """Verify failure of plate calibration triggers uncalibrated heuristic fallback and eliminates false claims."""
        plate_scale = {
            "detected": False,
            "confidence": "low",
            "reason": "plate rim not detected",
        }

        comps = [{
            "name": "Curry",
            "food_area_fraction": 0.4,
            "elevation": "medium",
            "side_view_used": False,
            "height_cm_min": 1.0,
            "height_cm_max": 2.5,
            "shape_factor": 0.7,
            "volume_ml_min": 80.0,
            "volume_ml_max": 160.0,
            "confidence": "medium",
        }]

        est = build_weight_estimates(comps, plate_scale, DENSITY_TABLE)[0]

        # 1. Confidence must be downgraded to low
        self.assertEqual(est["confidence"], "low")

        # 2. Method must truthfully state uncalibrated heuristic fallback
        self.assertIn("Heuristic volume fallback (uncalibrated: plate rim not detected)", est["method"])

        # 3. Method must NOT claim "Qwen two-view volume estimate"
        self.assertNotIn("Qwen two-view volume", est["method"])

        # 4. Weights must be finite and positive
        self.assertGreater(est["weight_g"]["estimate"], 0)
        self.assertGreater(est["weight_g"]["min"], 0)
        self.assertGreaterEqual(est["weight_g"]["max"], est["weight_g"]["min"])

    def test_weight_estimates_finite_and_non_negative(self):
        """Verify boundary cases (zero area, unknown items) always yield finite, non-negative numbers."""
        plate_scale = {"detected": False, "reason": "uncalibrated"}
        comps = [
            {
                "name": "Papad",
                "food_area_fraction": 0.05,
                "elevation": "flat",
                "side_view_used": False,
                "height_cm_min": 0.2,
                "height_cm_max": 0.5,
                "shape_factor": 0.3,
                "volume_ml_min": 10.0,
                "volume_ml_max": 25.0,
                "confidence": "medium",
            },
            {
                "name": "Unknown Dish Item",
                "food_area_fraction": 0.1,
                "elevation": "medium",
                "side_view_used": False,
                "height_cm_min": 0.5,
                "height_cm_max": 1.0,
                "shape_factor": 0.6,
                "volume_ml_min": 20.0,
                "volume_ml_max": 40.0,
                "confidence": "low",
            }
        ]

        estimates = build_weight_estimates(comps, plate_scale, DENSITY_TABLE)
        for item in estimates:
            self.assertGreaterEqual(item["weight_g"]["estimate"], 5.0)
            self.assertGreaterEqual(item["weight_g"]["min"], 5.0)
            self.assertGreaterEqual(item["weight_g"]["max"], 5.0)
            self.assertTrue(all(isinstance(v, int) for v in item["weight_g"].values()))


class TestPhase1APIEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    @patch("vision.app.query_qwen_vlm", new_callable=AsyncMock)
    def test_vision_endpoint_accepts_dual_images(self, mock_qwen):
        """Verify /vision endpoint accepts both top and side images, runs successfully, and reports two-view status."""
        mock_qwen.return_value = {
            "dish": "Paneer Rice Thali",
            "side_view_observation": "Rice is mounded; paneer cubes visible in elevation.",
            "components": [
                {"name": "Rice", "food_area_fraction": 0.6, "elevation": "heaped", "side_view_used": True,
                 "height_cm_min": 1.5, "height_cm_max": 3.0, "shape_factor": 0.8,
                 "volume_ml_min": 150.0, "volume_ml_max": 300.0, "confidence": "high"},
                {"name": "Paneer", "food_area_fraction": 0.4, "elevation": "medium", "side_view_used": True,
                 "height_cm_min": 1.0, "height_cm_max": 2.0, "shape_factor": 0.7,
                 "volume_ml_min": 100.0, "volume_ml_max": 200.0, "confidence": "high"},
            ]
        }

        top_bytes = _make_dummy_jpeg((220, 180, 140))
        side_bytes = _make_dummy_jpeg((100, 150, 200))  # distinct from top

        files = {
            "top_image": ("top.jpg", top_bytes, "image/jpeg"),
            "side_image": ("side.jpg", side_bytes, "image/jpeg"),
        }
        data = {"plate_diameter_cm": "25.0"}

        resp = self.client.post("/vision", files=files, data=data)
        self.assertEqual(resp.status_code, 200, f"Failed with: {resp.text}")
        body = resp.json()

        # Check response structure
        self.assertEqual(body["dish"], "Paneer Rice Thali")
        self.assertTrue(body["image_processing"]["side_view_provided"])
        self.assertIn("Dual 512px LANCZOS", body["image_processing"]["qwen_input"])
        self.assertIn("Rice is mounded", body["side_view_observation"])
        self.assertIn("Dual views analyzed", " ".join(body["limitations"]))
        self.assertIn("Rice", body["ingredients"])
        self.assertIn("Paneer", body["ingredients"])

    @patch("vision.app.query_qwen_vlm", new_callable=AsyncMock)
    def test_vision_endpoint_handles_missing_side_image(self, mock_qwen):
        """Verify /vision endpoint safely handles omitted side image without failing."""
        mock_qwen.return_value = {
            "dish": "Idli Sambar",
            "side_view_observation": "Side view not provided; standard category baseline assumed",
            "components": [
                {"name": "Idli", "food_area_fraction": 0.6, "elevation": "medium", "side_view_used": False,
                 "height_cm_min": 1.0, "height_cm_max": 2.0, "shape_factor": 0.7,
                 "volume_ml_min": 120.0, "volume_ml_max": 240.0, "confidence": "medium"},
            ]
        }

        top_bytes = _make_dummy_jpeg((240, 240, 240))
        files = {
            "top_image": ("top.jpg", top_bytes, "image/jpeg"),
        }
        data = {"plate_diameter_cm": "20.0"}

        resp = self.client.post("/vision", files=files, data=data)
        self.assertEqual(resp.status_code, 200, f"Failed with: {resp.text}")
        body = resp.json()

        self.assertFalse(body["image_processing"]["side_view_provided"])
        self.assertIn("Single 512px LANCZOS", body["image_processing"]["qwen_input"])
        self.assertIn("Only the overhead top-view photo was available", " ".join(body["limitations"]))

    @patch("vision.app.query_qwen_vlm", new_callable=AsyncMock)
    def test_vision_endpoint_detects_duplicate_side_image(self, mock_qwen):
        """Verify /vision endpoint detects when side image is an identical duplicate of top image."""
        mock_qwen.return_value = {
            "dish": "Dosa Plate",
            "side_view_observation": "Side view not provided; standard category baseline assumed",
            "components": [
                {"name": "Dosa", "food_area_fraction": 1.0, "elevation": "flat", "side_view_used": False,
                 "height_cm_min": 0.2, "height_cm_max": 0.5, "shape_factor": 0.4,
                 "volume_ml_min": 50.0, "volume_ml_max": 100.0, "confidence": "medium"},
            ]
        }

        top_bytes = _make_dummy_jpeg((180, 140, 80))
        files = {
            "top_image": ("top.jpg", top_bytes, "image/jpeg"),
            "side_image": ("side.jpg", top_bytes, "image/jpeg"),  # EXACT same bytes
        }
        data = {"plate_diameter_cm": "30.0"}

        resp = self.client.post("/vision", files=files, data=data)
        self.assertEqual(resp.status_code, 200, f"Failed with: {resp.text}")
        body = resp.json()

        # Duplicate image must NOT be reported as a distinct side view
        self.assertFalse(body["image_processing"]["side_view_provided"])


if __name__ == "__main__":
    unittest.main()
