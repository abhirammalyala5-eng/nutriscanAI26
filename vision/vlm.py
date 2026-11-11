# -*- coding: utf-8 -*-
"""
vision/vlm.py
=============
Local Vision-Language Model (Qwen3-VL:2B) client, thread-optimized Ollama payload,
and universal regex parser for multi-component plated meals.
"""

import base64
import io
import json
import logging
import os
import re
from typing import Any, Dict, List
from PIL import Image
import httpx

try:
    from vision.estimation import density_category
except ModuleNotFoundError:
    from estimation import density_category

logger = logging.getLogger("foodvision")


def get_vlm_thumbnail_b64(image: Image.Image, max_dim: int = 512) -> str:
    """Create a compact JPEG thumbnail for fast, accurate local VLM visual processing."""
    thumb = image.copy()
    thumb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


ELEVATION_FACTORS: Dict[str, Dict[str, float]] = {
    "flat": {"height_mult": 0.55, "shape_override": 0.45},
    "low": {"height_mult": 0.80, "shape_override": 0.55},
    "medium": {"height_mult": 1.00, "shape_override": 0.65},
    "heaped": {"height_mult": 1.35, "shape_override": 0.85},
    "tall": {"height_mult": 1.35, "shape_override": 0.85},
}

# Strict JSON Schema for Ollama structured output enforcement
VLM_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "dish": {"type": "string"},
        "side_view_observation": {"type": "string"},
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "fraction": {"type": "number"},
                    "elevation": {
                        "type": "string",
                        "enum": ["flat", "low", "medium", "heaped"],
                    },
                },
                "required": ["name", "fraction", "elevation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["dish", "components"],
    "additionalProperties": False,
}

REJECT_CONVERSATIONAL_PREFIXES = (
    "then ", "first ", "second ", "third ", "also ", "finally ",
    "and ", "let's check", "let's ", "check ", "s check", "we see ",
    "there is ", "there are ", "here is ", "based on "
)

INVALID_NAME_STOPWORDS = {
    "dish", "fraction", "component", "total", "schema", "first", "second", "third",
    "proportion", "observation", "image", "plate", "meal", "food", "ingredients",
    "item", "serving", "layer", "view", "check", "s check", "let's check",
    "the flatbreads", "the vegetables", "flatbreads", "vegetables",
}


def is_valid_food_component_name(name: str) -> bool:
    """Validate that name represents a legitimate food ingredient and not conversational reasoning text."""
    if not isinstance(name, str):
        return False
    clean = name.strip()
    # Strip any enclosing punctuation or quotes
    clean = re.sub(r"^[\"\'\-\*•\s]+|[\"\'\-\*•\s:]+$", "", clean)
    if len(clean) < 2 or len(clean) > 50:
        return False
    clean_lower = clean.lower()

    # Reject conversational sentence starters
    for prefix in REJECT_CONVERSATIONAL_PREFIXES:
        if clean_lower.startswith(prefix) or clean_lower == prefix.strip():
            return False

    # Reject exact generic stopwords
    if clean_lower in INVALID_NAME_STOPWORDS:
        return False

    # Must contain at least two letters
    if not re.search(r"[A-Za-z]{2,}", clean):
        return False

    # Reject conversational thinking fragments containing phrases like let's, we have, etc.
    if any(frag in clean_lower for frag in ("let's", "let us", "we have", "i see", "look at", "check:", "observe")):
        return False

    return True


def extract_universal_components(text: str) -> Dict[str, Any]:
    """Universal regex extraction for any dish and components, strictly filtering out conversational monologue."""
    dish = ""
    dish_match = re.search(r'"dish"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if not dish_match:
        dish_match = re.search(r'(?:dish is|dish name is|meal is)\s*[:"]?\s*([A-Za-z0-9\s\-/&]+?)(?:["\n,.]|components)', text, re.IGNORECASE)
    if dish_match:
        cand_dish = dish_match.group(1).strip().title()
        if is_valid_food_component_name(cand_dish):
            dish = cand_dish

    side_obs = ""
    side_match = re.search(r'"side_view_observation"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if side_match:
        side_obs = side_match.group(1).strip()

    # Match JSON component objects with name, fraction, and optional elevation
    comp_matches = []
    for m in re.finditer(r'\{[^{}]*?"name"\s*:\s*"([^"]+)"[^{}]*?"(?:fraction|food_area_fraction)"\s*:\s*([0-9.]+)[^{}]*?\}', text, re.IGNORECASE):
        block = m.group(0)
        c_name = m.group(1)
        c_frac = m.group(2)
        elev_m = re.search(r'"elevation"\s*:\s*"([A-Za-z_-]+)"', block, re.IGNORECASE)
        c_elev = elev_m.group(1).lower() if elev_m else "medium"
        comp_matches.append((c_name, c_frac, c_elev))

    if not comp_matches:
        raw_pairs = re.findall(
            r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"(?:fraction|food_area_fraction)"\s*:\s*([0-9.]+)\s*\}',
            text,
            re.IGNORECASE,
        )
        comp_matches = [(n, f, "medium") for n, f in raw_pairs]

    if not comp_matches:
        raw_pairs = [
            (name, frac)
            for frac, name in re.findall(
                r'\{\s*"(?:fraction|food_area_fraction)"\s*:\s*([0-9.]+)\s*,\s*"name"\s*:\s*"([^"]+)"\s*\}',
                text,
                re.IGNORECASE,
            )
        ]
        comp_matches = [(n, f, "medium") for n, f in raw_pairs]

    if not comp_matches:
        # Match lines with item: fraction while permitting hyphens and apostrophes in food names
        bullet_matches = re.findall(r'[-*•]?\s*["\']?([A-Za-z\s\'\-]+?)["\']?\s*:\s*(?:about\s*)?(0\.\d+|\d+%)', text)
        comp_matches = [
            (name.strip(), val.replace('%', '') if '%' not in val else str(float(val.replace('%', ''))/100), "medium")
            for name, val in bullet_matches if len(name.strip()) > 2
        ]

    components = []
    seen = set()
    for item in comp_matches:
        raw_name = item[0].strip()
        # Strip leading bullet points, dashes, quotes
        raw_name = re.sub(r"^[\"\'\-\*•\s]+|[\"\'\-\*•\s:]+$", "", raw_name).strip()
        # Clean title case
        name = raw_name.title()
        frac_str = item[1]
        elevation = item[2].lower() if len(item) > 2 and item[2].lower() in ELEVATION_FACTORS else "medium"
        clean_lower = name.lower()

        # Strict validation: reject conversational prefixes, thinking fragments, and stopwords
        if not is_valid_food_component_name(name):
            continue
        if clean_lower in seen:
            continue
        seen.add(clean_lower)
        try:
            val = float(frac_str)
            if val > 1.0:
                val = val / 100.0
            components.append({"name": name, "fraction": max(0.01, min(val, 1.0)), "elevation": elevation})
        except ValueError:
            pass

    for prefix in ("The ", "A ", "An "):
        if dish.startswith(prefix):
            dish = dish[len(prefix):].strip()

    if (not dish or dish.lower() in ("curry", "food", "meal", "plated meal")) and components:
        sorted_comps = sorted(components, key=lambda c: c["fraction"], reverse=True)
        if len(sorted_comps) >= 2:
            dish = f"{sorted_comps[0]['name']} with {sorted_comps[1]['name']}"
        elif sorted_comps:
            dish = f"{sorted_comps[0]['name']} Plate"

    if not dish:
        dish = "Plated Meal"
    return {"dish": dish, "side_view_observation": side_obs, "components": components}


def parse_component_response(text: str, plate_diameter_cm: float, has_side_view: bool = False) -> Dict[str, Any]:
    """Parse strictly structured JSON into geometric meal components with side-view elevation."""
    text = text.strip()
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()

    value = None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Minimal repair attempt for truncated trailing bracket
        repaired = re.sub(r',\s*\{[^{}]*$', '', text)
        if repaired.count('[') > repaired.count(']'):
            repaired += ']'
        if repaired.count('{') > repaired.count('}'):
            repaired += '}'
        try:
            value = json.loads(repaired)
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                try:
                    value = json.loads(match.group(0))
                except Exception:
                    pass

    # Never extract components from free-form reasoning text if JSON structure is missing
    if not isinstance(value, dict) or not isinstance(value.get("components"), list) or not value.get("components"):
        logger.error("VLM response could not be parsed as valid JSON: %r", text[:300])
        raise ValueError("VLM response did not contain valid structured JSON components")

    dish = str(value.get("dish", "Plated Meal")).strip()
    if not is_valid_food_component_name(dish):
        dish = "Plated Meal"

    side_obs = str(value.get("side_view_observation", "")).strip()
    if not side_obs:
        side_obs = "Side-view elevation profile observed" if has_side_view else "Side view not provided; standard category baseline assumed"

    components = []
    for item in value["components"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        raw_name = item["name"].strip()
        name = raw_name.title()

        # Strict validation: reject conversational fragments ("Then the...", "Let's check", etc.)
        if not is_valid_food_component_name(name):
            logger.warning("Rejected invalid / conversational component name: %r", raw_name)
            continue

        try:
            area_val = item.get("fraction", item.get("food_area_fraction", 0.3))
            area_share = max(0.01, min(float(area_val), 1.0))

            cat = density_category(name)
            if cat == "bread":
                default_hmin, default_hmax, default_shape = 0.015, 0.035, 0.5
            elif cat == "legume_or_curry":
                default_hmin, default_hmax, default_shape = 0.05, 0.12, 0.8
            elif cat == "cooked_grain":
                default_hmin, default_hmax, default_shape = 0.06, 0.15, 0.85
            elif cat == "cooked_meat":
                default_hmin, default_hmax, default_shape = 0.04, 0.10, 0.75
            elif cat == "papad":
                default_hmin, default_hmax, default_shape = 0.01, 0.02, 0.3
            else:
                default_hmin, default_hmax, default_shape = 0.03, 0.08, 0.65

            raw_elev = str(item.get("elevation", "")).lower().strip()
            if raw_elev in ELEVATION_FACTORS:
                elevation = raw_elev
            elif cat in ("bread", "papad"):
                elevation = "flat"
            else:
                elevation = "medium"

            elev_info = ELEVATION_FACTORS.get(elevation, ELEVATION_FACTORS["medium"])
            elev_mult = elev_info["height_mult"]
            if not has_side_view:
                elev_mult = 1.0

            h_min_ratio = float(item.get("height_fraction_min", default_hmin * elev_mult))
            h_max_ratio = float(item.get("height_fraction_max", default_hmax * elev_mult))
            height_fraction_min = max(0.005, min(h_min_ratio, 0.25))
            height_fraction_max = max(height_fraction_min, min(h_max_ratio, 0.25))
            height_min = height_fraction_min * plate_diameter_cm
            height_max = height_fraction_max * plate_diameter_cm

            base_shape = elev_info["shape_override"] or default_shape
            shape_factor = max(0.2, min(float(item.get("shape_factor", base_shape)), 1.0))

            low = float(item.get("volume_ml_min", max(20.0, area_share * 180.0 * elev_mult)))
            high = float(item.get("volume_ml_max", max(low * 1.3, area_share * 400.0 * elev_mult)))
            if high < 5.0:
                low = max(20.0, area_share * 180.0 * elev_mult)
                high = max(low * 1.3, area_share * 400.0 * elev_mult)
        except (KeyError, TypeError, ValueError):
            continue

        confidence = item.get("confidence", "high")
        if confidence not in ("low", "medium", "high"):
            confidence = "high"

        components.append({
            "name": name,
            "food_area_fraction": area_share,
            "elevation": elevation,
            "side_view_used": has_side_view,
            "height_cm_min": height_min,
            "height_cm_max": height_max,
            "shape_factor": shape_factor,
            "volume_ml_min": low,
            "volume_ml_max": high,
            "confidence": confidence,
        })

    if not components:
        raise ValueError("No valid culinary food components identified in VLM response")

    # Normalize area fractions so they sum to 1.0 within tolerance
    total_area = sum(c["food_area_fraction"] for c in components)
    if total_area > 0:
        for c in components:
            c["food_area_fraction"] = round(c["food_area_fraction"] / total_area, 3)

    return {"dish": dish, "side_view_observation": side_obs, "components": components}


async def query_qwen_vlm(
    top_thumb_b64: str,
    plate_diameter_cm: float,
    ollama_url: str,
    qwen_model: str,
    side_thumb_b64: str | None = None,
    num_threads: int = 4,
) -> Dict[str, Any]:
    """Execute optimized CPU VLM chat inference via Ollama with explicit thinking suppression and JSON Schema."""
    has_side = bool(side_thumb_b64 and side_thumb_b64 != top_thumb_b64)
    images = [top_thumb_b64, side_thumb_b64] if has_side else [top_thumb_b64]

    if has_side:
        prompt = (
            f"You are a professional culinary vision AI analyzing paired photos of a plated meal (plate diameter: {plate_diameter_cm:g} cm).\n"
            "Image 1 is the OVERHEAD (TOP-DOWN) view showing the 2D surface layout and component boundaries.\n"
            "Image 2 is the HORIZONTAL (SIDE-LEVEL) view showing the vertical food height.\n\n"
            "Tasks:\n"
            "1. Identify the exact dish name (e.g. Chicken Dum Biryani, Paneer Butter Masala with Naan, Steamed Rice with Dal).\n"
            "2. Identify visible food components (e.g. Rice, Curry, Dal, Roti, Chicken, Paneer, Salad, Vegetables, Onion) "
            "with top-view surface area fractions summing to 1.0.\n"
            "3. Observe Image 2 to classify side-view elevation as one of: 'flat', 'low', 'medium', 'heaped'.\n"
            "4. Provide a brief 1-sentence observation of mound elevation.\n\n"
            "Output strictly valid JSON matching schema:\n"
            '{"dish": "Exact Dish Name", "side_view_observation": "Brief elevation observation", '
            '"components": [{"name": "Food Name", "fraction": 0.6, "elevation": "medium"}]}'
        )
    else:
        prompt = (
            f"You are a professional culinary vision AI analyzing an overhead photo of a plated meal (plate diameter: {plate_diameter_cm:g} cm).\n"
            "Identify the exact dish name (e.g. Chicken Dum Biryani, Paneer Butter Masala with Naan, Steamed Rice with Dal) "
            "and visible food components (e.g. Rice, Curry, Dal, Roti, Chicken, Paneer, Salad, Vegetables, Onion) "
            "with top-view surface area fractions summing to 1.0.\n\n"
            "Output strictly valid JSON matching schema:\n"
            '{"dish": "Exact Dish Name", "side_view_observation": "Standard baseline elevation assumed (single overhead view)", '
            '"components": [{"name": "Food Name", "fraction": 0.6, "elevation": "medium"}]}'
        )

    # Messages array prefilled with closed thinking block to strictly suppress internal reasoning monologue
    messages = [
        {"role": "user", "content": prompt, "images": images},
        {"role": "assistant", "content": "<think>\n</think>\n"},
    ]

    payload = {
        "model": qwen_model,
        "messages": messages,
        "stream": False,
        "think": False,
        "format": VLM_OUTPUT_SCHEMA,
        "options": {
            "temperature": 0,
            "num_predict": 768,
            "num_ctx": 4096,
            "num_thread": num_threads,
        },
        "keep_alive": "10m",
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        response = await client.post(f"{ollama_url.rstrip('/')}/api/chat", json=payload)
        response.raise_for_status()

    ollama_result = response.json()
    done_reason = ollama_result.get("done_reason", "")
    if done_reason == "length":
        logger.warning("Qwen generation reached token limit (done_reason=length); output may be truncated.")

    message = ollama_result.get("message", {})
    content = message.get("content", "").strip()

    # STRICT RULE: Never interpret internal thinking text as a food-identification response!
    if not content:
        logger.error(
            "Qwen returned empty content (thinking suppressed or generation aborted). done_reason=%s, thinking_len=%d",
            done_reason, len(message.get("thinking", "")),
        )
        raise ValueError("Qwen returned empty content; model generation failed or was suppressed")

    return parse_component_response(content, plate_diameter_cm, has_side_view=has_side)
