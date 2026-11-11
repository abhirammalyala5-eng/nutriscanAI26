# -*- coding: utf-8 -*-
"""
vision/estimation.py
====================
Food density category mapping, IIIT-H empirical portion bounding, and calibrated
volume-to-gram weight estimation.
"""

from typing import List, Dict, Any


def density_category(name: str) -> str:
    """Classify ingredient into food density category."""
    name = name.lower()
    if any(x in name for x in ("papad", "appalam", "pappadam")):
        return "papad"
    if any(x in name for x in ("payasam", "kheer", "halwa", "gulab jamun", "sweet")):
        return "sweets"
    if any(x in name for x in ("rice", "noodle", "pasta", "grain", "biryani", "pulao", "khichdi", "bisi bele")):
        return "cooked_grain"
    if any(x in name for x in ("chicken", "beef", "pork", "lamb", "mutton", "fish", "meat", "shrimp", "prawn")):
        return "cooked_meat"
    if any(x in name for x in ("dal", "sambar", "rasam", "rajma", "chana", "chole", "pappu", "lentil", "bean", "chickpea", "curry", "stew", "salan")):
        return "legume_or_curry"
    if any(x in name for x in ("chutney", "pickle", "sauce", "gravy", "dressing")):
        return "sauce"
    if any(x in name for x in ("bread", "roti", "phulka", "chapati", "naan", "paratha", "flatbread", "puri", "bhatura")):
        return "bread"
    if any(x in name for x in ("cheese", "paneer", "yogurt", "curd", "dahi", "raita")):
        return "dairy"
    if any(x in name for x in ("fruit", "apple", "banana", "mango", "orange", "watermelon", "papaya", "muskmelon")):
        return "fruit"
    if any(x in name for x in ("vegetable", "poriyal", "fry", "sabzi", "aloo", "gobi", "onion", "pea", "carrot", "potato", "herb", "salad", "cilantro", "cabbage", "beetroot", "bhindi")):
        return "vegetable"
    return "mixed_food"


def build_weight_estimates(
    components: List[Dict[str, Any]],
    plate_scale: Dict[str, Any],
    density_table: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """Convert component area shares and elevations to calibrated gram weights using IIIT-H bounds."""
    estimates = []
    total_share = sum(item["food_area_fraction"] for item in components)
    default_density = density_table.get("mixed_food", {"min": 0.6, "max": 1.1})

    for item in components:
        category = density_category(item["name"])
        density = density_table.get(category, default_density)
        elevation = item.get("elevation", "medium")
        side_used = item.get("side_view_used", False)

        area_frac = float(item.get("food_area_fraction", 0.0))

        if plate_scale.get("detected") and total_share > 0:
            normalized_share = max(0.0, area_frac) / total_share
            area_cm2 = plate_scale["food_footprint_cm2"] * normalized_share
            shape = item["shape_factor"]
            vol_min = area_cm2 * item["height_cm_min"] * max(0.2, shape * 0.8)
            vol_max = area_cm2 * item["height_cm_max"] * min(1.0, shape * 1.2)
            if side_used:
                method = f"Calibrated top footprint ({area_cm2:.1f} cm²) × side-view elevation profile ('{elevation}') × geometric shape factor ({shape:.2f}) × density range"
            else:
                method = f"Calibrated top footprint ({area_cm2:.1f} cm²) × standard category height baseline × geometric shape factor ({shape:.2f}) × density range"
        else:
            vol_min, vol_max = item["volume_ml_min"], item["volume_ml_max"]
            reason = plate_scale.get("reason", "plate rim not detected")
            method = f"Heuristic volume fallback (uncalibrated: {reason}) × density range"

        # Check for zero detected area: do not assign artificial minimum weights to 0-area items
        if area_frac <= 0.0 or vol_max <= 0.0:
            vol_min = 0.0
            vol_max = 0.0
            grams_min = 0.0
            grams_max = 0.0
            estimate = 0.0
            w_min = 0
            w_max = 0
            w_est = 0
        else:
            grams_min = max(5.0, vol_min * density["min"])
            grams_max = max(grams_min, vol_max * density["max"])
            estimate = max(5.0, ((vol_min + vol_max) / 2) * ((density["min"] + density["max"]) / 2))

            # Empirical portion weight bounds from IIIT-H 3,278 weighed Indian servings
            name_lower = item["name"].lower()
            if "papad" in name_lower:
                estimate = min(estimate, 25.0)
                grams_min = min(grams_min, 5.0)
                grams_max = min(grams_max, 25.0)
            elif any(b in name_lower for b in ("phulka", "roti", "chapati")):
                if estimate > 90.0:
                    estimate = min(estimate, 80.0)
            elif "chutney" in name_lower or "pickle" in name_lower:
                estimate = min(estimate, 45.0)

            w_est = max(5, round(estimate / 10) * 10)
            w_min = max(5, round(grams_min / 10) * 10)
            w_max = max(w_min, round(grams_max / 10) * 10)

        confidence = item.get("confidence", "medium")
        if not plate_scale.get("detected"):
            confidence = "low"
        elif plate_scale.get("confidence") == "low" and confidence == "high":
            confidence = "medium"

        estimates.append({
            "name": item["name"],
            "elevation": elevation,
            "side_view_used": side_used,
            "volume_ml": {"min": round(vol_min), "estimate": round((vol_min + vol_max) / 2), "max": round(vol_max)},
            "density_g_per_ml": {"min": density["min"], "max": density["max"]},
            "weight_g": {"min": w_min, "estimate": w_est, "max": w_max},
            "confidence": confidence,
            "method": method,
        })
    return estimates

