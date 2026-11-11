# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
🥦 NutraVision Prediction API
==============================
FastAPI service orchestrating nutritional regression predictions, dual-view integration,
and FrontEnd UI static file delivery.
"""

import os
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import httpx

import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(MODEL_DIR.parent) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR.parent))

try:
    from ML.schemas import (
        IngredientIn, NutrientResult, RecipeTotals,
        PredictRequest, PredictResponse,
        VisionRequest, VisionNutrientResult, VisionPredictResponse
    )
    from ML.predictor import NutritionPredictor, TARGETS
except ModuleNotFoundError:
    from schemas import (
        IngredientIn, NutrientResult, RecipeTotals,
        PredictRequest, PredictResponse,
        VisionRequest, VisionNutrientResult, VisionPredictResponse
    )
    from predictor import NutritionPredictor, TARGETS

warnings.filterwarnings("ignore")

FRONTEND_DIR = MODEL_DIR.parent / "FrontEnd"
VISION_SERVICE_URL = os.getenv("VISION_SERVICE_URL", "http://127.0.0.1:8001/vision")

predictor = NutritionPredictor(MODEL_DIR)

# Backward-compatibility aliases for tests/scripts importing from ML.app
catboost_models = predictor.catboost_models
svd = predictor.svd
vectorizer = predictor.vectorizer

# ─────────────────────────────────────────────────────────────
# 2. FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="NutraVision Prediction API",
    description=(
        "Predict nutritional values for meals and ingredients using "
        "CatBoost multi-target regression with 3-tier contextual preparation resolution.\n\n"
        "Supported targets: energy_kcal, carb_g, protein_g, fat_g, fibre_g"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# 3. ROUTES
# ─────────────────────────────────────────────────────────────
@app.get("/api/info", tags=["Health"], summary="API info")
def root():
    return {
        "service": "NutraVision Prediction API",
        "status":  "running",
        "version": "1.0.0",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"], summary="Health check")
def health():
    return {
        "status":  "ok",
        "models_loaded": list(catboost_models.keys()),
        "svd_components": svd.n_components,
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Nutrition"],
    summary="Predict nutritional values for a list of ingredients",
)
def predict(request: PredictRequest) -> PredictResponse:
    return predictor.predict_ingredients(request.ingredients)


@app.post(
    "/predict/vision",
    response_model=VisionPredictResponse,
    tags=["Nutrition"],
    summary="Predict nutrition from vision model output",
)
def predict_vision(request: VisionRequest) -> VisionPredictResponse:
    pairs: List[Tuple[str, float]] = []

    if request.ingredients:
        for name, weight in request.ingredients.items():
            try:
                w_val = float(weight)
                if w_val < 0:
                    raise HTTPException(status_code=422, detail=f"Weight cannot be negative for '{name}': {weight}")
                pairs.append((str(name), w_val))
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid weight for ingredient '{name}': {weight}"
                )
    elif request.estimates:
        for est in request.estimates:
            name = est.get("name", "unknown")
            wg = est.get("weight_g", {})
            if isinstance(wg, dict):
                weight = wg.get("estimate", 100.0)
            else:
                weight = float(wg) if wg else 100.0
            pairs.append((str(name), float(weight)))
    else:
        raise HTTPException(
            status_code=422,
            detail="Request must contain either 'ingredients' dict or 'estimates' array."
        )

    return predictor.predict_vision_pairs(request.dish, pairs)


@app.post(
    "/analyze/dual",
    tags=["Integration"],
    summary="End-to-End Dual Image Food Analysis",
    description="Forwards dual photos to Vision Service (8001), strips extra keys, and predicts nutrition via CatBoost."
)
async def analyze_dual(
    top_image: UploadFile = File(..., description="Top-down overhead photo"),
    side_image: UploadFile = File(None, description="Side-view photo at plate level"),
    plate_diameter_cm: float = Form(25.0, description="Plate diameter in cm (default 25.0)"),
):
    if plate_diameter_cm <= 0:
        raise HTTPException(status_code=422, detail="plate_diameter_cm must be positive")

    top_bytes = await top_image.read()
    side_bytes = await side_image.read() if (side_image and getattr(side_image, "filename", None)) else None

    files = {
        "top_image": (top_image.filename or "top.jpg", top_bytes, top_image.content_type or "image/jpeg"),
    }
    if side_bytes:
        files["side_image"] = (
            side_image.filename or "side.jpg",
            side_bytes,
            side_image.content_type or "image/jpeg",
        )
    data = {"plate_diameter_cm": str(plate_diameter_cm)}

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            v_resp = await client.post(VISION_SERVICE_URL, files=files, data=data)
            v_resp.raise_for_status()
            vision_data = v_resp.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Vision API at {VISION_SERVICE_URL}. Ensure vision service is running on port 8001."
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Vision API returned error: {exc.response.text[:300]}"
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Vision communication failure: {exc}")

    dish = vision_data.get("dish")
    ingredients = vision_data.get("ingredients") or {}

    req = VisionRequest(dish=dish, ingredients=ingredients)
    nutr = predict_vision(req)

    return {
        "dish": dish,
        "ingredients": ingredients,
        "estimates": vision_data.get("estimates", []),
        "scale_reference": vision_data.get("scale_reference", {}),
        "nutrition": nutr.nutrition,
        "recipe_totals": nutr.recipe_totals,
        "limitations": vision_data.get("limitations", []),
        "segmentation": vision_data.get("segmentation", {}),
    }


# ─────────────────────────────────────────────────────────────
# 4. STATIC FRONTEND SERVING
# ─────────────────────────────────────────────────────────────
if FRONTEND_DIR.is_dir():
    images_dir = FRONTEND_DIR / "images"
    if images_dir.is_dir():
        app.mount("/images", StaticFiles(directory=images_dir), name="images")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/style.css", include_in_schema=False)
    async def serve_css():
        return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")

    @app.get("/script.js", include_in_schema=False)
    async def serve_js():
        return FileResponse(FRONTEND_DIR / "script.js", media_type="application/javascript")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
