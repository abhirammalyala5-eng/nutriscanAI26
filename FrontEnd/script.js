// FrontEnd/script.js - NutriScanAI Dual-View Controller & Interactive Engine

const topImageInput = document.getElementById("topImage");
const sideImageInput = document.getElementById("sideImage");
const topPreview = document.getElementById("topPreview");
const sidePreview = document.getElementById("sidePreview");
const topPlaceholder = document.getElementById("topPlaceholder");
const sidePlaceholder = document.getElementById("sidePlaceholder");
const topDropZone = document.getElementById("topDropZone");
const sideDropZone = document.getElementById("sideDropZone");
const topUploadPanel = document.getElementById("topUploadPanel");
const sideUploadPanel = document.getElementById("sideUploadPanel");
const plateDiameterInput = document.getElementById("plateDiameter");
const analyzeButton = document.getElementById("analyzeButton");
const resultDiv = document.getElementById("result");

// Global state for Human-in-the-Loop editing
window.currentAnalysisData = null;
window.currentIngredients = {};
window.currentDish = "Plated Meal";

// ─────────────────────────────────────────────────────────────
// 1. FILE UPLOAD & PREVIEW LOGIC
// ─────────────────────────────────────────────────────────────
function setPreviewImage(file, previewEl, placeholderEl) {
    if (!file) return;
    const url = URL.createObjectURL(file);
    previewEl.src = url;
    previewEl.style.display = "block";
    if (placeholderEl) placeholderEl.style.display = "none";
}

if (topImageInput) {
    topImageInput.addEventListener("change", function () {
        if (topImageInput.files.length > 0) {
            setPreviewImage(topImageInput.files[0], topPreview, topPlaceholder);
        }
    });
}

if (sideImageInput) {
    sideImageInput.addEventListener("change", function () {
        if (sideImageInput.files.length > 0) {
            setPreviewImage(sideImageInput.files[0], sidePreview, sidePlaceholder);
        }
    });
}

// Drag & Drop handlers
function setupDragAndDrop(panel, input, previewEl, placeholderEl) {
    if (!panel || !input) return;
    ["dragenter", "dragover"].forEach(evt => {
        panel.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            panel.classList.add("drag-over");
        });
    });
    ["dragleave", "drop"].forEach(evt => {
        panel.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            panel.classList.remove("drag-over");
        });
    });
    panel.addEventListener("drop", (e) => {
        if (e.dataTransfer && e.dataTransfer.files.length > 0) {
            input.files = e.dataTransfer.files;
            setPreviewImage(e.dataTransfer.files[0], previewEl, placeholderEl);
        }
    });
}

setupDragAndDrop(topUploadPanel, topImageInput, topPreview, topPlaceholder);
setupDragAndDrop(sideUploadPanel, sideImageInput, sidePreview, sidePlaceholder);

// ─────────────────────────────────────────────────────────────
// 2. QUICK DEMO LOAD & CLEAR FUNCTIONS
// ─────────────────────────────────────────────────────────────
async function loadSampleMeal(autoScroll = false) {
    try {
        const topResp = await fetch("images/sample_meal_top.jpg");
        const topBlob = await topResp.blob();
        const topFile = new File([topBlob], "sample_meal_top.jpg", { type: "image/jpeg" });

        const sideResp = await fetch("images/sample_meal_side.jpg");
        const sideBlob = await sideResp.blob();
        const sideFile = new File([sideBlob], "sample_meal_side.jpg", { type: "image/jpeg" });

        const dtTop = new DataTransfer();
        dtTop.items.add(topFile);
        topImageInput.files = dtTop.files;
        setPreviewImage(topFile, topPreview, topPlaceholder);

        const dtSide = new DataTransfer();
        dtSide.items.add(sideFile);
        sideImageInput.files = dtSide.files;
        setPreviewImage(sideFile, sidePreview, sidePlaceholder);

        if (autoScroll) {
            document.getElementById("analyzer").scrollIntoView({ behavior: "smooth" });
        }
    } catch (err) {
        console.error("Failed to load sample meal:", err);
    }
}

function clearUploads() {
    if (topImageInput) topImageInput.value = "";
    if (sideImageInput) sideImageInput.value = "";
    if (topPreview) { topPreview.src = ""; topPreview.style.display = "none"; }
    if (sidePreview) { sidePreview.src = ""; sidePreview.style.display = "none"; }
    if (topPlaceholder) topPlaceholder.style.display = "flex";
    if (sidePlaceholder) sidePlaceholder.style.display = "flex";
    resultDiv.innerHTML = `
        <div class="initial-prompt">
            <div class="prompt-icon">🍽️</div>
            <h3>Photos Cleared</h3>
            <p>Upload new photos or click <strong>Load Sample Meal</strong> to begin.</p>
        </div>
    `;
}

function escapeHtml(text) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return String(text).replace(/[&<>"']/g, (m) => map[m]);
}

function getConfidenceBadge(confidence) {
    const conf = String(confidence || "medium").toLowerCase();
    if (conf.includes("high") || conf === "high") {
        return `<span class="badge badge-high">High</span>`;
    } else if (conf.includes("low") || conf === "low") {
        return `<span class="badge badge-low">Low</span>`;
    }
    return `<span class="badge badge-med">Medium</span>`;
}

// ─────────────────────────────────────────────────────────────
// 3. MAIN DUAL-VIEW INFERENCE ORCHESTRATION
// ─────────────────────────────────────────────────────────────
async function analyzeFood() {
    if (!topImageInput || topImageInput.files.length === 0) {
        resultDiv.innerHTML = `
            <div class="initial-prompt" style="color: #b91c1c; background: #fef2f2; border-color: #fecaca;">
                <p>⚠️ <strong>Missing Top-Down Photo:</strong> Please select at least an overhead (top-down) photo of the plate.</p>
            </div>
        `;
        return;
    }

    const topFile = topImageInput.files[0];
    const sideFile = (sideImageInput && sideImageInput.files.length > 0) ? sideImageInput.files[0] : null;
    const plateDiameter = plateDiameterInput ? parseFloat(plateDiameterInput.value) || 25.0 : 25.0;

    const formData = new FormData();
    formData.append("top_image", topFile);
    if (sideFile) formData.append("side_image", sideFile);
    formData.append("plate_diameter_cm", plateDiameter.toString());

    analyzeButton.disabled = true;
    analyzeButton.innerHTML = `<span>⏳</span><span>Analyzing Plate Elevation & Geometry...</span>`;

    let _elapsed = 0;
    const steps = [
        { label: "Validating images & running Hough circle plate rim calibration", pct: 15 },
        { label: "DeepLabV3+ semantic segmentation on native overlapping tiles", pct: 35 },
        { label: "Qwen3-VL multimodal culinary reasoning on food components", pct: 75 },
        { label: "Applying IIIT-H density boundaries & running CatBoost models", pct: 95 }
    ];

    function renderStepperHtml(sec) {
        let activeIdx = 0;
        if (sec >= 4 && sec < 12) activeIdx = 1;
        else if (sec >= 12 && sec < 35) activeIdx = 2;
        else if (sec >= 35) activeIdx = 3;

        const currentStep = steps[activeIdx];
        const pct = Math.min(100, Math.max(10, currentStep.pct + Math.floor((sec % 8) * 1.5)));

        let listHtml = "";
        steps.forEach((s, idx) => {
            const statusClass = idx === activeIdx ? "active" : (idx < activeIdx ? "done" : "");
            const bullet = idx < activeIdx ? "✓" : (idx + 1);
            listHtml += `
                <div class="step-item ${statusClass}">
                    <div class="step-bullet">${bullet}</div>
                    <span>${s.label}</span>
                </div>
            `;
        });

        return `
            <div class="pipeline-progress-box">
                <div class="stepper-header">
                    <span class="stepper-title">🧠 AI Processing Pipeline</span>
                    <span class="stepper-timer">⏱️ ${_elapsed}s elapsed</span>
                </div>
                <div class="progress-bar-track">
                    <div class="progress-bar-fill" style="width: ${pct}%"></div>
                </div>
                <div class="stepper-list">
                    ${listHtml}
                </div>
            </div>
        `;
    }

    resultDiv.innerHTML = renderStepperHtml(0);
    const _timerInterval = setInterval(() => {
        _elapsed++;
        resultDiv.innerHTML = renderStepperHtml(_elapsed);
    }, 1000);

    try {
        const visionUrl = window.VISION_API_URL || "http://127.0.0.1:8001/vision";
        const visionResponse = await fetch(visionUrl, { method: "POST", body: formData });
        const visionData = await visionResponse.json();

        if (!visionResponse.ok || visionData.error || visionData.detail) {
            clearInterval(_timerInterval);
            const errMsg = visionData.error || visionData.detail || `Vision server returned status ${visionResponse.status}`;
            resultDiv.innerHTML = `
                <div class="initial-prompt" style="color: #b91c1c; background: #fef2f2; border-color: #fecaca;">
                    <p>❌ <strong>Vision Analysis Failed:</strong> ${escapeHtml(errMsg)}</p>
                </div>
            `;
            return;
        }

        const mlPayload = {
            dish: visionData.dish || null,
            ingredients: visionData.ingredients || {}
        };

        const nutritionUrl = window.NUTRITION_API_URL || "http://127.0.0.1:8000/predict/vision";
        const mlResponse = await fetch(nutritionUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(mlPayload),
        });

        const mlData = await mlResponse.json();
        clearInterval(_timerInterval);

        if (!mlResponse.ok || mlData.error || mlData.detail) {
            const errMsg = mlData.error || mlData.detail || `Nutrition regression returned status ${mlResponse.status}`;
            resultDiv.innerHTML = `
                <div class="initial-prompt" style="color: #b91c1c; background: #fef2f2; border-color: #fecaca;">
                    <p>❌ <strong>Nutrition Calculation Failed:</strong> ${escapeHtml(errMsg)}</p>
                </div>
            `;
            return;
        }

        window.currentDish = visionData.dish || "Plated Meal";
        window.currentIngredients = { ...(visionData.ingredients || {}) };
        window.currentAnalysisData = {
            ...visionData,
            nutrition: mlData.nutrition,
            recipe_totals: mlData.recipe_totals,
        };

        renderAnalysisResults(window.currentAnalysisData);

    } catch (error) {
        clearInterval(_timerInterval);
        console.error("API error:", error);
        resultDiv.innerHTML = `
            <div class="initial-prompt" style="color: #b91c1c; background: #fef2f2; border-color: #fecaca;">
                <p>❌ <strong>Connection Error:</strong> ${escapeHtml(error.message || "Unknown error")}.<br>
                Ensure Vision Service (port 8001) and Nutrition Service (port 8000) are both running.</p>
            </div>
        `;
    } finally {
        analyzeButton.disabled = false;
        analyzeButton.innerHTML = `<span class="btn-icon">🔬</span><span class="btn-text">Analyze Food & Run Regression Models</span>`;
    }
}

// ─────────────────────────────────────────────────────────────
// 4. INTERACTIVE INGREDIENT EDITING & RE-CALCULATION
// ─────────────────────────────────────────────────────────────
async function recalculateNutrition() {
    if (!window.currentIngredients || Object.keys(window.currentIngredients).length === 0) {
        alert("Please add at least one ingredient to calculate nutrition.");
        return;
    }

    const recalcBtn = document.getElementById("recalcBtn");
    if (recalcBtn) {
        recalcBtn.disabled = true;
        recalcBtn.innerHTML = `<span>⏳</span> Recalculating...`;
    }

    try {
        const nutritionUrl = window.NUTRITION_API_URL || "http://127.0.0.1:8000/predict/vision";
        const mlPayload = {
            dish: window.currentDish || null,
            ingredients: window.currentIngredients,
        };

        const resp = await fetch(nutritionUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(mlPayload),
        });

        const mlData = await resp.json();
        if (resp.ok && mlData.nutrition) {
            window.currentAnalysisData.nutrition = mlData.nutrition;
            window.currentAnalysisData.recipe_totals = mlData.recipe_totals;
            renderAnalysisResults(window.currentAnalysisData);
        } else {
            alert("Recalculation error: " + (mlData.detail || "Server error"));
        }
    } catch (err) {
        console.error("Recalculation error:", err);
    }
}

function adjustWeight(name, delta) {
    if (!window.currentIngredients[name]) return;
    const current = window.currentIngredients[name];
    const updated = Math.max(10, current + delta);
    window.currentIngredients[name] = updated;

    const input = document.getElementById(`weight_input_${name}`);
    if (input) input.value = updated;
}

function updateWeightManual(name, value) {
    const val = parseFloat(value);
    if (!isNaN(val) && val > 0) {
        window.currentIngredients[name] = Math.round(val);
    }
}

function removeIngredient(name) {
    delete window.currentIngredients[name];
    renderAnalysisResults(window.currentAnalysisData);
}

function addCustomIngredient() {
    const nameInput = document.getElementById("newIngrName");
    const weightInput = document.getElementById("newIngrWeight");
    if (!nameInput || !weightInput) return;

    const name = nameInput.value.trim();
    const weight = parseFloat(weightInput.value);

    if (!name) {
        alert("Please enter an ingredient name.");
        return;
    }
    if (isNaN(weight) || weight <= 0) {
        alert("Please enter a valid weight in grams.");
        return;
    }

    window.currentIngredients[name] = Math.round(weight);
    nameInput.value = "";
    weightInput.value = "100";
    recalculateNutrition();
}

// ─────────────────────────────────────────────────────────────
// 5. RENDER COMPLETE ANALYSIS DOM
// ─────────────────────────────────────────────────────────────
function renderAnalysisResults(data) {
    const dishName = escapeHtml(data.dish || "Detected Plated Meal");
    const scale = data.scale_reference || {};
    const scaleDetected = scale.detected ? "✅ Rim Calibrated" : "⚠️ Estimated (Uncalibrated)";
    const footprintText = scale.food_footprint_cm2 ? ` &bull; Footprint: ${scale.food_footprint_cm2} cm²` : "";
    const plateText = `🍽️ ${scaleDetected} (${scale.plate_diameter_cm || 25} cm Plate${footprintText})`;

    const totals = data.recipe_totals || {
        total_weight_g: 0, energy_kcal: 0, carb_g: 0, protein_g: 0, fat_g: 0, fibre_g: 0
    };

    // Macro Split calculation (Carb: 4 kcal/g, Prot: 4 kcal/g, Fat: 9 kcal/g)
    const carbKcal = Math.max(0, totals.carb_g * 4);
    const protKcal = Math.max(0, totals.protein_g * 4);
    const fatKcal  = Math.max(0, totals.fat_g * 9);
    const sumMacroKcal = (carbKcal + protKcal + fatKcal) || 1;

    const carbPct = Math.round((carbKcal / sumMacroKcal) * 100);
    const protPct = Math.round((protKcal / sumMacroKcal) * 100);
    const fatPct  = Math.max(0, 100 - carbPct - protPct);

    // 1. Totals Banner
    const totalsHtml = `
        <div class="totals-summary-banner">
            <div class="total-stat-box">
                <div class="stat-label">Total Weight</div>
                <div class="stat-value">${totals.total_weight_g} g</div>
            </div>
            <div class="total-stat-box">
                <div class="stat-label">Calories</div>
                <div class="stat-value">${totals.energy_kcal} kcal</div>
            </div>
            <div class="total-stat-box">
                <div class="stat-label">Carbs</div>
                <div class="stat-value">${totals.carb_g} g</div>
            </div>
            <div class="total-stat-box">
                <div class="stat-label">Protein</div>
                <div class="stat-value">${totals.protein_g} g</div>
            </div>
            <div class="total-stat-box">
                <div class="stat-label">Fats</div>
                <div class="stat-value">${totals.fat_g} g</div>
            </div>
            <div class="total-stat-box">
                <div class="stat-label">Fibre</div>
                <div class="stat-value">${totals.fibre_g} g</div>
            </div>
        </div>
    `;

    // 2. Macro Split Bar
    const macroSplitHtml = `
        <div class="macro-dist-box">
            <div class="macro-dist-header">
                <span>⚡ Caloric Macro Distribution</span>
                <span>${totals.energy_kcal} kcal total</span>
            </div>
            <div class="macro-split-bar">
                <div class="bar-seg-carbs" style="width: ${carbPct}%" title="Carbohydrates: ${carbPct}%"></div>
                <div class="bar-seg-prot" style="width: ${protPct}%" title="Protein: ${protPct}%"></div>
                <div class="bar-seg-fat" style="width: ${fatPct}%" title="Fats: ${fatPct}%"></div>
            </div>
            <div class="macro-legend">
                <div class="legend-item">
                    <span class="legend-dot" style="background: var(--accent-blue)"></span>
                    <span>Carbs: <strong>${totals.carb_g}g</strong> (${carbPct}%)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-dot" style="background: var(--primary)"></span>
                    <span>Protein: <strong>${totals.protein_g}g</strong> (${protPct}%)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-dot" style="background: var(--accent-amber)"></span>
                    <span>Fats: <strong>${totals.fat_g}g</strong> (${fatPct}%)</span>
                </div>
            </div>
        </div>
    `;

    // 3. Interactive Ingredient Tray
    let ingrRowsHtml = "";
    for (const [name, weight] of Object.entries(window.currentIngredients)) {
        const cleanName = escapeHtml(name);
        ingrRowsHtml += `
            <div class="ingr-chip-row">
                <div class="ingr-name-col">
                    <span>🍛</span>
                    <span>${cleanName}</span>
                </div>
                <div class="ingr-stepper-col">
                    <button type="button" class="btn-step" onclick="adjustWeight('${cleanName}', -10)">-</button>
                    <input type="number" id="weight_input_${cleanName}" class="input-gram" value="${weight}" min="5" step="5" onchange="updateWeightManual('${cleanName}', this.value)">
                    <span class="unit-label">g</span>
                    <button type="button" class="btn-step" onclick="adjustWeight('${cleanName}', 10)">+</button>
                    <button type="button" class="btn-remove-ingr" onclick="removeIngredient('${cleanName}')" title="Remove ingredient">✕</button>
                </div>
            </div>
        `;
    }

    const interactiveTrayHtml = `
        <div class="interactive-tray-container">
            <div class="tray-title-row">
                <h4>🎛️ Interactive Ingredient Review (Human-in-the-Loop)</h4>
                <span class="upload-hint" style="margin:0">Fine-tune weights or add components before confirming</span>
            </div>
            <div class="ingredient-chips-list">
                ${ingrRowsHtml || '<p style="color:#888;font-size:13px">No ingredients selected.</p>'}
            </div>
            <div class="tray-add-row">
                <input type="text" id="newIngrName" class="input-add-name" placeholder="Add ingredient (e.g. Papad, Roti, Ghee)...">
                <input type="number" id="newIngrWeight" class="input-add-weight" placeholder="100" value="100" min="5" step="5">
                <span class="unit-label">g</span>
                <button type="button" class="btn-add-ingr" onclick="addCustomIngredient()">+ Add</button>
            </div>
            <button type="button" id="recalcBtn" class="btn-recalc-nutrition" onclick="recalculateNutrition()">
                <span>🔄</span> Recalculate Nutrition Totals
            </button>
        </div>
    `;

    // 4. Regression Nutrition Results Table
    let regressionRows = "";
    const nutritionList = data.nutrition || [];
    nutritionList.forEach((item) => {
        regressionRows += `
            <tr>
                <td><strong>${escapeHtml(item.ingredient)}</strong></td>
                <td>${item.weight_g} g</td>
                <td><strong>${item.energy_kcal} kcal</strong></td>
                <td>${item.carb_g} g</td>
                <td>${item.protein_g} g</td>
                <td>${item.fat_g} g</td>
                <td>${item.fibre_g} g</td>
            </tr>
        `;
    });

    const regressionTableHtml = `
        <h4 class="result-section-title">📊 CatBoost Regression Nutrient Predictions</h4>
        <div class="table-responsive">
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Ingredient</th>
                        <th>Weight</th>
                        <th>Calories</th>
                        <th>Carbs</th>
                        <th>Protein</th>
                        <th>Fat</th>
                        <th>Fibre</th>
                    </tr>
                </thead>
                <tbody>
                    ${regressionRows || '<tr><td colspan="7">No nutritional regression predictions returned.</td></tr>'}
                </tbody>
            </table>
        </div>
    `;

    // 5. Vision Physical Calibration Table
    let estimatesRows = "";
    const estimates = data.estimates || [];
    if (estimates.length > 0) {
        estimates.forEach((item) => {
            const name = escapeHtml(item.name || "Item");
            const weight = item.weight_g ? item.weight_g.estimate : (window.currentIngredients[item.name] || "--");
            const vol = item.volume_ml ? `${item.volume_ml.min} - ${item.volume_ml.max}` : "--";
            const density = item.density_g_per_ml ? `${item.density_g_per_ml.min} - ${item.density_g_per_ml.max}` : "--";
            const badge = getConfidenceBadge(item.confidence);

            estimatesRows += `
                <tr>
                    <td><strong>${name}</strong></td>
                    <td><strong>${weight} g</strong></td>
                    <td>${vol} ml</td>
                    <td>${density} g/cm³</td>
                    <td>${badge}</td>
                </tr>
            `;
        });
    }

    const estimatesTableHtml = `
        <h4 class="result-section-title">⚖️ Physical Density & Volume Calibration</h4>
        <div class="table-responsive">
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Ingredient</th>
                        <th>Estimated Weight</th>
                        <th>Estimated Volume</th>
                        <th>Density Range</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    ${estimatesRows || '<tr><td colspan="5">No physical calibrations estimated.</td></tr>'}
                </tbody>
            </table>
        </div>
    `;

    // 6. Action Toolbar
    const actionToolbarHtml = `
        <div class="report-action-bar">
            <button type="button" class="btn-report" onclick="window.copyMealSummary()">
                📋 Copy Summary
            </button>
            <button type="button" class="btn-report" onclick="window.printMealReport()">
                🖨️ Print Report
            </button>
        </div>
    `;

    // Combine into final DOM
    resultDiv.innerHTML = `
        <div class="result-card-header">
            <h3>🍛 ${dishName}</h3>
            <span class="scale-badge">${plateText}</span>
        </div>
        ${totalsHtml}
        ${macroSplitHtml}
        ${interactiveTrayHtml}
        ${regressionTableHtml}
        ${estimatesTableHtml}
        ${actionToolbarHtml}
    `;

    // Update bottom nutrition grid
    updateNutritionCards(totals, data.micronutrients);
}

function updateNutritionCards(totals, micronutrients) {
    if (!totals) return;
    const micro = micronutrients || {};

    const setVal = (id, val, unit) => {
        const el = document.getElementById(id);
        if (el) el.textContent = `${val !== undefined && val !== null ? val : "--"} ${unit}`;
    };

    setVal("calories", totals.energy_kcal, "kcal");
    setVal("carbohydrates", totals.carb_g, "g");
    setVal("protein", totals.protein_g, "g");
    setVal("fats", totals.fat_g, "g");
    setVal("fibre", totals.fibre_g, "g");

    setVal("free-sugar", micro.free_sugar !== undefined ? micro.free_sugar : 0.0, "g");
    setVal("sodium", micro.sodium !== undefined ? micro.sodium : 0.0, "mg");
    setVal("calcium", micro.calcium !== undefined ? micro.calcium : 0.0, "mg");
    setVal("iron", micro.iron !== undefined ? micro.iron : 0.0, "mg");
    setVal("vitamin-c", micro.vitamin_c !== undefined ? micro.vitamin_c : 0.0, "mg");
    setVal("folate", micro.folate !== undefined ? micro.folate : 0.0, "µg");
}

window.printMealReport = function () {
    window.print();
};

window.copyMealSummary = function () {
    if (!window.currentAnalysisData) return;
    const d = window.currentAnalysisData;
    const t = d.recipe_totals || {};
    let text = `NutriScanAI Nutrition Summary:\n`;
    text += `Dish: ${d.dish || "Meal"}\n`;
    text += `Total Weight: ${t.total_weight_g}g | Calories: ${t.energy_kcal} kcal\n`;
    text += `Carbs: ${t.carb_g}g | Protein: ${t.protein_g}g | Fat: ${t.fat_g}g | Fibre: ${t.fibre_g}g\n\n`;
    text += `Ingredients:\n`;
    for (const [name, weight] of Object.entries(window.currentIngredients)) {
        text += `- ${name}: ${weight}g\n`;
    }
    navigator.clipboard.writeText(text).then(() => {
        alert("Summary copied to clipboard!");
    }).catch(err => {
        console.error("Clipboard copy failed:", err);
    });
};
