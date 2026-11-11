// FrontEnd/script.js - NutriScanAI Dual-View Controller & Interactive Engine

const topImageInput = document.getElementById("topImage");
const sideImageInput = document.getElementById("sideImage");
const topCameraInput = document.getElementById("topCameraInput");
const sideCameraInput = document.getElementById("sideCameraInput");
const topPreview = document.getElementById("topPreview");
const sidePreview = document.getElementById("sidePreview");
const topPlaceholder = document.getElementById("topPlaceholder");
const sidePlaceholder = document.getElementById("sidePlaceholder");
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
// 1. FILE UPLOAD & PREVIEW LOGIC (Desktop & Mobile Camera)
// ─────────────────────────────────────────────────────────────
function setPreviewImage(file, previewEl, placeholderEl) {
    if (!file) return;
    const url = URL.createObjectURL(file);
    previewEl.src = url;
    previewEl.style.display = "block";
    if (placeholderEl) placeholderEl.style.display = "none";
    const previewContainer = previewEl.closest(".upload-preview, .preview-area");
    if (previewContainer) previewContainer.classList.add("has-image");
}

if (topImageInput) {
    topImageInput.addEventListener("change", function () {
        if (topImageInput.files.length > 0) {
            setPreviewImage(topImageInput.files[0], topPreview, topPlaceholder);
        }
    });
}

if (topCameraInput) {
    topCameraInput.addEventListener("change", function () {
        if (topCameraInput.files.length > 0) {
            setPreviewImage(topCameraInput.files[0], topPreview, topPlaceholder);
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

if (sideCameraInput) {
    sideCameraInput.addEventListener("change", function () {
        if (sideCameraInput.files.length > 0) {
            setPreviewImage(sideCameraInput.files[0], sidePreview, sidePlaceholder);
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
// 2. OPTIONAL SIDE VIEW TOGGLE & CLEAR FUNCTIONS
// ─────────────────────────────────────────────────────────────
function showSideView() {
    const sidePanel = document.getElementById("sideUploadPanel");
    const addBtn = document.getElementById("addSideViewBtn");
    const grid = document.getElementById("uploadGrid");
    if (sidePanel) sidePanel.style.display = "flex";
    if (addBtn) addBtn.style.display = "none";
    if (grid) grid.classList.add("has-side-view");
}

function hideSideView() {
    const sidePanel = document.getElementById("sideUploadPanel");
    const addBtn = document.getElementById("addSideViewBtn");
    const grid = document.getElementById("uploadGrid");
    if (sidePanel) sidePanel.style.display = "none";
    if (addBtn) addBtn.style.display = "inline-flex";
    if (grid) grid.classList.remove("has-side-view");
    if (sideImageInput) sideImageInput.value = "";
    if (sideCameraInput) sideCameraInput.value = "";
    if (sidePreview) {
        sidePreview.src = "";
        sidePreview.style.display = "none";
        const sideCont = sidePreview.closest(".upload-preview, .preview-area");
        if (sideCont) sideCont.classList.remove("has-image");
    }
    if (sidePlaceholder) sidePlaceholder.style.display = "flex";
}

window.showSideView = showSideView;
window.hideSideView = hideSideView;

function clearUploads() {
    if (topImageInput) topImageInput.value = "";
    if (topCameraInput) topCameraInput.value = "";
    if (topPreview) {
        topPreview.src = "";
        topPreview.style.display = "none";
        const topCont = topPreview.closest(".upload-preview, .preview-area");
        if (topCont) topCont.classList.remove("has-image");
    }
    if (topPlaceholder) topPlaceholder.style.display = "flex";

    hideSideView();

    resultDiv.innerHTML = `
        <div class="empty-state">
            <div class="empty-state-icon">🍽️</div>
            <p class="empty-state-heading">Photos cleared</p>
            <p class="empty-state-body">Upload an overhead photo of your plate, then click Analyze.</p>
        </div>
    `;
}

function escapeHtml(text) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return String(text).replace(/[&<>"']/g, (m) => map[m]);
}

// ─────────────────────────────────────────────────────────────
// 3. MAIN DUAL-VIEW INFERENCE ORCHESTRATION
// ─────────────────────────────────────────────────────────────
async function analyzeFood() {
    const topFile = (topImageInput && topImageInput.files && topImageInput.files.length > 0)
        ? topImageInput.files[0]
        : ((topCameraInput && topCameraInput.files && topCameraInput.files.length > 0) ? topCameraInput.files[0] : null);

    if (!topFile) {
        resultDiv.innerHTML = `<div class="alert-error"><strong>Missing overhead photo.</strong> Please upload or capture an overhead photo of the plate.</div>`;
        return;
    }

    const sideFile = (sideImageInput && sideImageInput.files && sideImageInput.files.length > 0)
        ? sideImageInput.files[0]
        : ((sideCameraInput && sideCameraInput.files && sideCameraInput.files.length > 0) ? sideCameraInput.files[0] : null);

    const plateDiameter = plateDiameterInput ? parseFloat(plateDiameterInput.value) || 25.0 : 25.0;

    const formData = new FormData();
    formData.append("top_image", topFile);
    if (sideFile) formData.append("side_image", sideFile);
    formData.append("plate_diameter_cm", plateDiameter.toString());

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing…";

    let _elapsed = 0;
    const steps = [
        { label: "Validating images & calibrating plate reference scale", pct: 20 },
        { label: "Detecting plate boundary & food layout", pct: 45 },
        { label: "Identifying meal components & portion depth", pct: 75 },
        { label: "Calculating portion weights & nutritional breakdown", pct: 95 }
    ];

    // Render the stepper skeleton ONCE into the DOM
    function renderStepperSkeleton() {
        const stepListHtml = steps.map((s, i) => `
            <div class="step-item" id="stepItem${i}">
                <div class="step-bullet" id="stepBullet${i}">${i + 1}</div>
                <span>${s.label}</span>
            </div>
        `).join('');
        return `
            <div class="pipeline-progress-box">
                <div class="stepper-header">
                    <span class="stepper-title">Analyzing</span>
                    <span class="stepper-timer" id="stepperTimer">0s</span>
                </div>
                <div class="progress-bar-track">
                    <div class="progress-bar-fill" id="stepperFill" style="width: 10%"></div>
                </div>
                <div class="stepper-list">${stepListHtml}</div>
            </div>`;
    }

    // Update only the timer, fill, and step states — no full DOM replace
    function updateStepper(sec) {
        let activeIdx = 0;
        if (sec >= 3 && sec < 10) activeIdx = 1;
        else if (sec >= 10 && sec < 28) activeIdx = 2;
        else if (sec >= 28) activeIdx = 3;

        const pct = Math.min(95, Math.max(10, steps[activeIdx].pct + Math.floor((sec % 6) * 1.5)));

        const timerEl = document.getElementById('stepperTimer');
        if (timerEl) timerEl.textContent = `${sec}s`;

        const fillEl = document.getElementById('stepperFill');
        if (fillEl) fillEl.style.width = `${pct}%`;

        steps.forEach((_, idx) => {
            const itemEl = document.getElementById(`stepItem${idx}`);
            const bulletEl = document.getElementById(`stepBullet${idx}`);
            if (!itemEl || !bulletEl) return;
            itemEl.className = idx === activeIdx ? 'step-item active' : (idx < activeIdx ? 'step-item done' : 'step-item');
            bulletEl.textContent = idx < activeIdx ? '✓' : String(idx + 1);
        });
    }

    // Render skeleton once, then update in-place each second
    resultDiv.innerHTML = renderStepperSkeleton();
    const _timerInterval = setInterval(() => {
        _elapsed++;
        updateStepper(_elapsed);
    }, 1000);

    try {
        const visionUrl = window.VISION_API_URL || "http://127.0.0.1:8001/vision";
        const visionResponse = await fetch(visionUrl, { method: "POST", body: formData });
        const visionData = await visionResponse.json();

        if (!visionResponse.ok || visionData.error || visionData.detail) {
            clearInterval(_timerInterval);
            const errMsg = visionData.error || visionData.detail || `Vision server returned status ${visionResponse.status}`;
            resultDiv.innerHTML = `<div class="alert-error"><strong>Vision analysis failed.</strong> ${escapeHtml(errMsg)}</div>`;
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
            resultDiv.innerHTML = `<div class="alert-error"><strong>Nutrition calculation failed.</strong> ${escapeHtml(errMsg)}</div>`;
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
        resultDiv.innerHTML = `<div class="alert-error"><strong>Connection error:</strong> ${escapeHtml(error.message || 'Unknown error')}. Ensure the Vision Service (port 8001) and Nutrition Service (port 8000) are both running.</div>`;
    } finally {
        analyzeButton.disabled = false;
        analyzeButton.textContent = "Analyze meal";
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
        recalcBtn.textContent = "Recalculating…";
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
// 5. RENDER ANALYSIS RESULTS
// ─────────────────────────────────────────────────────────────
function renderAnalysisResults(data) {
    const dishName = escapeHtml(data.dish || "Plated Meal");
    const scale = data.scale_reference || {};
    const plateText = scale.detected
        ? `Calibrated — ${scale.plate_diameter_cm || 25} cm plate`
        : `${scale.plate_diameter_cm || 25} cm plate (estimated)`;

    const totals = data.recipe_totals || {
        total_weight_g: 0, energy_kcal: 0, carb_g: 0, protein_g: 0, fat_g: 0, fibre_g: 0
    };

    // Caloric macro proportions (Carb: 4 kcal/g, Prot: 4 kcal/g, Fat: 9 kcal/g)
    const carbKcal = Math.max(0, totals.carb_g * 4);
    const protKcal = Math.max(0, totals.protein_g * 4);
    const fatKcal  = Math.max(0, totals.fat_g * 9);
    const sumMacroKcal = (carbKcal + protKcal + fatKcal) || 1;

    const carbPct = Math.round((carbKcal / sumMacroKcal) * 100);
    const protPct = Math.round((protKcal / sumMacroKcal) * 100);
    const fatPct  = Math.max(0, 100 - carbPct - protPct);

    // 1. Data strip
    const totalsHtml = `
        <div class="result-data-strip">
            <div class="data-cell data-cell-primary">
                <div class="data-cell-label">Calories</div>
                <div class="data-cell-value">${totals.energy_kcal}<span class="data-cell-unit">kcal</span></div>
            </div>
            <div class="data-cell">
                <div class="data-cell-label">Weight</div>
                <div class="data-cell-value">${totals.total_weight_g}<span class="data-cell-unit">g</span></div>
            </div>
            <div class="data-cell">
                <div class="data-cell-label">Carbs</div>
                <div class="data-cell-value">${totals.carb_g}<span class="data-cell-unit">g</span></div>
            </div>
            <div class="data-cell">
                <div class="data-cell-label">Protein</div>
                <div class="data-cell-value">${totals.protein_g}<span class="data-cell-unit">g</span></div>
            </div>
            <div class="data-cell">
                <div class="data-cell-label">Fats</div>
                <div class="data-cell-value">${totals.fat_g}<span class="data-cell-unit">g</span></div>
            </div>
            <div class="data-cell">
                <div class="data-cell-label">Fibre</div>
                <div class="data-cell-value">${totals.fibre_g}<span class="data-cell-unit">g</span></div>
            </div>
        </div>
    `;

    // 2. Caloric macro distribution bar
    const macroSplitHtml = `
        <div class="macro-dist-box">
            <div class="macro-dist-header">
                <span>Caloric split</span>
                <span>${totals.energy_kcal} kcal total</span>
            </div>
            <div class="macro-split-bar">
                <div class="bar-seg-carbs" style="width: ${carbPct}%" title="Carbohydrates: ${carbPct}%"></div>
                <div class="bar-seg-prot" style="width: ${protPct}%" title="Protein: ${protPct}%"></div>
                <div class="bar-seg-fat" style="width: ${fatPct}%" title="Fats: ${fatPct}%"></div>
            </div>
            <div class="macro-legend">
                <div class="legend-item">
                    <span class="legend-dot" style="background: var(--carb-color)"></span>
                    <span>Carbs: ${totals.carb_g}g (${carbPct}%)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-dot" style="background: var(--protein-color)"></span>
                    <span>Protein: ${totals.protein_g}g (${protPct}%)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-dot" style="background: var(--fat-color)"></span>
                    <span>Fats: ${totals.fat_g}g (${fatPct}%)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-dot" style="background: var(--fibre-color)"></span>
                    <span>Fibre: ${totals.fibre_g}g</span>
                </div>
            </div>
        </div>
    `;

    // 3. Interactive ingredient tray
    let ingrRowsHtml = "";
    for (const [name, weight] of Object.entries(window.currentIngredients)) {
        const cleanName = escapeHtml(name);
        ingrRowsHtml += `
            <div class="ingr-chip-row">
                <div class="ingr-name-col">
                    <span class="food-emoji">🍲</span>
                    <span>${cleanName}</span>
                </div>
                <div class="ingr-stepper-col">
                    <button type="button" class="btn-step" onclick="adjustWeight('${cleanName}', -10)" title="Subtract 10g">−</button>
                    <input type="number" id="weight_input_${cleanName}" class="input-gram" value="${weight}" min="5" step="5" onchange="updateWeightManual('${cleanName}', this.value)">
                    <span class="unit-label">g</span>
                    <button type="button" class="btn-step" onclick="adjustWeight('${cleanName}', 10)" title="Add 10g">+</button>
                    <button type="button" class="btn-remove-ingr" onclick="removeIngredient('${cleanName}')" title="Remove item">✕</button>
                </div>
            </div>
        `;
    }

    const interactiveTrayHtml = `
        <div class="interactive-tray-container">
            <div class="tray-title-row">
                <h4>Meal components</h4>
                <span class="tray-subtext">Adjust portion weights before recalculating</span>
            </div>
            <div class="ingredient-chips-list">
                ${ingrRowsHtml || '<p style="color:var(--text-3);font-size:13px;padding:8px 0">No food items listed.</p>'}
            </div>
            <div class="tray-add-row">
                <input type="text" id="newIngrName" class="input-add-name" placeholder="Add food item (e.g. Roti, Dal, Salad)…">
                <input type="number" id="newIngrWeight" class="input-add-weight" placeholder="100" value="100" min="5" step="5">
                <span class="unit-label">g</span>
                <button type="button" class="btn-add-ingr" onclick="addCustomIngredient()">Add</button>
            </div>
            <button type="button" id="recalcBtn" class="btn-recalc-nutrition" onclick="recalculateNutrition()">
                Recalculate nutrition
            </button>
        </div>
    `;

    // 4. Nutrient breakdown table
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
        <p class="result-section-title">Nutrient breakdown</p>
        <div class="table-responsive">
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Food item</th>
                        <th>Portion</th>
                        <th>Calories</th>
                        <th>Carbs</th>
                        <th>Protein</th>
                        <th>Fats</th>
                        <th>Fibre</th>
                    </tr>
                </thead>
                <tbody>
                    ${regressionRows || '<tr><td colspan="7">No nutritional values calculated.</td></tr>'}
                </tbody>
            </table>
        </div>
    `;

    // 5. Action toolbar
    const actionToolbarHtml = `
        <div class="report-action-bar">
            <button type="button" class="btn-report" onclick="window.copyMealSummary()">
                Copy summary
            </button>
            <button type="button" class="btn-report" onclick="window.printMealReport()">
                Print
            </button>
            <button type="button" class="btn-report btn-report-primary" onclick="document.getElementById('analyzer').scrollIntoView({behavior:'smooth'})">
                Analyze another
            </button>
        </div>
    `;

    // 6. Medical Disclaimer Card
    const disclaimerHtml = `
        <div class="disclaimer-card">
            <span class="disclaimer-badge">Medical Disclaimer</span>
            <p>Nutritional calculations and weight estimates are automated approximations based on visual modeling and regression. They are not intended or recommended for medical, diagnostic, or clinical dietary purposes.</p>
        </div>
    `;

    // Combine into final DOM
    resultDiv.innerHTML = `
        <div class="result-card-header">
            <h3>${dishName}</h3>
            <span class="scale-badge">${plateText}</span>
        </div>
        ${totalsHtml}
        ${macroSplitHtml}
        ${interactiveTrayHtml}
        ${regressionTableHtml}
        ${actionToolbarHtml}
        ${disclaimerHtml}
    `;
}

window.printMealReport = function () {
    window.print();
};

window.copyMealSummary = function () {
    if (!window.currentAnalysisData) return;
    const d = window.currentAnalysisData;
    const t = d.recipe_totals || {};
    let text = `NutriScanAI — Meal summary\n`;
    text += `Meal: ${d.dish || "Plated Meal"}\n`;
    text += `Total Weight: ${t.total_weight_g}g | Calories: ${t.energy_kcal} kcal\n`;
    text += `Carbs: ${t.carb_g}g | Protein: ${t.protein_g}g | Fats: ${t.fat_g}g | Fibre: ${t.fibre_g}g\n\n`;
    text += `Components:\n`;
    for (const [name, weight] of Object.entries(window.currentIngredients)) {
        text += `- ${name}: ${weight}g\n`;
    }
    navigator.clipboard.writeText(text).then(() => {
        alert("Summary copied to clipboard!");
    }).catch(err => {
        console.error("Clipboard copy failed:", err);
    });
};
