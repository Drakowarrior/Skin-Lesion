# Academic Live Demonstration Runbook (5–7 Minutes)

## 1. Startup & Pre-Flight Checklist (10 Minutes Prior to Presentation)

### Terminal 1 — Start FastAPI Backend Service
```bash
cd "/Users/karangarg/Documents/coding/skin lesions"
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

**Verify Backend Health:**
Open a browser tab to `http://127.0.0.1:8001/api/v1/health`. Expected response:
```json
{
  "status": "healthy",
  "liveness": "alive",
  "readiness": "ready",
  "model": "convnext_small_v1",
  "model_version": "v1",
  "architecture": "ConvNeXt-Small",
  "model_loaded": true,
  "explainability_loaded": true
}
```

---

### Terminal 2 — Start React + Vite Frontend
```bash
cd "/Users/karangarg/Documents/coding/skin lesions/frontend"
npm run dev
```

**Verify Frontend Status:**
Open `http://localhost:5173`. Confirm that the top navigation bar displays the green **"Backend Online"** badge.

---

### Step 3: Prepare Demonstration Images
Have 2–3 clear dermoscopic skin lesion sample images ready in a desktop folder (e.g., Melanoma, Melanocytic nevi, Basal cell carcinoma). `backend/test_sample.jpg` is also available as a verified test sample.

---

## 2. Step-by-Step Presentation Script (5–7 Minutes)

### Minute 0:00 – 1:00 │ The Problem & Architectural Scope
- **Screen:** Open `http://localhost:5173` on the default **"About / How It Works"** page.
- **Narrative:**
  > "Good morning faculty members. Skin disease accounts for a massive burden in primary healthcare, yet expert dermatological access is heavily constrained. Black-box deep learning models often output predictions without explainability, risking clinical distrust. Our project, DermaVision AI, addresses this by coupling high-capacity image classification with spatial Grad-CAM explainability, pre-inference quality gates, uncertainty scoring, and peer-reviewed PubMed evidence retrieval."
- **Key Visual:** Highlight the 3-step pipeline and the 14-class clinical taxonomy pointers.

---

### Minute 1:00 – 2:30 │ Live Analysis & Calibrated Prediction
- **Screen:** Click **"Analyze Skin Lesion"** in the top navbar.
- **Action:** Drag and drop a skin lesion sample into the upload area. Click **"Analyze Image"**.
- **Narrative:**
  > "Our backend first passes the image through a quality gate to verify sharpness, resolution, and contrast. The PyTorch ConvNeXt-Small model (50.2M parameters) classifies the lesion across the 14 categories. Notice the results display: we report calibrated probability distributions, along with entropy-based uncertainty indicators and top-3 candidates, avoiding overconfident assertions."
- **Key Visual:** Point out the primary predicted class, confidence percentage, and the Top-3 candidate ranking bar.

---

### Minute 2:30 – 4:00 │ Grad-CAM Explainability & Opacity Slider
- **Screen:** Scroll down to the **Grad-CAM Visual Explainability** card.
- **Step 1 (0% Opacity):** Drag the slider to 0%.
  > "At 0% opacity, we see the unadulterated clinical photograph, allowing the clinician to evaluate lesion margins and pigmentation directly."
- **Step 2 (40% Opacity):** Move the slider to 40% (the default).
  > "At 40% opacity, we achieve balanced alpha blending. Notice how the activation hotspot aligns with the central pigment network of the lesion."
- **Step 3 (100% Opacity):** Move the slider to 100%.
  > "At 100%, we isolate the activation map extracted from layer `features.7` of our ConvNeXt architecture. As emphasized in our scientific disclaimer, Grad-CAM reveals relative neural network attention—not a histopathological excision boundary."

---

### Minute 4:00 – 5:15 │ Patient Educational Guidance & Verified Research Retrieval
- **Screen:** Scroll to the **AI Patient Guidance** and **Scientific Literature Discovery** cards.
- **Narrative:**
  > "Beyond raw classifications, our system provides patient education covering symptoms, precautions, and warning signs. Most importantly, evidence is grounded in peer-reviewed PubMed literature with verified PMIDs."
- **Action:** Point to the **Google Scholar** card.
  > "We maintain strict architectural separation between clinical evidence and external exploratory search. Google Scholar is isolated to this single external action button, allowing researchers to explore active clinical trials."

---

### Minute 5:15 – 6:30 │ Safety Hardening & Honest Limitations
- **Screen:** Scroll back to view the disclaimers and concluding notes.
- **Narrative:**
  > "To ensure operational safety, the system enforces a strict 15 MB upload defense, Prometheus observability with Bearer token authentication, production CORS fail-safe logic, and preserves a frozen ResNet50 baseline checkpoint for instant rollback. In accordance with scientific integrity, we explicitly acknowledge that our dataset splits lack patient-level grouping identifiers and the model operates under a closed 14-class assumption."
- **Closing:** Transition to faculty Q&A.

---

## 3. Presentation Troubleshooting & Safe Recovery

- **Port 8001 Conflict:**
  If port 8001 is already occupied:
  ```bash
  lsof -i :8001
  kill -TERM <PID>
  ```
  Then re-run the backend launch command.
- **If External AI Provider Fails or Network Drops:**
  The server-side guidance module automatically transitions to its internal verified clinical rules and landmark registry, outputting complete guidance and verified evidence citations [1] with zero application crash.
- **Safe Shutdown After Presentation:**
  Press `Ctrl + C` in both terminal windows.
