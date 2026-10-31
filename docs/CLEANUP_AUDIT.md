# REPOSITORY CLEANUP & DEPENDENCY AUDIT REPORT

**Project:** Skin Lesion Classification & Explainable AI System  
**Institution:** Manipal University Jaipur — ETE Final Project  
**Date:** October 5, 2026  

---

## 1. KEEP (Production & Verification Dependencies)

### Production Model
- `models/production/resnet50_skin_lesion_v1.keras` (Frozen production weights, 207 MB)
- `models/production/model_metadata.json` (Canonical class indices, input resolution, version tag)
- `models/production/model_sha256.txt` (Target SHA-256 verification hash: `93a121c51900dc70d9e95b5a1d75e54c442fcdb4f20a1397625bcffb6dedd58e`)

### FastAPI Backend (`backend/`)
- `backend/main.py` (FastAPI app, lifespan model loader, CORS, security middleware, REST endpoints)
- `backend/config.py` (Production environment variables, relative path resolution, API metadata)
- `backend/guidance.py` (AI patient education service, 4-level triage, red-flag detection, rule fallback)
- `backend/explainability.py` (Grad-CAM service wrapper)
- `backend/test_api.py` (Integration test suite for health, model, prediction, invalid uploads)
- `backend/test_deployment.py` (Deployment restart test)
- `backend/test_explainability.py` (Grad-CAM heatmap/overlay verification test)
- `backend/test_guidance.py` (Triage, red-flag, and class-limitation test suite)
- `backend/test_qa_suite.py` (Full end-to-end QA & performance test)
- `backend/Dockerfile` (Production Docker container configuration)
- `backend/.env.example` (Environment configuration template without secrets)

### Core Model & Inference Modules (`src/`)
- `src/__init__.py` (Package declaration)
- `src/class_labels.py` (14 canonical class labels, display names, index mapping)
- `src/inference.py` (Production model prediction pipeline, preprocessing, softmax transformation)
- `src/gradcam.py` (Production Grad-CAM explainability generator, conv5_block3_out activation mapping)
- `src/resnet_phase26_train.py` (Phase 26 Champion ResNet50 training pipeline definition & reproducibility script)

### React + Vite Frontend (`frontend/`)
- `frontend/src/App.jsx` (Root layout and page tab routing)
- `frontend/src/main.jsx` (React 19 entry point)
- `frontend/src/components/` (All 15 active components: `Navbar`, `Hero`, `AboutAndHowItWorks`, `AboutModel`, `HowItWorks`, `AnalysisWorkspace`, `ImageUploader`, `ImagePreview`, `LoadingState`, `ErrorState`, `PredictionCard`, `TopPredictions`, `ExplainabilityPanel`, `PatientGuidanceSection`, `Footer`)
- `frontend/src/hooks/usePrediction.js` (State management hook)
- `frontend/src/services/api.js` (HTTP client for backend REST APIs)
- `frontend/src/styles/` (`globals.css`, `components.css`)
- `frontend/package.json` & `frontend/package-lock.json` (NPM dependencies including `react-markdown` and `lucide-react`)
- `frontend/vite.config.js` & `frontend/index.html` (Vite build configuration)

### Documentation (`docs/` & Root)
- `docs/ETE_PRESENTATION_COMPLETE_PROJECT_KNOWLEDGE.md` (Complete presentation & technical reference)
- `docs/FINAL_PROJECT_STATUS.md` (Final project summary, architecture, and verification status)
- `README.md` (Repository overview and startup guide)
- `requirements.txt` (Python backend dependencies)
- `.gitignore` (Git ignore patterns)

---

## 2. DELETE (Confirmed Unused / Historical Artifacts)

### Raw Datasets & Large Archives (18.7 GB Total)
- `archive.zip` (9.3 GB raw dataset zip, unused by runtime application)
- `data/raw/` (9.4 GB extracted raw training/validation/test images, unused by runtime application)
- `data/processed/train_old.csv`, `val_old.csv`, `test_old.csv` (Obsolete training split CSVs)

### Obsolete Model Checkpoints & Training Outputs (1.6 GB Total)
- `models/resnet50_best.keras` (Phase 16 training checkpoint)
- `models/resnet50_improved_best.keras` (Phase 17 training checkpoint)
- `models/resnet50_phase17_best.keras` (Phase 17 checkpoint)
- `models/resnet50_phase20_best.keras` (Phase 20 checkpoint)
- `models/resnet50_phase23_best.keras` (Phase 23 checkpoint)
- `models/resnet50_phase24_best.keras` (Phase 24 checkpoint)
- `models/resnet50_phase25_best.keras` (Phase 25 checkpoint)
- `models/resnet50_phase26_best.keras` (Phase 26 checkpoint)
- `models/skin_lesion_cnn.pth` (Early PyTorch prototype model file)
- `models/training_history.json` (Early prototype training log)

### Historical Development & Training Scripts (`src/` & Root)
- `app.py` & `assets/` (Early Streamlit prototype app and CSS stylesheet)
- `src/resnet_train.py`, `src/train.py`, `src/evaluate.py`, `src/model.py`, `src/data_pipeline.py`, `src/split_dataset.py`, `src/rebuild_splits.py`, `src/prepare_dataset.py`, `src/audit_dataset.py`, `src/check_gpu.py`, `src/check_tensorflow_gpu.py` (Historical training data processing scripts)
- `src/resnet_phase17_train.py`, `src/resnet_phase17_analysis.py`, `src/resnet_phase18_evaluate.py`, `src/resnet_phase19_analysis.py`, `src/resnet_phase20_train.py`, `src/resnet_phase20_evaluate.py`, `src/resnet_phase21_evaluate.py`, `src/resnet_phase22_error_analysis.py`, `src/resnet_phase23_train.py`, `src/resnet_phase23_evaluate.py`, `src/resnet_phase24_train.py`, `src/resnet_phase24_evaluate.py`, `src/resnet_phase25_train.py`, `src/resnet_phase25_evaluate.py`, `src/resnet_phase26_train.py`, `src/resnet_phase26_evaluate.py`, `src/resnet_phase27_freeze.py`, `src/resnet_error_analysis.py`, `src/resnet_improved_evaluate.py`, `src/resnet_evaluate.py`, `src/resnet_data.py`, `src/resnet_model.py`, `src/test_inference.py` (Historical experimental scripts)

### Historical Evaluation Outputs & Phase Documents (39 MB Total)
- `outputs/` (Historical confusion matrices, training histories, phase comparison JSON files)
- `docs/resnet_phase*.md` & `docs/dataset.md` (17 intermediate phase Markdown logs consolidated into ETE documentation)

### Unused Frontend Prototype Components
- `frontend/src/components/InfoPage.jsx`
- `frontend/src/components/ModelInfo.jsx`
- `frontend/src/components/Disclaimer.jsx`

### Build & Environment Caches
- `.DS_Store`
- `__pycache__` / `.pytest_cache`
- `frontend/dist/` (Rebuilt dynamically during `npm run build`)

---

## 3. REVIEW (Conservatively Retained)
- `data/processed/metadata.csv`, `train.csv`, `val.csv`, `test.csv` (Retained lightweight metadata CSVs for dataset reference)

---

**Audit Status:** Approved for Safe Removal. Production model and backend/frontend application integrity strictly preserved.
