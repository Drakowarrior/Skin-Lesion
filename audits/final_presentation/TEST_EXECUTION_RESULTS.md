# Test Execution & Verification Results

## 1. Environment & Runtime Context

- **Operating System:** macOS Darwin arm64 (Apple Silicon)
- **Python Environment:** Python 3.11.15 in `.venv`
- **Node.js Environment:** Node v20+, npm v10+
- **Frontend Framework:** React 19 + Vite 8.3.2

---

## 2. Test Execution Register

| Test Domain | Exact Command / Procedure | Tests / Checks | Status | Observed Output & Evidence |
| :--- | :--- | :---: | :---: | :--- |
| **Model Checkpoint Hashes** | `shasum -a 256 models/production/*` | 2 checkpoints | **PASS** | `convnext_small_v1.pt`: `67f4d8...fe3`<br>`resnet50_v1.keras`: `93a121...d58e`<br>(Bit-for-bit match). |
| **Backend Unit Test Suite** | `KERAS_HOME=/tmp/.keras MPLCONFIGDIR=/tmp/.matplotlib .venv/bin/python -m unittest discover backend` | 23 tests | **PASS** | `Ran 23 tests in 15.247s: OK`<br>Validates Grad-CAM, uncertainty, quality gate, API routes. |
| **Security Hardening Suite** | `.venv/bin/python scripts/test_phase22_security.py` | 5 checks | **PASS** | `5/5 tests passed`<br>CORS fail-safe, 401 WWW-Authenticate, 403 token check, 200 metrics export. |
| **Direct ConvNeXt Inference** | `ConvNeXtLesionPredictor().predict('backend/test_sample.jpg')` | 1 inference | **PASS** | Class: `Monkeypox`, Confidence: `0.4902`, Prob sum: `1.0000`, Top-3 sorted. |
| **Direct ConvNeXt Grad-CAM** | `ConvNeXtGradCAMGenerator().generate_explanation('backend/test_sample.jpg')` | 1 generation | **PASS** | Target layer: `features.7`, Shape: $12 \times 12$ activation grid, Data URL overlays generated. |
| **Evidence Retrieval & Fallback** | `retrieve_and_verify_sources()` on Melanoma | 1 query | **PASS** | Retrieved landmark NCI guideline (PMID 26389389). Fallback generates grounded inline citations. |
| **Frontend Static Code Analysis**| `cd frontend && npx oxlint` | 26 source files | **PASS** | **0 errors**, 15 non-blocking warnings. |
| **Frontend Production Build** | `cd frontend && npm run build` | 1 build | **PASS** | Vite built production bundle in **220 ms** (`dist/index.html` 0.95 kB, `dist/assets/index.js` 393.7 kB). |

