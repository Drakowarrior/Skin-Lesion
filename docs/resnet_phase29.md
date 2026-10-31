# Phase 29 — Production React Frontend Report

**Project:** Skin Lesion Classification  
**Phase:** 29 — Production React Frontend Implementation  
**Status:** COMPLETE  
**Production Checkpoint:** `models/production/resnet50_skin_lesion_v1.keras`  

> **EXPLICIT STATEMENT:**  
> "No model training or model modification was performed during Phase 29."

---

## 1. Objective

Phase 29 transforms the skin lesion classification project from backend REST API engineering into a modern, user-facing AI research application. A production-quality React frontend was constructed to provide an intuitive dashboard experience for image upload, client validation, inference execution, multi-class prediction display, and model metadata inspection.

---

## 2. Architecture & Design Pattern

The application is structured around a modular component architecture with clear separation of concerns:

```
App.jsx
├── Navbar.jsx (Status Indicator, Navigation & Mobile Drawer)
├── Hero.jsx (Title, Description, CTA, Micro Statistics Grid)
├── AnalysisWorkspace.jsx (Central State-Driven Workspace)
│   ├── ImageUploader.jsx (Drag & Drop + File Picker)
│   ├── ImagePreview.jsx (Image Display, Metadata & Actions)
│   ├── LoadingState.jsx (Multi-step Indeterminate Progress Tracker)
│   ├── PredictionCard.jsx (Primary Predicted Class & Confidence Bar)
│   ├── TopPredictions.jsx (Ranked Top-3 Candidate Classifications)
│   ├── ExplainabilityPlaceholder.jsx (Phase 30 Grad-CAM Reserved Area)
│   └── ErrorState.jsx (User-friendly Rejection & Error Card)
├── HowItWorks.jsx (3-step Technical Pipeline Breakdown)
├── ModelInfo.jsx (Dynamic Model Specs from GET /api/v1/model)
├── Disclaimer.jsx (Prominent Non-Diagnostic Safety Banner)
└── Footer.jsx (Footer Branding & Quick Links)
```

* **Service Layer (`src/services/api.js`):** Centralizes all HTTP communication with the Phase 28 FastAPI backend (`VITE_API_BASE_URL`).
* **Hook Layer (`src/hooks/usePrediction.js`):** Encapsulates upload handling, object URL lifecycle management, image dimension parsing, API request dispatching, and state management.

---

## 3. Technology Stack

* **Framework:** React 18
* **Build Tool:** Vite 8.3
* **Language:** JavaScript (ES6+ / JSX)
* **Styling:** Vanilla CSS with custom CSS Custom Properties (`styles/variables.css`), CSS Reset (`styles/globals.css`), and Component Styles (`styles/components.css`)
* **Icons:** `lucide-react`
* **Forbidden Technologies:** TypeScript and Tailwind CSS were explicitly avoided in strict accordance with project directives.

---

## 4. UI Design & Aesthetics

* **Color Palette:** Dark-first medical AI theme featuring deep slate backgrounds (`#070a12`), dark glass surface cards (`#0f172a`), electric cyan accents (`#06b6d4`), and royal purple highlights (`#6366f1`).
* **Typography:** `Inter` for clean body and headings; `JetBrains Mono` for numeric metrics and class indices.
* **Visual Hierarchy:** Directs user focus from the Hero headline to the central Analysis Workspace, leading into structured prediction results and safety notices.

---

## 5. Upload & Analysis Flow

1. **Initial Dropzone State:** Renders an intuitive drag-and-drop dropzone supporting click-to-browse file pickers and format pills (`JPG`, `PNG`, `WEBP`, `Max 10MB`).
2. **Client Validation:** Verifies file extension, MIME type, and size limits before sending payloads across the network.
3. **Image Preview:** Displays thumbnail, file size (e.g. `2.4 MB`), dimensions (e.g. `2240 × 1680 px`), `Remove` button, and explicit `Analyze Image` button. No automatic background uploads.
4. **Indeterminate Loading:** Triggers `LoadingState` with multi-step progress trackers ("Preparing image data" → "Running ResNet50 inference" → "Computing top-3 predictions").
5. **Results Presentation:** Displays the main predicted class, confidence bar, top-3 candidates, and "Analyze Another Image" reset button.

---

## 6. API Integration

Exclusively consumes the Phase 28 FastAPI backend contract:
* `GET /api/v1/health` → Sets `AI Service Online` badge status.
* `GET /api/v1/model` → Dynamically populates ResNet50 architecture specs, canonical class count (`NUM_CLASSES=14`), and checkpoint status (`frozen`).
* `POST /api/v1/predict` → Sends `multipart/form-data` with `image: File` payload, rendering the exact backend response structure.

---

## 7. Error Handling

Converts raw HTTP status codes into human-readable user messages while hiding Python tracebacks, stack traces, or internal server paths:
* `HTTP 415` → "The uploaded image format isn't supported. Please use JPG, PNG, or WEBP."
* `HTTP 413` → "The selected image file is too large. Please upload an image under 10MB."
* `HTTP 400` → "We couldn't read this image. Please choose another file."
* `Connection Failure` → "Unable to connect to the AI server. Please verify the FastAPI backend is running."

---

## 8. Responsive Design & Accessibility

* **Breakpoints Tested:** Mobile (`320px`, `375px`), Tablet (`768px`), Desktop (`1024px`, `1280px`, `1440px+`).
* **Accessibility:** Semantic HTML5 elements (`<nav>`, `<main>`, `<section>`, `<header>`, `<footer>`), keyboard navigation (`tabIndex`, `onKeyDown`), visible focus rings, high contrast text ratios, and `@media (prefers-reduced-motion: reduce)` support.

---

## 9. Security & Privacy

* Client environment variables use `VITE_*` prefix and store zero backend secrets or API keys.
* Uploaded images are kept transiently in memory and transmitted directly to the configured FastAPI server without third-party tracking.

---

## 10. Performance & Build Validation

* **Production Bundle Build:**
  ```bash
  npm run build
  ```
  * Build Status: **PASSED** (Built in 167 ms)
  * Transformation: 1904 modules transformed cleanly
  * Errors / ESLint Warnings: **0**

---

## 11. Future Grad-CAM Integration (Phase 30)

`ExplainabilityPlaceholder.jsx` provides a dedicated reserved visual container ("Visual Attention Heatmap (Grad-CAM)") marked as `Phase 30 • Reserved`. This allows Phase 30 to seamlessly insert feature map attention visualizations without rearchitecting the dashboard.

---

## 12. Next Phase

* **Phase 30:** Grad-CAM Explainability Implementation (generating spatial attention heatmaps over lesion images).
