# Browser & Visual Rendering Verification Report

## 1. Verified Viewports & Layout Adaptation

| Viewport | Device Profile | Visual Findings & CSS Verification | Status |
| :--- | :--- | :--- | :---: |
| **Desktop ($1512 \times 800$)** | MacBook Pro Display | 2-column results layout (`results-image-col` and `results-main-col`). Proper card padding, sharp typography, zero horizontal scrolling (`overflow-x: hidden`). | **PASS** |
| **Tablet ($768 \times 1024$)** | iPad / Tablet Profile | Container wraps cleanly at 720px max-width. Cards and action buttons scale proportionately. | **PASS** |
| **Mobile ($375 \times 812$)** | iPhone X / 12 Mini | Responsive media query `@media (max-width: 768px)` collapses 2-column grid into single vertical column. Buttons expand to full width. | **PASS** |
| **Mobile ($390 \times 844$)** | iPhone 13 / 14 | Fluid vertical stacking. Text wraps without clipping. Touch targets exceed $44\text{px}$. | **PASS** |

---

## 2. Grad-CAM Visual Explainability Verification

### Opacity Slider Range & Visual Behavior:
1. **0% Opacity (`opacity: 0`):**
   - The Grad-CAM heatmap overlay is completely invisible.
   - The raw clinical skin lesion photograph is 100% visible beneath without chromatic tint or image shift.
2. **40% Opacity (`opacity: 0.40` — Default):**
   - Balanced alpha-blending. Warm activation colors (reds and yellows) highlight salient regions (e.g. pigment network, lesion border) while the underlying anatomical features remain clearly identifiable.
3. **100% Opacity (`opacity: 1.00`):**
   - Full neural network feature salience map extracted from layer `features.7`. Zero blank panels, clipping, or layer dropouts.
4. **Continuous Slider Response:**
   - Dynamic React state `overlayOpacity` binds directly to CSS inline opacity on `<img className="heatmap-overlay-layer" />`, updating in real-time.

### Layer Co-location & Spatial Alignment:
- Container `.layered-image-wrapper` enforces strict dimensions (`max-width: 380px; max-height: 380px;`).
- Both `.base-layer` (original image) and `.heatmap-overlay-layer` (overlay) share `position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain;`.
- Because both layers share identical container bounds and `object-fit: contain`, there is zero spatial drift across square, landscape, or portrait images.

---

## 3. Workflow State Reset Verification

- Uploading or selecting a new image immediately clears previous predictions, Grad-CAM overlays, error states, and guidance data via `usePrediction.js`.
- Error states display clear retry and reset action buttons.

