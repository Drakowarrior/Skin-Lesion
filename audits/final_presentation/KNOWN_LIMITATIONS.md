# Known Scientific, Dataset & Operational Limitations

## 1. Scientific & Dataset Limitations (Mandatory Faculty Disclosure)

1. **Investigational Academic Prototype Only:**
   The software is an academic computer-vision research project designed to explore deep learning interpretability, uncertainty handling, and multimodal evidence retrieval. It is **not** a cleared medical device or diagnostic instrument.

2. **Absence of Patient-Level / Lesion-Level Identifiers:**
   The training and evaluation splits were derived from public benchmark datasets (such as HAM10000). The underlying manifests lack patient-level and lesion-level grouping identifiers. As a result, cross-split patient or lesion leakage cannot be completely ruled out.

3. **Re-used Historical Test Split ($N = 5,529$):**
   The historical test split was referenced during prior development phases and does not represent a virgin external clinical validation cohort.

4. **Closed-Set Taxonomy Assumption (14 Classes):**
   The deep learning model operates within a closed 14-class universe. Skin conditions outside these 14 categories cannot be classified and will map to the closest in-distribution vector.

5. **Grad-CAM Interpretability Boundary:**
   Grad-CAM visualizes gradient activations in layer `features.7`. It highlights regions that drove model activation, but it does **not** generate true anatomical margins, tumor depths, or histopathological borders.

---

## 2. Operational Boundaries

1. **Server Daemon Environment Reload:**
   Runtime configuration changes made to `backend/.env` (such as metrics access tokens and production CORS rules) require a standard operator service restart to reflect on the live network port.

2. **Clean-Machine Bare-Metal Platforms:**
   The system is fully verified and reproducible on macOS arm64 (Apple Silicon). Automated container recipes (`Dockerfile`) are provided, but bare-metal installations on Linux x86_64 and Windows have not been independently tested in this phase.

