# Dataset Documentation & Classification Scope Audit

## Dataset Overview

- **Dataset Name**: Skin Lesions & Viral Exanthems Combined Dataset (`skin-ds`)
- **Dataset Source**: Kaggle [`ahmedxc4/skin-ds`](https://www.kaggle.com/datasets/ahmedxc4/skin-ds), created by merging:
  1. **ISIC 2019 / HAM10000** (Dermatoscopy images of skin lesions, neoplasms, carcinomas, and vascular lesions)
  2. **MSLDv2.0** (Mpox Skin Lesion Dataset v2.0 - Clinical photography of mpox, chickenpox, cowpox, HFMD, measles, and healthy controls)
- **Total Images**: 36,656
- **Total Classes**: 14
- **Image Formats**: 100% JPEG (`.jpg`)
- **Integrity Status**: **PASS** (36,656 / 36,656 images verified readable, 0 corrupt files)

---

## Target Classes & Distribution

The raw dataset contains **14 target classes** structured into existing `train`, `val`, and `test` subdirectories.

| Class Name | Train | Val | Test | Total Images | % of Dataset |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Actinic keratoses** | 693 | 86 | 88 | **867** | 2.37% |
| **Basal cell carcinoma** | 2,658 | 332 | 333 | **3,323** | 9.07% |
| **Benign keratosis-like lesions** | 2,099 | 262 | 263 | **2,624** | 7.16% |
| **Chickenpox** | 900 | 112 | 113 | **1,125** | 3.07% |
| **Cowpox** | 792 | 99 | 99 | **990** | 2.70% |
| **Dermatofibroma** | 191 | 23 | 25 | **239** | 0.65% |
| **HFMD** (Hand, Foot & Mouth) | 1,932 | 241 | 242 | **2,415** | 6.59% |
| **Healthy** | 1,368 | 171 | 171 | **1,710** | 4.66% |
| **Measles** | 660 | 82 | 83 | **825** | 2.25% |
| **Melanocytic nevi** | 10,300 | 1,287 | 1,288 | **12,875** | 35.12% |
| **Melanoma** | 3,617 | 452 | 453 | **4,522** | 12.34% |
| **Monkeypox** | 3,408 | 426 | 426 | **4,260** | 11.62% |
| **Squamous cell carcinoma** | 502 | 62 | 64 | **628** | 1.71% |
| **Vascular lesions** | 202 | 25 | 26 | **253** | 0.69% |
| **TOTAL** | **29,322** | **3,660** | **3,674** | **36,656** | **100.0%** |

---

## Key Image Metadata & Statistics

- **Format**: 100% JPEG
- **Dimensions Summary**:
  - `1024x1024` px: 12,414 images (33.9%)
  - `600x450` px: 10,015 images (27.3%)
  - `224x224` px: 6,690 images (18.3%)
  - `1024x680` px: 1,121 images (3.1%)
  - `1024x768` px: 774 images (2.1%)
  - Total Unique Dimension Variations: 305
- **Class Imbalance**: Minimum class is *Dermatofibroma* (239 images), Maximum class is *Melanocytic nevi* (12,875 images). Class imbalance ratio is **53.87:1**.
- **Duplicates**: 979 exact SHA-256 hash match groups (991 duplicate files across folders).

---

## Known Limitations & Observations

1. **Dataset Source Mismatch**:
   - The initial baseline prototype was described around a 7-class HAM10000 setup. However, the actual dataset present in `data/raw/` is a 14-class merged dataset containing both dermatoscopy skin lesion classes (HAM10000/ISIC 2019) and viral exanthem condition classes + healthy skin controls (MSLDv2.0).
2. **Imaging Modality Heterogeneity**:
   - HAM10000 images are high-resolution dermoscopy photos taken under specialized lighting/magnification.
   - MSLDv2.0 images are standard clinical photography of patients presenting with viral rashes.
3. **Severe Class Imbalance**:
   - Melanocytic nevi makes up over 35% of the entire dataset, while Dermatofibroma and Vascular lesions account for under 1% each. Resampling/loss-weighting strategies will be required in Phase 13.
4. **Data Redundancy**:
   - 991 images share identical SHA-256 binary hashes with another image in the raw dataset.

---

## Final Classification Scope

The target scope for the production project is explicitly confirmed as a **14-Class Skin Condition Classification Problem**:

1. **Actinic keratoses**
2. **Basal cell carcinoma**
3. **Benign keratosis-like lesions**
4. **Chickenpox**
5. **Cowpox**
6. **Dermatofibroma**
7. **HFMD** (Hand, Foot, and Mouth Disease)
8. **Healthy**
9. **Measles**
10. **Melanocytic nevi**
11. **Melanoma**
12. **Monkeypox**
13. **Squamous cell carcinoma**
14. **Vascular lesions**
