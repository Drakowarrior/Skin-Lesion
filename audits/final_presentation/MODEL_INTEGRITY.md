# Model Checkpoint Integrity Audit

## 1. Checkpoint Verification Matrix

Both model checkpoints were verified bit-for-bit against their authoritative reference SHA-256 hashes before and after all audit operations.

| Checkpoint Name | File Path | Expected SHA-256 | Actual Verified SHA-256 | Verification Status |
| :--- | :--- | :--- | :--- | :---: |
| **Active Production Checkpoint** | `models/production/convnext_small_skin_lesion_v1.pt` | `67f4d8d04edef9389aeac77688dcc90e193b717de67f8c4c0c1ebf15483a3fe3` | `67f4d8d04edef9389aeac77688dcc90e193b717de67f8c4c0c1ebf15483a3fe3` | **MATCH (Intact)** |
| **Rollback Baseline Checkpoint** | `models/production/resnet50_skin_lesion_v1.keras` | `93a121c51900dc70d9e95b5a1d75e54c442fcdb4f20a1397625bcffb6dedd58e` | `93a121c51900dc70d9e95b5a1d75e54c442fcdb4f20a1397625bcffb6dedd58e` | **MATCH (Intact)** |

---

## 2. Model Integrity Rules Compliance

1. **No Checkpoint Modifications:** Neither checkpoint was retrained, fine-tuned, re-exported, renamed, or modified.
2. **Untouched Historical Test Set:** The historical test dataset split (`data/processed/test.csv`, $N = 5,529$) was not accessed, re-evaluated, or tuned against.
3. **Canonical 14-Class Taxonomy Preserved:** Output tensor dimensions and class mappings remain strictly identical to the verified taxonomy.

