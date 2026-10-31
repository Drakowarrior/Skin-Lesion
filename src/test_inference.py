"""
Production Inference Smoke Test Suite (Phase 27)

Runs end-to-end validation of the production SkinLesionPredictor API
against multiple non-test training and validation images.

Verifies:
1. Structural integrity of prediction outputs.
2. Probability array normalization (sums to ~1.0).
3. Robust handling of image paths and format preprocessing.
4. Zero usage of the isolated test dataset split.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.inference import SkinLesionPredictor


TRAIN_CSV = Path("data/processed/train.csv")
VAL_CSV = Path("data/processed/val.csv")


def main():
    print("========================================")
    print("PRODUCTION INFERENCE SMOKE TEST SUITE")
    print("========================================")

    # 1. Initialize Production Predictor
    predictor = SkinLesionPredictor()
    print(f"Loaded Production Model: {predictor.model_path}")
    print(f"Model Input Dimensions: {predictor.input_shape}")
    print(f"Model Class Count: {predictor.num_classes}\n")

    # 2. Select Non-Test Samples from Train and Validation splits
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    sample_items = [
        ("Train Split", train_df.iloc[0]),
        ("Train Split", train_df.iloc[100]),
        ("Val Split", val_df.iloc[0]),
        ("Val Split", val_df.iloc[50]),
    ]

    all_passed = True

    for source_label, row in sample_items:
        img_path = Path(row["image_path"])
        true_class = row["label"]

        print(f"--- Testing Sample [{source_label}] ---")
        print(f"Image Path: {img_path}")
        print(f"True Class: {true_class}")

        if not img_path.exists():
            print(f"ERROR: Image path does not exist: {img_path}")
            all_passed = False
            continue

        try:
            res = predictor.predict(img_path)
        except Exception as e:
            print(f"ERROR: Predictor threw exception: {str(e)}")
            all_passed = False
            continue

        pred_cls = res["predicted_class"]
        pred_idx = res["predicted_index"]
        conf = res["confidence"]
        prob_sum = res["probability_sum"]
        top3 = res["top3"]

        print(f"Predicted Class: {pred_cls} (Index {pred_idx})")
        print(f"Model Confidence: {conf:.4f}")
        print(f"Probability Sum: {prob_sum:.6f}")
        print("Top 3 Predictions:")
        for t in top3:
            print(f"  - {t['class']}: {t['confidence']:.4f}")

        # Validation Checks
        check_shape = (len(res["probabilities"]) == 14)
        check_sum = abs(prob_sum - 1.0) < 1e-4
        check_finite = np.isfinite(conf) and all(np.isfinite(v) for v in res["probabilities"].values())

        if check_shape and check_sum and check_finite:
            print("Status: PASS\n")
        else:
            print("Status: FAIL\n")
            all_passed = False

    print("========================================")
    if all_passed:
        print("ALL INFERENCE SMOKE TESTS PASSED!")
    else:
        print("INFERENCE SMOKE TESTS FAILED!")
        sys.exit(1)
    print("========================================")


if __name__ == "__main__":
    main()
