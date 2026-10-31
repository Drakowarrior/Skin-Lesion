"""
ResNet50 Error Analysis & Prediction Quality Review (Phase 14)

Reads existing evaluation artifacts from Phase 13 and performs an in-depth,
read-only analysis of model misclassifications, Top-3 recovery mechanics,
and softmax prediction probability distributions.
"""

import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict

from src.resnet_data import CLASS_LIST, CLASS_TO_IDX


PREDICTIONS_JSON = Path("outputs/resnet50_predictions.json")
EVALUATION_JSON = Path("outputs/resnet50_evaluation_results.json")
OUTPUT_DIR = Path("outputs")


def analyze_errors():
    if not PREDICTIONS_JSON.exists():
        print(f"ERROR: Predictions file '{PREDICTIONS_JSON}' not found.")
        sys.exit(1)

    print("========================================")
    print("RESNET50 ERROR ANALYSIS")
    print("========================================")
    print(f"\nModel:\n  models/resnet50_best.keras")

    with open(PREDICTIONS_JSON, "r") as f:
        records = json.load(f)

    total_images = len(records)
    num_classes = len(CLASS_LIST)

    print(f"\nTest Images: {total_images}")
    print(f"Classes: {num_classes}")

    # 1. Separate Correct vs Incorrect Predictions
    correct_records = [r for r in records if r["top1_correct"]]
    incorrect_records = [r for r in records if not r["top1_correct"]]

    top1_correct_count = len(correct_records)
    top1_incorrect_count = len(incorrect_records)

    # 2. Top-3 Recovery & Rank Breakdown
    top3_recovery_records = [r for r in incorrect_records if r["top3_correct"]]
    top3_failed_records = [r for r in incorrect_records if not r["top3_correct"]]

    top3_recovery_count = len(top3_recovery_records)
    top3_failed_count = len(top3_failed_records)

    rank2_count = 0
    rank3_count = 0

    for r in top3_recovery_records:
        true_cls = r["true_class"]
        top3_classes = [item["class"] for item in r["top3"]]
        if len(top3_classes) >= 2 and top3_classes[1] == true_cls:
            rank2_count += 1
        elif len(top3_classes) >= 3 and top3_classes[2] == true_cls:
            rank3_count += 1

    print("\n========================================")
    print("TOP-3 RECOVERY")
    print("========================================")
    print(f"Top-1 incorrect → Top-3 correct: {top3_recovery_count} / {total_images} ({top3_recovery_count / total_images * 100:.2f}%)")
    print(f"Rank #2: {rank2_count} ({rank2_count / total_images * 100:.2f}%)")
    print(f"Rank #3: {rank3_count} ({rank3_count / total_images * 100:.2f}%)")

    # 3. Probability Analysis
    correct_probs = [r["top1"]["probability"] * 100.0 for r in correct_records]
    incorrect_probs = [r["top1"]["probability"] * 100.0 for r in incorrect_records]

    correct_mean = float(np.mean(correct_probs)) if correct_probs else 0.0
    correct_median = float(np.median(correct_probs)) if correct_probs else 0.0
    correct_std = float(np.std(correct_probs)) if correct_probs else 0.0
    correct_min = float(np.min(correct_probs)) if correct_probs else 0.0
    correct_max = float(np.max(correct_probs)) if correct_probs else 0.0

    incorrect_mean = float(np.mean(incorrect_probs)) if incorrect_probs else 0.0
    incorrect_median = float(np.median(incorrect_probs)) if incorrect_probs else 0.0
    incorrect_std = float(np.std(incorrect_probs)) if incorrect_probs else 0.0
    incorrect_min = float(np.min(incorrect_probs)) if incorrect_probs else 0.0
    incorrect_max = float(np.max(incorrect_probs)) if incorrect_probs else 0.0

    print("\n========================================")
    print("PROBABILITY ANALYSIS")
    print("========================================")
    print("Correct predictions:")
    print(f"  Mean:   {correct_mean:.2f}%")
    print(f"  Median: {correct_median:.2f}%")
    print(f"  Std:    {correct_std:.2f}%")
    print(f"  Min:    {correct_min:.2f}%")
    print(f"  Max:    {correct_max:.2f}%")

    print("\nIncorrect predictions:")
    print(f"  Mean:   {incorrect_mean:.2f}%")
    print(f"  Median: {incorrect_median:.2f}%")
    print(f"  Std:    {incorrect_std:.2f}%")
    print(f"  Min:    {incorrect_min:.2f}%")
    print(f"  Max:    {incorrect_max:.2f}%")

    # 4. Overconfident Errors & Low Probability Correct Predictions
    overconf_50 = sum(1 for p in incorrect_probs if p >= 50.0)
    overconf_70 = sum(1 for p in incorrect_probs if p >= 70.0)
    overconf_80 = sum(1 for p in incorrect_probs if p >= 80.0)
    overconf_90 = sum(1 for p in incorrect_probs if p >= 90.0)
    overconf_95 = sum(1 for p in incorrect_probs if p >= 95.0)

    print("\n========================================")
    print("OVERCONFIDENT ERRORS")
    print("========================================")
    print(f"Incorrect >= 50%: {overconf_50}")
    print(f"Incorrect >= 70%: {overconf_70}")
    print(f"Incorrect >= 80%: {overconf_80}")
    print(f"Incorrect >= 90%: {overconf_90}")
    print(f"Incorrect >= 95%: {overconf_95}")

    low_corr_50 = sum(1 for p in correct_probs if p < 50.0)
    low_corr_70 = sum(1 for p in correct_probs if p < 70.0)
    low_corr_80 = sum(1 for p in correct_probs if p < 80.0)

    print("\n========================================")
    print("LOW-PROBABILITY CORRECT PREDICTIONS")
    print("========================================")
    print(f"Correct < 50%: {low_corr_50}")
    print(f"Correct < 70%: {low_corr_70}")
    print(f"Correct < 80%: {low_corr_80}")

    # 5. Confusion Pair Counts
    confusion_counter = Counter()
    for r in incorrect_records:
        pair = (r["true_class"], r["top1"]["class"])
        confusion_counter[pair] += 1

    top_confusions = confusion_counter.most_common(15)

    print("\n========================================")
    print("TOP CONFUSIONS")
    print("========================================")
    for idx, ((true_c, pred_c), cnt) in enumerate(top_confusions, 1):
        print(f"{idx:2d}. True: {true_c:<30} → Predicted: {pred_c:<30} Count: {cnt}")

    # 6. Class-wise Error & Recovery Analysis
    class_stats = {}
    for cls in CLASS_LIST:
        cls_records = [r for r in records if r["true_class"] == cls]
        sup = len(cls_records)
        corr = sum(1 for r in cls_records if r["top1_correct"])
        inc = sup - corr
        top3_rec = sum(1 for r in cls_records if not r["top1_correct"] and r["top3_correct"])
        top3_corr = sum(1 for r in cls_records if r["top3_correct"])

        top1_acc = (corr / sup * 100.0) if sup > 0 else 0.0
        err_rate = 100.0 - top1_acc
        top3_acc = (top3_corr / sup * 100.0) if sup > 0 else 0.0

        class_stats[cls] = {
            "support": sup,
            "correct_top1": corr,
            "incorrect_top1": inc,
            "top1_accuracy": round(top1_acc, 2),
            "error_rate": round(err_rate, 2),
            "top3_recovery": top3_rec,
            "top3_correct": top3_corr,
            "top3_accuracy": round(top3_acc, 2),
        }

    # 7. Generate Visualizations
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Graphic 1: Top 15 Confusion Pairs
    pair_labels = [f"{t} → {p}" for (t, p), _ in top_confusions[::-1]]
    pair_counts = [cnt for _, cnt in top_confusions[::-1]]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(pair_labels, pair_counts, color="#2b5c8f", edgecolor="#1a385c")
    plt.xlabel("Number of Misclassified Images", fontsize=11)
    plt.title("Top 15 ResNet50 Class-to-Class Confusion Pairs", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        plt.text(w + 2, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va="center", fontsize=10)

    plt.tight_layout()
    conf_fig_path = OUTPUT_DIR / "resnet50_top_confusions.png"
    plt.savefig(conf_fig_path, dpi=300)
    plt.close()

    # Graphic 2: Top-3 Distribution Chart
    plt.figure(figsize=(9, 6))
    categories = ["Top-1 Correct", "Top-1 Wrong → Top-3 Correct", "Top-1 & Top-3 Wrong"]
    cat_counts = [top1_correct_count, top3_recovery_count, top3_failed_count]
    colors = ["#2e7d32", "#1976d2", "#c62828"]

    bars = plt.bar(categories, cat_counts, color=colors, width=0.5, edgecolor="black")
    plt.ylabel("Number of Test Images", fontsize=11)
    plt.title("ResNet50 Test Set Prediction Outcome Breakdown", fontsize=13, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, cnt in zip(bars, cat_counts):
        pct = (cnt / total_images) * 100.0
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{cnt}\n({pct:.1f}%)",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.ylim(0, max(cat_counts) + 500)
    plt.tight_layout()
    top3_fig_path = OUTPUT_DIR / "resnet50_top3_distribution.png"
    plt.savefig(top3_fig_path, dpi=300)
    plt.close()

    # Graphic 3: Probability Distribution Comparison
    plt.figure(figsize=(10, 6))
    plt.hist(correct_probs, bins=25, alpha=0.6, label=f"Correct Top-1 Predictions (N={top1_correct_count})", color="#2e7d32", density=True)
    plt.hist(incorrect_probs, bins=25, alpha=0.6, label=f"Incorrect Top-1 Predictions (N={top1_incorrect_count})", color="#c62828", density=True)
    plt.xlabel("Top-1 Softmax Probability (%)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.title("Top-1 Prediction Softmax Probability Distribution", fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    prob_fig_path = OUTPUT_DIR / "resnet50_probability_distribution.png"
    plt.savefig(prob_fig_path, dpi=300)
    plt.close()

    # 8. Save Machine-Readable Analysis JSON
    analysis_json_data = {
        "model": "models/resnet50_best.keras",
        "test_images": total_images,
        "num_classes": num_classes,
        "summary": {
            "top1_correct": top1_correct_count,
            "top1_incorrect": top1_incorrect_count,
            "top3_recovery": top3_recovery_count,
            "top3_failed": top3_failed_count,
        },
        "top3_recovery_analysis": {
            "top1_incorrect_top3_correct": top3_recovery_count,
            "rank_2_count": rank2_count,
            "rank_3_count": rank3_count,
        },
        "probability_statistics": {
            "correct_predictions": {
                "mean": round(correct_mean, 2),
                "median": round(correct_median, 2),
                "std": round(correct_std, 2),
                "min": round(correct_min, 2),
                "max": round(correct_max, 2),
            },
            "incorrect_predictions": {
                "mean": round(incorrect_mean, 2),
                "median": round(incorrect_median, 2),
                "std": round(incorrect_std, 2),
                "min": round(incorrect_min, 2),
                "max": round(incorrect_max, 2),
            },
        },
        "overconfident_errors": {
            "gte_50_pct": overconf_50,
            "gte_70_pct": overconf_70,
            "gte_80_pct": overconf_80,
            "gte_90_pct": overconf_90,
            "gte_95_pct": overconf_95,
        },
        "low_probability_correct_predictions": {
            "lt_50_pct": low_corr_50,
            "lt_70_pct": low_corr_70,
            "lt_80_pct": low_corr_80,
        },
        "top_confusion_pairs": [
            {"true_class": t, "predicted_class": p, "count": cnt}
            for (t, p), cnt in top_confusions
        ],
        "class_metrics": class_stats,
    }

    analysis_json_path = OUTPUT_DIR / "resnet50_error_analysis.json"
    with open(analysis_json_path, "w") as f:
        json.dump(analysis_json_data, f, indent=2)

    print("\n========================================")
    print("ARTIFACTS")
    print("========================================")
    print(f"{analysis_json_path}")
    print(f"{conf_fig_path}")
    print(f"{top3_fig_path}")
    print(f"{prob_fig_path}")

    print("\n========================================")
    print("STATUS: PASS")
    print("========================================")
    return 0


if __name__ == "__main__":
    sys.exit(analyze_errors())
