"""
Model evaluation script for Face Verification (ResNet50 + Triplet Loss).

This script evaluates verification quality using labeled image pairs and outputs:
- Core metrics (accuracy, precision, recall, F1, AUC, EER)
- Confusion matrix and classification report
- Score statistics for genuine vs impostor pairs
- Plots (ROC, PR curve, score distribution, confusion matrix)

Usage example:
    python evaluate_model.py \
        --pairs-csv data/pairs.csv \
        --output-dir evaluation_reports

CSV format (required columns):
    image_a,image_b,label
    Dataset/person1/img1.jpg,Dataset/person1/img2.jpg,1
    Dataset/person1/img3.jpg,Dataset/person2/img1.jpg,0

Label rules:
    1 = genuine pair (same person)
    0 = impostor pair (different people)
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from src.config_loader import load_config
from src.face_verification import FaceVerificationEngine


@dataclass
class PairResult:
    image_a: str
    image_b: str
    label: int
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Face Verification model")
    parser.add_argument(
        "--pairs-csv",
        type=str,
        required=True,
        help="CSV file with columns: image_a,image_b,label",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_reports",
        help="Directory where reports and plots are saved",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold override. If omitted, uses config value.",
    )
    return parser.parse_args()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def find_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])
    return eer, eer_threshold


def validate_pairs_df(df: pd.DataFrame) -> None:
    required = {"image_a", "image_b", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    unique_labels = set(df["label"].astype(int).unique().tolist())
    invalid = unique_labels - {0, 1}
    if invalid:
        raise ValueError(
            f"label must contain only 0/1. Found invalid values: {sorted(invalid)}"
        )


def evaluate_pairs(
    engine: FaceVerificationEngine,
    pairs_df: pd.DataFrame,
    base_dir: Path,
) -> List[PairResult]:
    cache: Dict[str, np.ndarray] = {}
    results: List[PairResult] = []

    for idx, row in pairs_df.iterrows():
        raw_a = Path(str(row["image_a"]))
        raw_b = Path(str(row["image_b"]))

        # Support both styles in CSV:
        # 1) paths relative to project working directory
        # 2) paths relative to CSV location
        path_a = raw_a if raw_a.is_absolute() else raw_a.resolve()
        path_b = raw_b if raw_b.is_absolute() else raw_b.resolve()

        if not path_a.exists():
            path_a = (base_dir / raw_a).resolve()
        if not path_b.exists():
            path_b = (base_dir / raw_b).resolve()

        label = int(row["label"])

        if not path_a.exists():
            raise FileNotFoundError(f"Image not found: {path_a}")
        if not path_b.exists():
            raise FileNotFoundError(f"Image not found: {path_b}")

        key_a = str(path_a)
        key_b = str(path_b)

        if key_a not in cache:
            cache[key_a] = engine.extract_embedding(path_a, return_numpy=True)
        if key_b not in cache:
            cache[key_b] = engine.extract_embedding(path_b, return_numpy=True)

        score = cosine_similarity(cache[key_a], cache[key_b])
        results.append(
            PairResult(
                image_a=str(path_a),
                image_b=str(path_b),
                label=label,
                score=score,
            )
        )

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(pairs_df)} pairs...")

    return results


def save_plots(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Face Verification")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Face Verification")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=160)
    plt.close()

    # Score distribution
    genuine_scores = y_scores[y_true == 1]
    impostor_scores = y_scores[y_true == 0]
    plt.figure(figsize=(9, 6))
    sns.kdeplot(genuine_scores, fill=True, label="Genuine (label=1)", alpha=0.4)
    sns.kdeplot(impostor_scores, fill=True, label="Impostor (label=0)", alpha=0.4)
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Density")
    plt.title("Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=160)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()

    pairs_csv = Path(args.pairs_csv).resolve()
    if not pairs_csv.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {pairs_csv}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base directory for resolving relative image paths in CSV
    base_dir = pairs_csv.parent

    config = load_config(args.config)
    threshold = float(args.threshold) if args.threshold is not None else float(
        config.get("verification.threshold", 0.8096)
    )

    engine = FaceVerificationEngine(config)

    pairs_df = pd.read_csv(pairs_csv)
    validate_pairs_df(pairs_df)

    print(f"Loaded {len(pairs_df)} pairs from {pairs_csv}")
    print(f"Using threshold: {threshold:.4f}")

    pair_results = evaluate_pairs(engine, pairs_df, base_dir=base_dir)

    y_true = np.array([r.label for r in pair_results], dtype=np.int32)
    y_scores = np.array([r.score for r in pair_results], dtype=np.float32)
    y_pred = (y_scores >= threshold).astype(np.int32)

    # Core metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))
    eer, eer_threshold = find_eer(y_true, y_scores)

    genuine_scores = y_scores[y_true == 1]
    impostor_scores = y_scores[y_true == 0]

    metrics = {
        "num_pairs": int(len(y_true)),
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "score_stats": {
            "genuine_mean": float(np.mean(genuine_scores)) if len(genuine_scores) else None,
            "genuine_std": float(np.std(genuine_scores)) if len(genuine_scores) else None,
            "impostor_mean": float(np.mean(impostor_scores)) if len(impostor_scores) else None,
            "impostor_std": float(np.std(impostor_scores)) if len(impostor_scores) else None,
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["impostor", "genuine"],
            output_dict=True,
            zero_division=0,
        ),
    }

    # Save detailed pair-level scores
    score_rows = [
        {
            "image_a": r.image_a,
            "image_b": r.image_b,
            "label": r.label,
            "score": r.score,
            "prediction": int(r.score >= threshold),
        }
        for r in pair_results
    ]
    pd.DataFrame(score_rows).to_csv(output_dir / "pair_scores.csv", index=False)

    # Save metrics JSON
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save plots
    save_plots(y_true, y_scores, y_pred, output_dir)

    print("\n=== Evaluation Summary ===")
    print(f"Pairs:        {metrics['num_pairs']}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1-score:     {f1:.4f}")
    print(f"ROC AUC:      {roc_auc:.4f}")
    print(f"EER:          {eer:.4f}")
    print(f"EER threshold:{eer_threshold:.4f}")
    print(f"\nSaved report files to: {output_dir}")


if __name__ == "__main__":
    main()
