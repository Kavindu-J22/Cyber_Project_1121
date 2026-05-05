"""
Paper-style evaluation for face verification.

What this script does:
1) Reads identity folders from Dataset (default split: test)
2) Builds genuine pairs (same identity) and impostor pairs (different identities)
3) Runs embedding extraction with the existing FaceVerificationEngine
4) Computes metrics requested for classification + verification papers:
   - Accuracy, Precision, Recall, Specificity, Confusion Matrix
   - FAR, FRR, EER
5) Saves CSV/JSON/plots and a human-readable markdown summary

Example:
    python paper_metrics_evaluator.py \
        --dataset-root "Dataset/dataset" \
        --split test \
        --output-dir evaluation_reports/paper_metrics
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
)

from src.config_loader import load_config
from src.face_verification import FaceVerificationEngine


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class PairSample:
    image_a: Path
    image_b: Path
    label: int  # 1 genuine, 0 impostor


@dataclass
class PairScore:
    image_a: str
    image_b: str
    label: int
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate face verification with paper metrics (FAR/FRR/EER + classification metrics)."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="Dataset/dataset",
        help="Root folder containing train/val/test identity folders",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold override. If omitted, uses config threshold.",
    )
    parser.add_argument(
        "--genuine-pairs-per-identity",
        type=int,
        default=8,
        help="Max genuine pairs sampled per identity",
    )
    parser.add_argument(
        "--impostor-ratio",
        type=float,
        default=1.0,
        help="impostor_pairs = genuine_pairs * impostor_ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_reports/paper_metrics",
        help="Directory for reports and plots",
    )
    return parser.parse_args()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def list_identities(split_dir: Path) -> Dict[str, List[Path]]:
    identities: Dict[str, List[Path]] = {}

    for person_dir in sorted(split_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        images = [
            p
            for p in sorted(person_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if images:
            identities[person_dir.name] = images

    return identities


def sample_genuine_pairs(
    identities: Dict[str, List[Path]],
    max_pairs_per_identity: int,
    rng: random.Random,
) -> List[PairSample]:
    pairs: List[PairSample] = []

    for person_id, imgs in identities.items():
        if len(imgs) < 2:
            continue

        all_combinations = list(itertools.combinations(imgs, 2))
        rng.shuffle(all_combinations)
        selected = all_combinations[: min(max_pairs_per_identity, len(all_combinations))]

        for img_a, img_b in selected:
            pairs.append(PairSample(image_a=img_a, image_b=img_b, label=1))

        if len(selected) == 0:
            print(f"Warning: No genuine pairs sampled for identity {person_id}")

    return pairs


def sample_impostor_pairs(
    identities: Dict[str, List[Path]],
    target_count: int,
    rng: random.Random,
) -> List[PairSample]:
    pairs: List[PairSample] = []

    valid_ids = [pid for pid, imgs in identities.items() if len(imgs) > 0]
    if len(valid_ids) < 2:
        return pairs

    max_attempts = max(2000, target_count * 20)
    attempts = 0
    seen_keys = set()

    while len(pairs) < target_count and attempts < max_attempts:
        attempts += 1

        id_a, id_b = rng.sample(valid_ids, 2)
        img_a = rng.choice(identities[id_a])
        img_b = rng.choice(identities[id_b])

        # Canonical key avoids accidental duplicates regardless of order.
        key = tuple(sorted([str(img_a), str(img_b)]))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        pairs.append(PairSample(image_a=img_a, image_b=img_b, label=0))

    if len(pairs) < target_count:
        print(
            "Warning: Could not sample enough impostor pairs. "
            f"Requested={target_count}, got={len(pairs)}"
        )

    return pairs


def build_pair_set(
    identities: Dict[str, List[Path]],
    genuine_pairs_per_identity: int,
    impostor_ratio: float,
    rng: random.Random,
) -> List[PairSample]:
    genuine_pairs = sample_genuine_pairs(
        identities=identities,
        max_pairs_per_identity=genuine_pairs_per_identity,
        rng=rng,
    )

    impostor_target = int(math.ceil(len(genuine_pairs) * impostor_ratio))
    impostor_pairs = sample_impostor_pairs(
        identities=identities,
        target_count=impostor_target,
        rng=rng,
    )

    all_pairs = genuine_pairs + impostor_pairs
    rng.shuffle(all_pairs)
    return all_pairs


def score_pairs(engine: FaceVerificationEngine, pairs: Sequence[PairSample]) -> List[PairScore]:
    cache: Dict[str, np.ndarray] = {}
    results: List[PairScore] = []

    for idx, pair in enumerate(pairs, start=1):
        key_a = str(pair.image_a.resolve())
        key_b = str(pair.image_b.resolve())

        if key_a not in cache:
            cache[key_a] = engine.extract_embedding(pair.image_a, return_numpy=True)
        if key_b not in cache:
            cache[key_b] = engine.extract_embedding(pair.image_b, return_numpy=True)

        score = cosine_similarity(cache[key_a], cache[key_b])
        results.append(
            PairScore(
                image_a=key_a,
                image_b=key_b,
                label=pair.label,
                score=score,
            )
        )

        if idx % 100 == 0:
            print(f"Scored {idx}/{len(pairs)} pairs...")

    return results


def compute_eer_and_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr

    i = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[i] + fnr[i]) / 2.0)
    eer_threshold = float(thresholds[i])
    return eer, eer_threshold


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> Dict[str, object]:
    y_pred = (y_scores >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    frr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    eer, eer_threshold = compute_eer_and_threshold(y_true, y_scores)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "far": far,
        "frr": frr,
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix_2x2": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
    }


def save_plots(y_true: np.ndarray, y_scores: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # Confusion matrix heatmap
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    # FAR/FRR vs threshold curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, fpr, label="FAR (FPR)", linewidth=2)
    plt.plot(thresholds, fnr, label="FRR (FNR)", linewidth=2)
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("FAR/FRR vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "far_frr_curve.png", dpi=160)
    plt.close()

    # Score distributions
    genuine_scores = y_scores[y_true == 1]
    impostor_scores = y_scores[y_true == 0]
    plt.figure(figsize=(9, 6))
    sns.kdeplot(genuine_scores, fill=True, alpha=0.4, label="Genuine (1)")
    sns.kdeplot(impostor_scores, fill=True, alpha=0.4, label="Impostor (0)")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title("Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=160)
    plt.close()


def write_markdown_summary(
    output_file: Path,
    metrics: Dict[str, object],
    split_dir: Path,
    num_identities: int,
    num_pairs: int,
    num_genuine: int,
    num_impostor: int,
) -> None:
    lines = [
        "# Face Verification Metrics Report",
        "",
        "## Evaluated Data",
        f"- Split folder: {split_dir}",
        f"- Number of identities: {num_identities}",
        f"- Number of pairs: {num_pairs}",
        f"- Genuine pairs (label=1): {num_genuine}",
        f"- Impostor pairs (label=0): {num_impostor}",
        "",
        "## Metric Definitions",
        "- Accuracy: Overall fraction of correct decisions.",
        "- Precision: Among predicted matches, how many are true matches.",
        "- Recall (TPR): Among true matches, how many were correctly accepted.",
        "- Specificity (TNR): Among true non-matches, how many were correctly rejected.",
        "- Confusion Matrix: Counts of TN, FP, FN, TP.",
        "- FAR (False Acceptance Rate): FP / (FP + TN). Lower is better for security.",
        "- FRR (False Rejection Rate): FN / (FN + TP). Lower is better for usability.",
        "- EER (Equal Error Rate): Error rate where FAR and FRR are equal (or closest). Lower is better.",
        "",
        "## Results",
        f"- Threshold: {metrics['threshold']:.6f}",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        f"- Precision: {metrics['precision']:.6f}",
        f"- Recall: {metrics['recall']:.6f}",
        f"- Specificity: {metrics['specificity']:.6f}",
        f"- FAR: {metrics['far']:.6f}",
        f"- FRR: {metrics['frr']:.6f}",
        f"- EER: {metrics['eer']:.6f}",
        f"- EER threshold: {metrics['eer_threshold']:.6f}",
        "",
        "## Confusion Matrix",
        f"- TN: {metrics['confusion_matrix']['tn']}",
        f"- FP: {metrics['confusion_matrix']['fp']}",
        f"- FN: {metrics['confusion_matrix']['fn']}",
        f"- TP: {metrics['confusion_matrix']['tp']}",
    ]

    output_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    dataset_root = Path(args.dataset_root).resolve()
    split_dir = dataset_root / args.split

    if not split_dir.exists() or not split_dir.is_dir():
        raise FileNotFoundError(f"Split folder does not exist: {split_dir}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    threshold = float(args.threshold) if args.threshold is not None else float(
        config.get("verification.threshold", 0.8096)
    )

    identities = list_identities(split_dir)
    if len(identities) < 2:
        raise ValueError("Need at least two identities to compute impostor pairs")

    pairs = build_pair_set(
        identities=identities,
        genuine_pairs_per_identity=max(1, int(args.genuine_pairs_per_identity)),
        impostor_ratio=max(0.1, float(args.impostor_ratio)),
        rng=rng,
    )
    if len(pairs) == 0:
        raise ValueError("No pairs were generated from the dataset")

    num_genuine = sum(1 for p in pairs if p.label == 1)
    num_impostor = sum(1 for p in pairs if p.label == 0)

    print("=== Pair Sampling Summary ===")
    print(f"Split:      {split_dir}")
    print(f"Identities: {len(identities)}")
    print(f"Pairs:      {len(pairs)}")
    print(f"Genuine:    {num_genuine}")
    print(f"Impostor:   {num_impostor}")

    engine = FaceVerificationEngine(config)
    scored_pairs = score_pairs(engine, pairs)

    y_true = np.array([x.label for x in scored_pairs], dtype=np.int32)
    y_scores = np.array([x.score for x in scored_pairs], dtype=np.float32)
    y_pred = (y_scores >= threshold).astype(np.int32)

    metrics = compute_metrics(y_true=y_true, y_scores=y_scores, threshold=threshold)
    metrics["num_pairs"] = int(len(scored_pairs))
    metrics["num_identities"] = int(len(identities))
    metrics["num_genuine"] = int(num_genuine)
    metrics["num_impostor"] = int(num_impostor)

    # Save per-pair details for reproducibility.
    rows = [
        {
            "image_a": x.image_a,
            "image_b": x.image_b,
            "label": int(x.label),
            "score": float(x.score),
            "prediction": int(x.score >= threshold),
        }
        for x in scored_pairs
    ]
    pd.DataFrame(rows).to_csv(output_dir / "pair_scores.csv", index=False)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_plots(y_true=y_true, y_scores=y_scores, y_pred=y_pred, output_dir=output_dir)
    write_markdown_summary(
        output_file=output_dir / "metrics_report.md",
        metrics=metrics,
        split_dir=split_dir,
        num_identities=len(identities),
        num_pairs=len(scored_pairs),
        num_genuine=num_genuine,
        num_impostor=num_impostor,
    )

    print("\n=== Final Metrics ===")
    print(f"Threshold:   {metrics['threshold']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"FAR:         {metrics['far']:.4f}")
    print(f"FRR:         {metrics['frr']:.4f}")
    print(f"EER:         {metrics['eer']:.4f}")
    print(f"EER thr:     {metrics['eer_threshold']:.4f}")
    print(f"\nSaved reports to: {output_dir}")


if __name__ == "__main__":
    main()
