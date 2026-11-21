"""利用残差统计量对单张 Cover/Stego 图像进行分类（带类型注解版本）。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal, TypedDict

import matplotlib
import numpy as np
from PIL import Image
from scipy.ndimage import correlate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ImagePair = tuple[Path, Path]
FeatureSet = Literal["H", "H+Cooc"]
ModelName = Literal["LR", "FLD", "RBF-SVM"]
ScoreMetric = Literal["acc", "auc"]
ScoreCurve = tuple[np.ndarray, np.ndarray]


class EvaluationRecord(TypedDict):
    features: FeatureSet
    dimensions: int
    model: ModelName
    acc: float
    auc: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]


class Metrics(TypedDict):
    seed: int
    pairs: int
    train_pairs: int
    test_pairs: int
    test_filenames: list[str]
    results: list[EvaluationRecord]


ROOT: Path = Path(__file__).resolve().parent
SEED: int = 2026
TEST_SIZE: float = 0.2
MAX_BIN: int = 10
CLIP: int = 3
LEVELS: int = 2 * CLIP + 1
KERNELS: tuple[np.ndarray, ...] = tuple(
    np.array(k, dtype=np.int16)
    for k in (
        [[1, -1]],
        [[1], [-1]],
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
        [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
    )
)
HIST_DIM: int = len(KERNELS) * (MAX_BIN + 1)
FEATURE_DIM: int = HIST_DIM + 4 * ((LEVELS**2 + 1) // 2)
FEATURE_SETS: dict[FeatureSet, int] = {"H": HIST_DIM, "H+Cooc": FEATURE_DIM}
MODELS: tuple[ModelName, ...] = ("LR", "FLD", "RBF-SVM")
SCORE_METRICS: tuple[ScoreMetric, ScoreMetric] = ("acc", "auc")


def list_pairs(data_dir: Path) -> list[ImagePair]:
    cover = {p.name: p for p in (data_dir / "Cover").glob("*.pgm")}
    stego = {p.name: p for p in (data_dir / "Stego").glob("*.pgm")}
    if not cover or cover.keys() != stego.keys():
        raise ValueError(
            "Cover/ and Stego/ must contain matching sets of PGM filenames"
        )
    return [(cover[name], stego[name]) for name in sorted(cover)]


def valid_residual(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """对图像做相关运算，只保留核完全覆盖的像素。"""
    support = tuple(
        slice((size - 1) // 2, -(size // 2) or None) for size in kernel.shape
    )
    return correlate(image, kernel, mode="constant")[support]


def residual_histogram(residual: np.ndarray) -> np.ndarray:
    # |r|=j 同时统计正负两种符号；将非零 bin 减半以得到符号均值。
    counts = np.bincount(np.abs(residual).ravel(), minlength=MAX_BIN + 1)
    hist = counts[: MAX_BIN + 1] / residual.size
    hist[1:] *= 0.5
    return hist


def cooccurrences(residual: np.ndarray, axis: int) -> np.ndarray:
    """统计相邻的截断残差，将 (a,b) 与 (-a,-b) 合并计数。"""
    q = np.moveaxis(np.clip(residual, -CLIP, CLIP) + CLIP, axis, 0)
    code = (LEVELS * q[:-1] + q[1:]).ravel()
    counts = np.bincount(code, minlength=LEVELS**2)
    center = LEVELS**2 // 2
    counts[:center] += counts[:center:-1]
    return counts[: center + 1] / code.size


def image_features(image: np.ndarray | Image.Image) -> np.ndarray:
    image = np.asarray(image, dtype=np.int16)
    if image.ndim != 2 or min(image.shape) < 3:
        raise ValueError("Expected a grayscale image of at least 3 × 3 pixels")
    residuals = [valid_residual(image, kernel) for kernel in KERNELS]
    return np.concatenate(
        [
            *(residual_histogram(r) for r in residuals),
            *(cooccurrences(r, axis) for r in residuals[:2] for axis in (0, 1)),
        ]
    )


def pair_features(pair: ImagePair) -> list[np.ndarray]:
    rows = []
    for path in pair:
        with Image.open(path) as image:
            if image.mode != "L":
                raise ValueError(f"Expected an 8-bit grayscale image: {path}")
            rows.append(image_features(image))
    return rows


def extract_features(pairs: Sequence[ImagePair], workers: int) -> np.ndarray:
    features = np.empty((len(pairs), 2, FEATURE_DIM))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rows in enumerate(pool.map(pair_features, pairs, chunksize=8), start=1):
            features[i - 1] = rows
            if i % 1000 == 0 or i == len(pairs):
                print(f"Extracted {i}/{len(pairs)} pairs", flush=True)
    return features


def split_pairs(n_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    train, test = train_test_split(
        np.arange(n_pairs), test_size=TEST_SIZE, random_state=SEED
    )
    return train, test


def fisher_scores(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    """拟合带正则化的 Fisher LDA，以两类均值的中心作为阈值。"""
    means = np.stack([x_train[y_train == label].mean(axis=0) for label in (0, 1)])
    centered = x_train - means[y_train]
    covariance = centered.T @ centered / (len(y_train) - 2)
    # 按平均方差缩放岭项，并为常量特征保留一个很小的下限。
    ridge = max(0.01 * np.trace(covariance) / len(covariance), 1e-8)
    covariance.flat[:: len(covariance) + 1] += ridge
    weights = np.linalg.solve(covariance, means[1] - means[0])
    return (x_test - means.mean(axis=0)) @ weights


def evaluate(
    features: np.ndarray, train: np.ndarray, test: np.ndarray
) -> tuple[list[EvaluationRecord], list[ScoreCurve]]:
    y_train = np.tile([0, 1], len(train))
    y_test = np.tile([0, 1], len(test))
    records: list[EvaluationRecord] = []
    curves: list[ScoreCurve] = []
    for feature_set, n in FEATURE_SETS.items():
        x_train = features[train, :, :n].reshape(-1, n)
        x_test = features[test, :, :n].reshape(-1, n)
        scaler = StandardScaler()
        x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)
        for name in MODELS:
            if name == "FLD":
                scores = fisher_scores(x_train, y_train, x_test)
            else:
                model = (
                    LogisticRegression(max_iter=2000)
                    if name == "LR"
                    else SVC(C=4, gamma="scale", cache_size=512)
                )
                scores = model.fit(x_train, y_train).decision_function(x_test)
            predicted = (scores > 0).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, predicted, average="binary", zero_division=0
            )
            result: EvaluationRecord = {
                "features": feature_set,
                "dimensions": n,
                "model": name,
                "acc": accuracy_score(y_test, predicted),
                "auc": roc_auc_score(y_test, scores),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": confusion_matrix(
                    y_test, predicted, labels=[0, 1]
                ).tolist(),
            }
            records.append(result)
            fpr, tpr, _ = roc_curve(y_test, scores)
            curves.append((fpr, tpr))
            print(
                f"{feature_set:6} {name:7} ACC={result['acc']:.4f}  AUC={result['auc']:.4f}",
                flush=True,
            )
    return records, curves


def save_plots(
    records: Sequence[EvaluationRecord],
    curves: Sequence[ScoreCurve],
    output: Path,
) -> None:
    fig, (bar, roc) = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    colors: tuple[str, str, str] = ("#16697A", "#D97706", "#6649A7")
    for j, feature_set in enumerate(FEATURE_SETS):
        subset = [r for r in records if r["features"] == feature_set]
        for k, metric in enumerate(SCORE_METRICS):
            offset = (2 * j + k - 1.5) * 0.19
            bar.bar(
                np.arange(len(MODELS)) + offset,
                [r[metric] for r in subset],
                0.18,
                label=f"{feature_set} {metric.upper()}",
                color=("#4D90A4", "#C27635")[j],
                hatch=("", "//")[k],
                edgecolor="#333333",
                linewidth=0.4,
            )
    bar.axhline(0.5, color="#666666", linestyle=":", linewidth=1)
    bar.set(
        xticks=np.arange(len(MODELS)),
        xticklabels=MODELS,
        ylim=(0, 1.02),
        ylabel="Score",
        title="Test ACC and ROC-AUC",
    )
    bar.legend(fontsize=8, loc="lower center", ncol=2)
    roc.plot([0, 1], [0, 1], ":", color="#777777")
    for result, (fpr, tpr) in zip(records, curves):
        name, fs = result["model"], result["features"]
        roc.plot(
            fpr,
            tpr,
            color=colors[MODELS.index(name)],
            linestyle="--" if fs == "H" else "-",
            label=f"{name}, {fs} ({result['auc']:.3f})",
        )
    roc.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC curves (Stego positive)",
    )
    roc.legend(fontsize=8, loc="lower right")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(10, 3.4), layout="constrained")
    subset = [r for r in records if r["features"] == "H+Cooc"]
    for ax, result in zip(axes, subset):
        matrix = np.asarray(result["confusion_matrix"])
        ConfusionMatrixDisplay(matrix, display_labels=["Cover", "Stego"]).plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            values_format="d",
            im_kw={"vmin": 0, "vmax": matrix[0].sum()},
        )
        ax.set_title(result["model"])
    fig.suptitle("Confusion matrices: H+Cooc")
    fig.savefig(output / "confusion.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--limit-pairs", type=int, help="use a deterministic subset for a smoke test"
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    pairs: list[ImagePair] = list_pairs(args.data_dir)
    if args.limit_pairs is not None:
        if not 3 <= args.limit_pairs <= len(pairs):
            parser.error("--limit-pairs must be between 3 and the available pair count")
        selection = np.random.default_rng(SEED).choice(
            len(pairs), args.limit_pairs, replace=False
        )
        pairs = [pairs[i] for i in sorted(selection)]
    if len(pairs) < 3:
        parser.error("at least three pairs are required")
    train, test = split_pairs(len(pairs))
    print(f"Pairs: {len(train)} train, {len(test)} test; seed={SEED}", flush=True)
    features = extract_features(pairs, args.workers)
    with threadpool_limits(limits=4):
        records, curves = evaluate(features, train, test)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: Metrics = {
        "seed": SEED,
        "pairs": len(pairs),
        "train_pairs": len(train),
        "test_pairs": len(test),
        "test_filenames": [pairs[i][0].name for i in test],
        "results": records,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    save_plots(records, curves, args.output_dir)
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
