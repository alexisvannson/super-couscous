"""
Evaluate a trained model on the test set.

Computes per-label and macro F1 with 95% bootstrap confidence intervals,
following the methodology of Rajpurkar et al. (CheXNet).

Usage:
    python scripts/evaluate.py densenet --weights models/checkpoints/best.pth
    python scripts/evaluate.py densenet --weights models/checkpoints/best.pth --n-bootstrap 2000
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.thedataloader import get_dataloaders
from scripts.training import create_model, load_config, get_transforms, tune_thresholds


def collect_predictions(model, loader, device, thresholds=0.5):
    """Return (preds, targets) as numpy arrays of shape (N, C)."""
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.sigmoid(model(images.to(device))).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.float().numpy())
    probs   = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    preds   = (probs >= thresholds).astype(float)
    return preds, targets

def f1_per_label(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-label F1 scores, shape (C,)."""
    tp = (preds * targets).sum(axis=0)
    fp = (preds * (1 - targets)).sum(axis=0)
    fn = ((1 - preds) * targets).sum(axis=0)
    return 2 * tp / (2 * tp + fp + fn + 1e-8)


def macro_f1(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(f1_per_label(preds, targets).mean())

def bootstrap_ci(preds: np.ndarray, targets: np.ndarray,
                 metric_fn, n: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for a scalar metric.

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n_samples = len(preds)
    scores = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        scores[i] = metric_fn(preds[idx], targets[idx])
    lo = float(np.percentile(scores, 100 * alpha / 2))
    hi = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return metric_fn(preds, targets), lo, hi


def bootstrap_ci_per_label(preds: np.ndarray, targets: np.ndarray,
                            n: int = 1000, alpha: float = 0.05,
                            seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap CI for every label independently.

    Returns:
        (point_estimates, lower_bounds, upper_bounds) each of shape (C,)
    """
    rng = np.random.default_rng(seed)
    n_samples, n_classes = preds.shape
    boot = np.empty((n, n_classes))
    for i in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        boot[i] = f1_per_label(preds[idx], targets[idx])
    lo = np.percentile(boot, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
    return f1_per_label(preds, targets), lo, hi


def print_report(point, lo, hi, per_label_point, per_label_lo, per_label_hi, label_names):
    w = max(len(l) for l in label_names) + 2
    print(f"  Macro F1: {point:.3f}  (95% CI {lo:.3f} – {hi:.3f})")
    print(f"  {'Label':<{w}}  {'F1':>6}   {'95% CI':>18}")
    print(f"  {'-'*w}  {'------':>6}   {'------------------':>18}")
    for name, p, l, h in zip(label_names, per_label_point, per_label_lo, per_label_hi):
        print(f"  {name:<{w}}  {p:.3f}   ({l:.3f} – {h:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model with bootstrap CI")
    parser.add_argument("model", type=str, help="Model name (must match configs/)")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed threshold (overrides tuned thresholds)")
    parser.add_argument("--thresholds", type=str, default=None, help="Path to .npy file with per-label thresholds")
    parser.add_argument("--tune", action="store_true", help="Tune thresholds on val set before evaluating")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = create_model(args.model, config)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    print(f"Loaded weights: {args.weights}")

    data_cfg = config.get("data", {})
    dl_cfg   = config.get("dataloader", {})
    eval_transform = get_transforms(config.get("transform", {}))
    _, val_loader, test_loader = get_dataloaders(
        data_path=data_cfg.get("image_dir", "data/sample/images"),
        label_path=data_cfg.get("labels_csv", "data/sample_labels.csv"),
        batch_size=dl_cfg.get("batch_size", 32),
        val_split=dl_cfg.get("val_split", 0.2),
        test_split=dl_cfg.get("test_split", 0.1),
        seed=dl_cfg.get("seed", 42),
        num_workers=dl_cfg.get("num_workers", 0),
        eval_transform=eval_transform,
    )

    label_names = test_loader.dataset.labels

    # Resolve thresholds: CLI flag > saved .npy > tune on val > default 0.5
    if args.threshold is not None:
        thresholds = args.threshold
        print(f"Using fixed threshold: {thresholds}")
    elif args.thresholds:
        thresholds = np.load(args.thresholds)
        print(f"Loaded thresholds from {args.thresholds}")
    elif args.tune:
        print("Tuning thresholds on validation set...")
        thresholds = tune_thresholds(model, val_loader, device)
        for name, t in zip(label_names, thresholds):
            print(f"  {name}: {t:.2f}")
    else:
        thresholds = 0.5
        print("Using default threshold: 0.5")

    print(f"Running inference on {len(test_loader.dataset)} test samples…")
    preds, targets = collect_predictions(model, test_loader, device, thresholds)

    print(f"Computing bootstrap CIs (n={args.n_bootstrap})…")
    point, lo, hi = bootstrap_ci(
        preds, targets, macro_f1,
        n=args.n_bootstrap, seed=args.seed,
    )
    per_label_point, per_label_lo, per_label_hi = bootstrap_ci_per_label(
        preds, targets,
        n=args.n_bootstrap, seed=args.seed,
    )

    print_report(point, lo, hi, per_label_point, per_label_lo, per_label_hi, label_names)


if __name__ == "__main__":
    main()
