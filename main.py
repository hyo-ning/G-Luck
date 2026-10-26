import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "2")

import argparse

import numpy as np
import torch

from gluck.data import DATASETS, SPLIT_SEED, load_dataset
from gluck.train import train_fold


def parse_args():
    parser = argparse.ArgumentParser(description="G-LUCK graph classification")
    parser.add_argument("--dataset", choices=DATASETS, default="MUTAG")
    parser.add_argument("--folds", type=int, nargs="+", choices=range(10), default=[0])
    parser.add_argument("--layers", type=int, choices=(4, 5, 6), default=5)
    parser.add_argument("--dropout", type=float, choices=(0.0, 0.3), default=0.3)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")
    if len(set(args.folds)) != len(args.folds):
        parser.error("folds must be unique")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    args.device = device
    print(
        f"Dataset: {args.dataset} | Device: {device} | Depth: {args.layers} | "
        f"Dropout: {args.dropout:g} | Training seed: {args.seed} | Split seed: {SPLIT_SEED}",
        flush=True,
    )
    graphs, splits, n_classes = load_dataset(args.dataset)
    results = [
        train_fold(graphs, splits[fold], n_classes, args, fold)
        for fold in args.folds
    ]
    print(f"\n{args.dataset} results (accuracy %)")
    print(f"{'Fold':>4}  {'Best epoch':>10}  {'Validation':>10}  {'Test':>10}")
    print("-" * 40)
    for result in results:
        print(
            f"{result['fold']:>4}  {result['selected_epoch']:>10}  "
            f"{100 * result['validation']['accuracy']:>9.2f}%  "
            f"{100 * result['test']['accuracy']:>9.2f}%"
        )
    accuracies = np.array([result["test"]["accuracy"] for result in results]) * 100
    if len(results) > 1:
        print(
            f"{len(results)}-fold test accuracy: "
            f"{accuracies.mean():.2f}% +/- {accuracies.std():.2f}% (mean +/- std)"
        )
    else:
        print(f"Test accuracy: {accuracies[0]:.2f}%")


if __name__ == "__main__":
    main()
