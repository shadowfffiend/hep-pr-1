"""Train and evaluate the coursework XGBoost classifier on a local CSV."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.pipeline import FEATURES, fit_preprocessing, read_dataset, save_artifacts, transform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to HepatitisCdata.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    features, target = read_dataset(args.data)
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=0.3, random_state=args.seed, stratify=target
    )
    train_scaled, scaler, medians, bounds = fit_preprocessing(train_x)
    test_scaled = transform(test_x, scaler, medians, bounds)

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        eval_metric="logloss",
        reg_alpha=0,
        reg_lambda=0.1,
        gamma=0.1,
        scale_pos_weight=1,
        random_state=args.seed,
        n_jobs=1,
    )
    model.fit(train_scaled, train_y)
    predictions = model.predict(test_scaled)
    probabilities = model.predict_proba(test_scaled)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(test_y, predictions)),
        "precision": float(precision_score(test_y, predictions, zero_division=0)),
        "recall": float(recall_score(test_y, predictions, zero_division=0)),
        "f1": float(f1_score(test_y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(test_y, probabilities)),
        "confusion_matrix": confusion_matrix(test_y, predictions, labels=[0, 1]).tolist(),
        "train_rows": len(train_y),
        "test_rows": len(test_y),
    }
    save_artifacts(args.out_dir, model, scaler, medians, bounds)
    params = {
        "data_file": args.data.name,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "seed": args.seed,
        "test_size": 0.3,
        "features": FEATURES,
        "model": "XGBClassifier",
        "model_params": model.get_params(),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    (args.out_dir / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved experiment to {args.out_dir}")


if __name__ == "__main__":
    main()

