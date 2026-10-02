"""Create class probabilities from a CSV using saved model artifacts."""

import argparse
from pathlib import Path

import pandas as pd

from src.pipeline import load_artifacts, transform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, scaler, medians, bounds = load_artifacts(args.model_dir)
    frame = pd.read_csv(args.input)
    scaled = transform(frame, scaler, medians, bounds)
    probability = model.predict_proba(scaled)[:, 1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pathology_probability": probability, "predicted_class": (probability >= 0.5).astype(int)}).to_csv(
        args.output, index=False
    )
    print(f"Saved {len(frame)} predictions to {args.output}")


if __name__ == "__main__":
    main()

