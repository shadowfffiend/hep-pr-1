"""Shared preprocessing for model training and inference."""

from pathlib import Path

import joblib
import pandas as pd


FEATURES = ["Age", "Sex", "ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"]
LABELS = {
    "0=Blood Donor": 0,
    "0s=suspect Blood Donor": 0,
    "1=Hepatitis": 1,
    "2=Fibrosis": 1,
    "3=Cirrhosis": 1,
}
ARTIFACTS = ("xgb_model.joblib", "robust_scaler.joblib", "medians_dict.joblib", "outlier_bounds.joblib")


def read_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path)
    missing = set(FEATURES + ["Category"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {', '.join(sorted(missing))}")
    unknown = set(frame["Category"].dropna()) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown Category values: {sorted(unknown)}")
    if frame["Category"].isna().any():
        raise ValueError("Category contains missing values")
    features = frame[FEATURES].copy()
    features["Sex"] = features["Sex"].map({"m": 0, "f": 1})
    for name in FEATURES:
        features[name] = pd.to_numeric(features[name], errors="raise")
    target = frame["Category"].map(LABELS).astype(int)
    return features, target


def fit_preprocessing(train: pd.DataFrame) -> tuple[pd.DataFrame, object, dict, dict]:
    from sklearn.preprocessing import RobustScaler

    medians = train.median().to_dict()
    filled = train.fillna(medians)
    bounds = {
        name: {"lower": float(filled[name].quantile(0.01)), "upper": float(filled[name].quantile(0.99))}
        for name in FEATURES
    }
    clipped = apply_bounds(filled, bounds)
    scaler = RobustScaler().fit(clipped)
    scaled = pd.DataFrame(scaler.transform(clipped), columns=FEATURES, index=train.index)
    return scaled, scaler, medians, bounds


def apply_bounds(frame: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    result = frame.copy()
    for name in FEATURES:
        result[name] = result[name].clip(bounds[name]["lower"], bounds[name]["upper"])
    return result


def transform(frame: pd.DataFrame, scaler: object, medians: dict, bounds: dict) -> pd.DataFrame:
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(sorted(missing))}")
    ordered = frame[FEATURES].copy()
    ordered["Sex"] = ordered["Sex"].replace({"m": 0, "f": 1, "Мужской": 0, "Женский": 1})
    for name in FEATURES:
        ordered[name] = pd.to_numeric(ordered[name], errors="raise")
    clipped = apply_bounds(ordered.fillna(medians), bounds)
    return pd.DataFrame(scaler.transform(clipped), columns=FEATURES, index=frame.index)


def save_artifacts(directory: Path, model: object, scaler: object, medians: dict, bounds: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in zip(ARTIFACTS, (model, scaler, medians, bounds)):
        joblib.dump(value, directory / name)


def load_artifacts(directory: Path) -> tuple[object, object, dict, dict]:
    return tuple(joblib.load(directory / name) for name in ARTIFACTS)

