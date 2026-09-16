"""
predict.py
----------
Loads the saved pump failure model and the feature-engineered dataset,
regenerates predictions for the held-out test period (same time-based
split and classification threshold used in train_model.py), and saves
them to output/test_predictions.csv.

Run:
    python src/predict.py
"""

import os
import warnings

import joblib
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FEATURES_CSV = os.path.join(DATA_DIR, "pump_features.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "pump_failure_model.joblib")
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "test_predictions.csv")


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run 'python src/train_model.py' first.")
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"{FEATURES_CSV} not found. Run 'python src/engineer_features.py' first.")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    val_cutoff = bundle["val_cutoff"]
    threshold = bundle["threshold"]

    df = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
    test_df = df[df["timestamp"] >= val_cutoff].copy()

    X_test = test_df[feature_cols]
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    predictions = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "pump_id": test_df["pump_id"],
            "actual_failure_flag": test_df["failure_flag"].values,
            "predicted_failure_flag": y_pred,
            "failure_probability": y_proba.round(4),
        }
    )
    predictions.to_csv(PREDICTIONS_CSV, index=False)

    n_correct_alerts = int(((predictions["actual_failure_flag"] == 1) & (predictions["predicted_failure_flag"] == 1)).sum())
    n_missed = int(((predictions["actual_failure_flag"] == 1) & (predictions["predicted_failure_flag"] == 0)).sum())
    n_false_alarms = int(((predictions["actual_failure_flag"] == 0) & (predictions["predicted_failure_flag"] == 1)).sum())

    print(f"Loaded model: {bundle['model_name']} (threshold={threshold:.3f})")
    print(f"Test period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    print(f"Predictions saved: {PREDICTIONS_CSV} ({len(predictions):,} rows)")
    print(f"Correctly flagged at-risk readings: {n_correct_alerts:,}")
    print(f"Missed at-risk readings          : {n_missed:,}")
    print(f"False alarms                     : {n_false_alarms:,}")


if __name__ == "__main__":
    main()
