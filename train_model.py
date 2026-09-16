import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FEATURES_CSV = os.path.join(DATA_DIR, "pump_features.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
for d in (FIGURES_DIR, MODELS_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TRAIN_FRACTION = 0.65      # first 65% of the time range: model training
VALIDATION_FRACTION = 0.75  # 65-75% of the time range: threshold tuning (not used to fit the model)
# The remaining 75-100% is the held-out test set used only for final reporting.

# Model choice: "random_forest" (default, always available) or "xgboost"
# (used only if the xgboost package is installed; falls back to
# RandomForest with a printed notice otherwise, so this script never
# raises an import error out of the box).
MODEL_TYPE = "random_forest"

RANDOM_STATE = 42

TARGET_COL = "failure_flag"
NON_FEATURE_COLS = ["timestamp", "pump_id", TARGET_COL]

EXAMPLE_PLOT_PUMP = "PUMP_003"  # pump used for the illustrative timeline plot


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(
            f"{FEATURES_CSV} not found. Run 'python src/generate_data.py' then "
            "'python src/engineer_features.py' first."
        )
    df = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def time_based_split(df: pd.DataFrame, train_fraction: float, validation_fraction: float):
    """Splits by absolute timestamp cutoffs computed over the whole
    dataset's time range (all pumps share the same observation period,
    so these cutoffs apply uniformly per pump too). No shuffling --
    preserves time order throughout to avoid leakage from future to past.

    Three chronological slices are produced:
      - train:      used to fit the model
      - validation: a held-out slice (still before the test period) used
                     only to choose a classification threshold, since the
                     default 0.5 cutoff is a poor fit for a rare-event
                     problem like this one
      - test:       the final held-out slice, touched only for reporting
    """
    min_ts, max_ts = df["timestamp"].min(), df["timestamp"].max()
    train_cutoff = min_ts + train_fraction * (max_ts - min_ts)
    val_cutoff = min_ts + validation_fraction * (max_ts - min_ts)

    train_df = df[df["timestamp"] < train_cutoff].copy()
    val_df = df[(df["timestamp"] >= train_cutoff) & (df["timestamp"] < val_cutoff)].copy()
    test_df = df[df["timestamp"] >= val_cutoff].copy()
    return train_df, val_df, test_df, train_cutoff, val_cutoff


def select_threshold(y_val, y_val_proba, default: float = 0.5) -> float:
    """Chooses the classification threshold that maximizes F1 on the
    validation set (a slice the model was not trained on, and distinct
    from the final test set). Falls back to the default threshold if the
    validation set has no positive examples to tune against."""
    if y_val.sum() == 0:
        print("Validation set has no positive examples; using default threshold 0.5.")
        return default

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    f1_scores = np.divide(
        2 * precisions[:-1] * recalls[:-1],
        precisions[:-1] + recalls[:-1],
        out=np.zeros_like(precisions[:-1]),
        where=(precisions[:-1] + recalls[:-1]) > 0,
    )
    if len(f1_scores) == 0 or np.all(f1_scores == 0):
        print("Could not find a threshold improving on default; using 0.5.")
        return default

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def build_model(model_type: str):
    """Returns a fitted-ready classifier instance. Falls back to
    RandomForest if xgboost is requested but not installed."""
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            print("xgboost is not installed -- falling back to RandomForestClassifier. "
                  "Install with 'pip install xgboost' to use MODEL_TYPE='xgboost'.")
            model_type = "random_forest"
        else:
            return (
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.08,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                "XGBClassifier",
            )

    return (
        RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "RandomForestClassifier",
    )


def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["No Failure (0)", "Failure (1)"],
        yticklabels=["No Failure (0)", "Failure (1)"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix - Pump Failure Prediction", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, auc_score: float, save_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, color="#C44E52", linewidth=2, label=f"ROC curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Pump Failure Prediction", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(feature_names, importances, save_path: str, top_n: int = 15):
    order = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in order]
    top_importances = importances[order]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_features[::-1], top_importances[::-1], color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances - Pump Failure Model", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_example_pump_timeline(df: pd.DataFrame, pump_id: str, save_path: str):
    """Plots vibration and bearing temperature for one pump, highlighting
    the failure_flag=1 windows, so a reader can visually see the
    precursor pattern the model is trained to detect."""
    pump_df = df[df["pump_id"] == pump_id].sort_values("timestamp")
    if pump_df.empty:
        print(f"No data found for {pump_id}; skipping example timeline plot.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(pump_df["timestamp"], pump_df["vibration_mm_s"], color="#4C72B0", label="Vibration (mm/s)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Vibration (mm/s)", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    ax2 = ax1.twinx()
    ax2.plot(pump_df["timestamp"], pump_df["bearing_temp_c"], color="#C44E52", alpha=0.7, label="Bearing Temp (C)")
    ax2.set_ylabel("Bearing Temperature (C)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    # Shade the failure_flag=1 windows
    flagged = pump_df["failure_flag"].values
    ts = pump_df["timestamp"].values
    in_window = False
    window_start = None
    for i in range(len(flagged)):
        if flagged[i] == 1 and not in_window:
            in_window = True
            window_start = ts[i]
        elif flagged[i] == 0 and in_window:
            in_window = False
            ax1.axvspan(window_start, ts[i], color="orange", alpha=0.25)
    if in_window:
        ax1.axvspan(window_start, ts[-1], color="orange", alpha=0.25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title(
        f"{pump_id}: Sensor Trends with Failure-Risk Windows Highlighted (orange)",
        fontsize=13, fontweight="bold",
    )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    df = load_dataset()
    feature_cols = get_feature_columns(df)

    train_df, val_df, test_df, train_cutoff, val_cutoff = time_based_split(
        df, TRAIN_FRACTION, VALIDATION_FRACTION
    )

    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

    print("=" * 70)
    print(f"Train period     : {train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()}")
    print(f"Validation period: {val_df['timestamp'].min().date()} to {val_df['timestamp'].max().date()}")
    print(f"Test period      : {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}")
    print(f"Train samples     : {len(train_df):,} (failure: {y_train.mean() * 100:.2f}%)")
    print(f"Validation samples: {len(val_df):,} (failure: {y_val.mean() * 100:.2f}%)")
    print(f"Test samples      : {len(test_df):,} (failure: {y_test.mean() * 100:.2f}%)")
    print(f"Number of features: {len(feature_cols)}")

    model, model_name = build_model(MODEL_TYPE)
    print(f"Model: {model_name}")
    model.fit(X_train, y_train)

    # ---- Threshold tuning on validation set (never seen by the model, and
    # kept separate from the final test set to avoid tuning on the data
    # we report results on) ----
    y_val_proba = model.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y_val, y_val_proba)
    print(f"Selected classification threshold (tuned on validation set): {threshold:.3f}")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("-" * 70)
    print(f"Precision (failure): {precision:.3f}")
    print(f"Recall (failure)   : {recall:.3f}")
    print(f"F1 (failure)       : {f1:.3f}")
    print(f"ROC-AUC            : {auc:.3f}")
    print(f"Confusion matrix   : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("-" * 70)
    print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"], zero_division=0))

    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    interpretation = (
        f"The model correctly identifies ~{recall * 100:.0f}% of impending failures "
        f"(recall) with a false alarm rate of ~{false_alarm_rate * 100:.1f}% among normal "
        f"operating periods. This can enable maintenance teams to intervene 24-72 hours "
        f"before most failures, at the cost of investigating some false alarms."
    )
    print(f"Interpretation: {interpretation}")

    # ---- Feature importance ----
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.zeros(len(feature_cols))
    top_idx = np.argsort(importances)[::-1][:10]
    top_features_summary = [(feature_cols[i], round(float(importances[i]), 4)) for i in top_idx]

    # ---- Plots ----
    plot_confusion_matrix(cm, os.path.join(FIGURES_DIR, "confusion_matrix.png"))
    plot_roc_curve(y_test, y_proba, auc, os.path.join(FIGURES_DIR, "roc_curve.png"))
    plot_feature_importance(feature_cols, importances, os.path.join(FIGURES_DIR, "feature_importance.png"))
    plot_example_pump_timeline(df, EXAMPLE_PLOT_PUMP, os.path.join(FIGURES_DIR, "example_pump_timeline.png"))
    print(f"Saved plots to: {FIGURES_DIR}")

    # ---- Model persistence ----
    model_path = os.path.join(MODELS_DIR, "pump_failure_model.joblib")
    joblib.dump(
        {
            "model": model,
            "model_name": model_name,
            "feature_cols": feature_cols,
            "train_cutoff": train_cutoff,
            "val_cutoff": val_cutoff,
            "threshold": threshold,
        },
        model_path,
    )
    print(f"Saved model: {model_path}")

    # ---- Metrics summary ----
    metrics_txt_path = os.path.join(OUTPUT_DIR, "metrics_summary.txt")
    with open(metrics_txt_path, "w") as f:
        f.write("Predictive Maintenance - Pump Failure Model: Metrics Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Train period     : {train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()}\n")
        f.write(f"Validation period: {val_df['timestamp'].min().date()} to {val_df['timestamp'].max().date()}\n")
        f.write(f"Test period      : {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}\n")
        f.write(f"Train samples     : {len(train_df):,} (failure: {y_train.mean() * 100:.2f}%)\n")
        f.write(f"Validation samples: {len(val_df):,} (failure: {y_val.mean() * 100:.2f}%)\n")
        f.write(f"Test samples      : {len(test_df):,} (failure: {y_test.mean() * 100:.2f}%)\n")
        f.write(f"Classification threshold (tuned on validation): {threshold:.3f}\n\n")
        f.write(f"Precision (failure): {precision:.3f}\n")
        f.write(f"Recall (failure)   : {recall:.3f}\n")
        f.write(f"F1 (failure)       : {f1:.3f}\n")
        f.write(f"ROC-AUC            : {auc:.3f}\n")
        f.write(f"Confusion matrix   : TN={tn}, FP={fp}, FN={fn}, TP={tp}\n\n")
        f.write("Top 10 important features:\n")
        for name, imp in top_features_summary:
            f.write(f"  {name}: {imp}\n")
        f.write(f"\nInterpretation: {interpretation}\n")

    metrics_csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    metrics_row = {
        "model": model_name,
        "train_start": train_df["timestamp"].min().strftime("%Y-%m-%d"),
        "train_end": train_df["timestamp"].max().strftime("%Y-%m-%d"),
        "validation_start": val_df["timestamp"].min().strftime("%Y-%m-%d"),
        "validation_end": val_df["timestamp"].max().strftime("%Y-%m-%d"),
        "test_start": test_df["timestamp"].min().strftime("%Y-%m-%d"),
        "test_end": test_df["timestamp"].max().strftime("%Y-%m-%d"),
        "n_train": len(train_df),
        "n_validation": len(val_df),
        "n_test": len(test_df),
        "train_failure_rate_pct": round(y_train.mean() * 100, 3),
        "test_failure_rate_pct": round(y_test.mean() * 100, 3),
        "classification_threshold": round(threshold, 4),
        "precision_failure": round(precision, 4),
        "recall_failure": round(recall, 4),
        "f1_failure": round(f1, 4),
        "roc_auc": round(auc, 4),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "top_features": "; ".join(f"{n}({i})" for n, i in top_features_summary),
    }
    pd.DataFrame([metrics_row]).to_csv(metrics_csv_path, index=False)

    print(f"Saved metrics summary: {metrics_txt_path} and {metrics_csv_path}")


if __name__ == "__main__":
    main()
