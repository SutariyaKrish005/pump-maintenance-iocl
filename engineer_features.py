import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_SENSOR_CSV = os.path.join(DATA_DIR, "pump_sensor_data.csv")
FAILURE_EVENTS_CSV = os.path.join(DATA_DIR, "pump_failure_events.csv")
FEATURES_CSV = os.path.join(DATA_DIR, "pump_features.csv")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
FREQ_MINUTES = 10
STEPS_PER_HOUR = 60 // FREQ_MINUTES         # 6 steps = 1 hour at 10-min frequency
WINDOW_1H = STEPS_PER_HOUR                  # 6
WINDOW_6H = STEPS_PER_HOUR * 6              # 36

MAINTENANCE_DUE_THRESHOLD_HOURS = 700.0     # ~29 days of continuous runtime

# "Pump-relative baseline" window: how a pump's CURRENT readings compare to
# its OWN typical normal operating level, rather than an absolute value that
# differs from pump to pump. This generalizes much better to pumps that had
# no failure examples during training, since two pumps can have very
# different normal baselines (e.g. 1.6 mm/s vs 2.4 mm/s vibration) while
# both showing, say, a 2x rise above their own baseline before failure.
BASELINE_WINDOW_DAYS = 30
BASELINE_LAG_DAYS = 5   # baseline is computed from data more than this many
                        # days old, so it stays well clear of the 96-hour
                        # (4-day) pre-failure anomaly ramp and isn't itself
                        # dragged up by the anomaly it's meant to detect
STEPS_PER_DAY = 24 * (60 // FREQ_MINUTES)  # 144 steps/day at 10-min frequency
BASELINE_WINDOW_STEPS = BASELINE_WINDOW_DAYS * STEPS_PER_DAY
BASELINE_LAG_STEPS = BASELINE_LAG_DAYS * STEPS_PER_DAY
BASELINE_MIN_PERIODS = 3 * STEPS_PER_DAY  # need at least 3 days of history to trust the baseline


def add_rolling_and_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds rolling statistics, rate-of-change, and lag features, computed
    per pump_id and causally (each row only uses its own past + current
    values -- pandas' rolling()/shift() are backward-looking by default,
    so there is no leakage from future readings)."""
    df = df.sort_values(["pump_id", "timestamp"]).reset_index(drop=True)
    grouped = df.groupby("pump_id", group_keys=False)

    # ---- Rolling statistics ----
    df["rolling_vibration_mean_1h"] = grouped["vibration_mm_s"].transform(
        lambda s: s.rolling(window=WINDOW_1H, min_periods=WINDOW_1H).mean()
    )
    df["rolling_vibration_std_1h"] = grouped["vibration_mm_s"].transform(
        lambda s: s.rolling(window=WINDOW_1H, min_periods=WINDOW_1H).std()
    )
    df["rolling_temp_mean_1h"] = grouped["bearing_temp_c"].transform(
        lambda s: s.rolling(window=WINDOW_1H, min_periods=WINDOW_1H).mean()
    )
    df["rolling_temp_max_6h"] = grouped["bearing_temp_c"].transform(
        lambda s: s.rolling(window=WINDOW_6H, min_periods=WINDOW_6H).max()
    )
    df["rolling_current_mean_1h"] = grouped["motor_current_a"].transform(
        lambda s: s.rolling(window=WINDOW_1H, min_periods=WINDOW_1H).mean()
    )
    df["rolling_suction_min_6h"] = grouped["suction_pressure_bar"].transform(
        lambda s: s.rolling(window=WINDOW_6H, min_periods=WINDOW_6H).min()
    )

    # ---- Rate-of-change features (current value minus value 1 hour ago) ----
    df["vibration_change_1h"] = grouped["vibration_mm_s"].transform(
        lambda s: s - s.shift(WINDOW_1H)
    )
    df["temp_change_1h"] = grouped["bearing_temp_c"].transform(
        lambda s: s - s.shift(WINDOW_1H)
    )
    df["current_change_1h"] = grouped["motor_current_a"].transform(
        lambda s: s - s.shift(WINDOW_1H)
    )

    # ---- Lag features ----
    df["vibration_lag_1"] = grouped["vibration_mm_s"].transform(lambda s: s.shift(1))
    df["vibration_lag_6"] = grouped["vibration_mm_s"].transform(lambda s: s.shift(WINDOW_1H))

    # ---- Pump-relative baseline features ----
    # Each pump's "normal" operating level differs (different equipment,
    # different baseline vibration/temperature/pressure). A ratio to that
    # pump's own recent baseline generalizes far better across pumps than
    # an absolute sensor value does -- especially for pumps that had no
    # failure examples in the training period.
    def baseline(series: pd.Series) -> pd.Series:
        return series.shift(BASELINE_LAG_STEPS).rolling(
            window=BASELINE_WINDOW_STEPS, min_periods=BASELINE_MIN_PERIODS
        ).mean()

    df["vibration_baseline_30d"] = grouped["vibration_mm_s"].transform(baseline)
    df["temp_baseline_30d"] = grouped["bearing_temp_c"].transform(baseline)
    df["suction_baseline_30d"] = grouped["suction_pressure_bar"].transform(baseline)

    df["vibration_ratio_to_baseline"] = df["vibration_mm_s"] / df["vibration_baseline_30d"]
    df["temp_ratio_to_baseline"] = df["bearing_temp_c"] / df["temp_baseline_30d"]
    df["suction_ratio_to_baseline"] = df["suction_pressure_bar"] / df["suction_baseline_30d"]

    return df


def add_operating_context_features(df: pd.DataFrame, failure_events: pd.DataFrame) -> pd.DataFrame:
    """Adds hours_since_last_failure (time since the most recent PAST
    failure of that pump, using only failures that occurred strictly
    before the current timestamp -- causal by construction) and
    maintenance_due_flag (derived from the raw runtime column)."""
    df = df.sort_values(["pump_id", "timestamp"]).reset_index(drop=True)

    df["hours_since_last_failure"] = np.nan
    for pump_id, pump_df in df.groupby("pump_id"):
        pump_failures = failure_events.loc[
            failure_events["pump_id"] == pump_id, "failure_time"
        ].sort_values().values

        ts = pump_df["timestamp"].values
        if len(pump_failures) == 0:
            # No historical failures yet for this pump: use hours since the
            # start of the observation period as a neutral fallback.
            hours_since = (ts - ts[0]) / np.timedelta64(1, "h")
        else:
            idx_last_failure = np.searchsorted(pump_failures, ts, side="right") - 1
            has_prior_failure = idx_last_failure >= 0
            idx_clipped = np.clip(idx_last_failure, 0, len(pump_failures) - 1)
            last_failure_time = pump_failures[idx_clipped]
            hours_since_failure = (ts - last_failure_time) / np.timedelta64(1, "h")
            # Before any failure has occurred yet, fall back to hours since
            # the start of the observation period (no prior failure to reference).
            hours_since_start = (ts - ts[0]) / np.timedelta64(1, "h")
            hours_since = np.where(has_prior_failure, hours_since_failure, hours_since_start)

        df.loc[df["pump_id"] == pump_id, "hours_since_last_failure"] = hours_since

    df["maintenance_due_flag"] = (
        df["runtime_hours_since_maintenance"] > MAINTENANCE_DUE_THRESHOLD_HOURS
    ).astype(int)

    return df


def main():
    if not os.path.exists(RAW_SENSOR_CSV):
        raise FileNotFoundError(f"{RAW_SENSOR_CSV} not found. Run 'python src/generate_data.py' first.")
    if not os.path.exists(FAILURE_EVENTS_CSV):
        raise FileNotFoundError(f"{FAILURE_EVENTS_CSV} not found. Run 'python src/generate_data.py' first.")

    df = pd.read_csv(RAW_SENSOR_CSV, parse_dates=["timestamp"])
    failure_events = pd.read_csv(FAILURE_EVENTS_CSV, parse_dates=["failure_time"])

    print(f"Loaded raw sensor data: {len(df):,} rows across {df['pump_id'].nunique()} pumps")

    df = add_rolling_and_lag_features(df)
    df = add_operating_context_features(df, failure_events)

    n_before = len(df)
    # Rolling/lag features are undefined for the first WINDOW_6H records of
    # each pump (needs at least window worth of prior history); drop those
    # rows rather than fill, since fabricated values here could mislead the
    # classifier during the (small) pump-startup period.
    engineered_cols = [
        "rolling_vibration_mean_1h", "rolling_vibration_std_1h", "rolling_temp_mean_1h",
        "rolling_temp_max_6h", "rolling_current_mean_1h", "rolling_suction_min_6h",
        "vibration_change_1h", "temp_change_1h", "current_change_1h",
        "vibration_lag_1", "vibration_lag_6",
        "vibration_ratio_to_baseline", "temp_ratio_to_baseline", "suction_ratio_to_baseline",
    ]
    df = df.dropna(subset=engineered_cols).reset_index(drop=True)
    n_after = len(df)
    print(f"Dropped {n_before - n_after:,} startup rows with insufficient rolling/lag history "
          f"({n_before:,} -> {n_after:,} rows).")

    assert df.isnull().sum().sum() == 0, "Unexpected missing values remain after feature engineering!"

    df.to_csv(FEATURES_CSV, index=False)
    print(f"Saved feature-engineered dataset: {FEATURES_CSV}")
    print(f"Failure rate after processing: {df['failure_flag'].mean() * 100:.2f}%")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
