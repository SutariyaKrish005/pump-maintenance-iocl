import os
import numpy as np
import pandas as pd

RNG_SEED = 42

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
SENSOR_CSV = os.path.join(DATA_DIR, "pump_sensor_data.csv")
FAILURE_EVENTS_CSV = os.path.join(DATA_DIR, "pump_failure_events.csv")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
START_DATE = pd.Timestamp("2024-01-01 00:00:00")
END_DATE = pd.Timestamp("2024-06-30 23:50:00")  # ~6 months
FREQ_MINUTES = 10

PUMP_IDS = [f"PUMP_{i:03d}" for i in range(1, 9)]  # 8 pumps

RAMP_HOURS = 96.0          # anomaly precursor window before a failure
LABEL_WINDOW_MIN_HOURS = 24.0   # failure_flag=1 window: 24-72h before failure
LABEL_WINDOW_MAX_HOURS = 72.0

FAILURE_PROBABILITY_PER_CYCLE = 0.5   # chance a given operating cycle ends in failure
CYCLE_LEN_DAYS_RANGE = (25, 45)       # length of an operating cycle if no failure
FAILURE_OFFSET_MIN_DAYS = 15          # earliest a failure can occur within a cycle
FAILURE_OFFSET_BUFFER_DAYS = 2        # failure occurs at least this many days before cycle end


def build_event_schedule(rng: np.random.RandomState, start: pd.Timestamp, end: pd.Timestamp):
    """Builds a sequence of (time, type) events for one pump, where type is
    'failure' (unplanned breakdown, preceded by an anomaly ramp) or
    'maintenance' (routine preventive maintenance, no anomaly precursor).
    Both event types reset runtime_hours_since_maintenance to 0."""
    events = []
    current_time = start
    while True:
        cycle_len_days = rng.uniform(*CYCLE_LEN_DAYS_RANGE)
        will_fail = rng.random() < FAILURE_PROBABILITY_PER_CYCLE
        if will_fail:
            max_offset = max(FAILURE_OFFSET_MIN_DAYS + 1, cycle_len_days - FAILURE_OFFSET_BUFFER_DAYS)
            fail_offset_days = rng.uniform(FAILURE_OFFSET_MIN_DAYS, max_offset)
            event_time = current_time + pd.Timedelta(days=fail_offset_days)
            event_type = "failure"
        else:
            event_time = current_time + pd.Timedelta(days=cycle_len_days)
            event_type = "maintenance"

        if event_time >= end:
            break

        events.append((event_time, event_type))
        current_time = event_time

    return events


def simulate_pump(pump_id: str, rng: np.random.RandomState) -> pd.DataFrame:
    """Simulates one pump's full sensor history, including normal
    operating fluctuations and pre-failure anomaly ramps."""
    timestamps = pd.date_range(START_DATE, END_DATE, freq=f"{FREQ_MINUTES}min")
    n = len(timestamps)
    ts_values = timestamps.values.astype("datetime64[ns]")

    events = build_event_schedule(rng, START_DATE, END_DATE)
    if len(events) == 0:
        # Extremely unlikely given the configured ranges, but guard anyway.
        events = [(END_DATE, "maintenance")]

    event_times = np.array([e[0] for e in events], dtype="datetime64[ns]")
    event_types_list = [e[1] for e in events]
    event_types_ext = np.array(event_types_list + ["none"], dtype=object)
    far_future = END_DATE + pd.Timedelta(days=3650)
    event_times_ext = np.append(event_times, np.datetime64(far_future))

    # For each timestamp, find the index of the last event at/before it.
    idx_last_event = np.searchsorted(event_times, ts_values, side="right") - 1
    idx_last_event_clipped = np.clip(idx_last_event, 0, len(event_times) - 1)
    last_reset_time = np.where(
        idx_last_event >= 0,
        event_times[idx_last_event_clipped],
        np.datetime64(START_DATE),
    )
    runtime_hours = (ts_values - last_reset_time) / np.timedelta64(1, "h")

    # Next event (used for the anomaly ramp and the failure_flag label window)
    idx_next_event = idx_last_event + 1
    idx_next_event_ext = np.clip(idx_next_event, 0, len(event_types_list))
    next_event_type = event_types_ext[idx_next_event_ext]
    next_event_time = event_times_ext[idx_next_event_ext]
    hours_until_next = (next_event_time - ts_values) / np.timedelta64(1, "h")

    is_next_failure = next_event_type == "failure"
    failure_flag = (
        is_next_failure
        & (hours_until_next >= LABEL_WINDOW_MIN_HOURS)
        & (hours_until_next <= LABEL_WINDOW_MAX_HOURS)
    ).astype(int)

    # Anomaly intensity ramps from 0 to 1 as the failure approaches (within RAMP_HOURS),
    # using a nonlinear (power) curve so the anomaly worsens faster close to failure.
    ramp_active = is_next_failure & (hours_until_next <= RAMP_HOURS)
    raw_intensity = np.clip((RAMP_HOURS - hours_until_next) / RAMP_HOURS, 0.0, 1.0)
    anomaly_intensity = np.where(ramp_active, raw_intensity ** 1.5, 0.0)

    # ---- Pump-specific baseline characteristics (adds inter-pump variety) ----
    vib_base = rng.uniform(1.5, 2.5)
    temp_base = rng.uniform(55, 65)
    current_base = rng.uniform(40, 60)
    discharge_base = rng.uniform(12, 18)
    suction_base = rng.uniform(2.5, 4.0)
    flow_base = rng.uniform(150, 250)

    hours_of_day = timestamps.hour + timestamps.minute / 60.0
    # Mild diurnal operating-load cycle (refinery load varies slightly through the day)
    load_cycle = 0.92 + 0.08 * np.sin(2 * np.pi * (hours_of_day - 6) / 24.0)

    vibration = (
        vib_base * load_cycle
        + rng.normal(0, 0.15, n)
        + anomaly_intensity * rng.uniform(4.0, 7.0)
    )
    bearing_temp = (
        temp_base * (load_cycle ** 0.3)
        + rng.normal(0, 1.2, n)
        + anomaly_intensity * rng.uniform(20.0, 35.0)
    )
    motor_current = (
        current_base * load_cycle
        + rng.normal(0, 1.5, n)
        + anomaly_intensity * rng.uniform(8.0, 15.0)
    )
    # Cavitation-like behavior: suction pressure drops as anomaly worsens
    suction_pressure = (
        suction_base
        - anomaly_intensity * rng.uniform(1.0, 1.8)
        + rng.normal(0, 0.1, n)
    )
    discharge_pressure = (
        discharge_base * load_cycle
        + rng.normal(0, 0.3, n)
        - anomaly_intensity * rng.uniform(0.5, 1.0)
    )
    flow_rate = (
        flow_base * load_cycle
        + rng.normal(0, 5.0, n)
        - anomaly_intensity * rng.uniform(10.0, 25.0)
    )

    # Clip to physically sensible ranges
    vibration = np.clip(vibration, 0.2, None)
    bearing_temp = np.clip(bearing_temp, 20.0, None)
    motor_current = np.clip(motor_current, 5.0, None)
    suction_pressure = np.clip(suction_pressure, 0.2, None)
    discharge_pressure = np.clip(discharge_pressure, 1.0, None)
    flow_rate = np.clip(flow_rate, 10.0, None)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "pump_id": pump_id,
            "vibration_mm_s": np.round(vibration, 3),
            "bearing_temp_c": np.round(bearing_temp, 2),
            "motor_current_a": np.round(motor_current, 2),
            "discharge_pressure_bar": np.round(discharge_pressure, 2),
            "suction_pressure_bar": np.round(suction_pressure, 2),
            "flow_rate_m3_h": np.round(flow_rate, 1),
            "runtime_hours_since_maintenance": np.round(runtime_hours, 2),
            "failure_flag": failure_flag,
        }
    )

    failure_times = pd.to_datetime([e[0] for e in events if e[1] == "failure"])
    failure_events_df = pd.DataFrame(
        {
            "pump_id": pump_id,
            "failure_time": failure_times,
        }
    )

    return df, failure_events_df


def main():
    all_pump_frames = []
    all_failure_events = []

    for i, pump_id in enumerate(PUMP_IDS):
        rng = np.random.RandomState(RNG_SEED + i)
        df, failure_events_df = simulate_pump(pump_id, rng)
        all_pump_frames.append(df)
        all_failure_events.append(failure_events_df)

    full_df = pd.concat(all_pump_frames, ignore_index=True)
    full_df = full_df.sort_values(["pump_id", "timestamp"]).reset_index(drop=True)

    failure_events_df = pd.concat(all_failure_events, ignore_index=True)
    failure_events_df = failure_events_df.sort_values(["pump_id", "failure_time"]).reset_index(drop=True)

    # ---- Sanity checks ----
    n_expected = len(pd.date_range(START_DATE, END_DATE, freq=f"{FREQ_MINUTES}min"))
    counts = full_df.groupby("pump_id")["timestamp"].count()
    assert (counts == n_expected).all(), "Missing timestamps detected for some pump!"
    assert full_df.isnull().sum().sum() == 0, "Unexpected missing values in generated data!"

    failure_rate = full_df["failure_flag"].mean() * 100
    print(f"Overall failure_flag positive rate: {failure_rate:.2f}%")
    assert 1.0 <= failure_rate <= 8.0, (
        f"Failure rate {failure_rate:.2f}% is outside the expected realistic range; "
        "adjust FAILURE_PROBABILITY_PER_CYCLE or CYCLE_LEN_DAYS_RANGE."
    )

    timestamp_str = full_df.copy()
    timestamp_str["timestamp"] = timestamp_str["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_str.to_csv(SENSOR_CSV, index=False)

    failure_events_out = failure_events_df.copy()
    failure_events_out["failure_time"] = failure_events_out["failure_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    failure_events_out.to_csv(FAILURE_EVENTS_CSV, index=False)

    print(f"Saved sensor data: {SENSOR_CSV}")
    print(f"Rows: {len(full_df):,} | Pumps: {len(PUMP_IDS)} | "
          f"Date range: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Total simulated failure events: {len(failure_events_df)}")
    print(f"Saved failure event log: {FAILURE_EVENTS_CSV}")
    print(full_df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
