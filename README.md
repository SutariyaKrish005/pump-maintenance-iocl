# Predictive Maintenance for Refinery Pumps (IOCL) using Python + ML

Predicts the risk of refinery pump failure in the next 24-72 hours from
sensor data, using engineered time-series features and a RandomForest
classifier. Uses realistic **synthetic** sensor data (no external
downloads), so the project runs fully offline and reproducibly.

## Business Problem

Unplanned pump failures in refineries cause production loss, safety
risks, and high maintenance costs. This project trains a classifier
that flags at-risk pumps 24-72 hours ahead of failure, giving
maintenance teams a window to intervene proactively.

## Project Structure

```
.
├── data/
│   ├── pump_sensor_data.csv        # raw synthetic sensor log (10-min frequency)
│   ├── pump_failure_events.csv     # ground-truth failure event timestamps
│   └── pump_features.csv           # feature-engineered dataset used for modeling
├── src/
│   ├── generate_data.py            # simulates pump sensor data + failure events
│   ├── engineer_features.py        # builds rolling/lag/baseline-relative features
│   ├── train_model.py              # trains, evaluates, plots, saves the model
│   └── predict.py                  # reloads the saved model -> test set predictions
├── figures/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── example_pump_timeline.png
├── models/
│   └── pump_failure_model.joblib
├── output/
│   ├── test_predictions.csv
│   ├── metrics_summary.txt
│   └── metrics_summary.csv
└── README.md
```

## Requirements

- Python 3.10+
- pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
- Optional: xgboost (only if you set `MODEL_TYPE = "xgboost"` in
  `train_model.py`; the script runs perfectly well without it)

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## How to Run

Run the four scripts in order from the project root:

```bash
# 1. Simulate ~6 months of 10-minute sensor readings for 8 pumps,
#    with realistic pre-failure anomaly ramps and ~19 failure events
python src/generate_data.py

# 2. Build rolling/lag/rate-of-change/baseline-relative features
python src/engineer_features.py

# 3. Time-based train/validation/test split, train RandomForest,
#    evaluate, plot, and save the model
python src/train_model.py

# 4. (Optional) Reload the saved model to regenerate test-set
#    predictions without retraining
python src/predict.py
```

All paths are relative to each script's location, so the project works
regardless of where it's cloned, as long as you run from the project
root (or preserve the folder structure above).

## Data Generation (`generate_data.py`)

Simulates 8 pumps at 10-minute resolution over ~6 months, with:

- **Normal operation**: sensor values fluctuate around pump-specific
  baselines (each pump has its own typical vibration/temperature/
  current/pressure level) with a mild daily load cycle and noise.
- **Pre-failure anomaly ramp**: starting 96 hours before each simulated
  failure, vibration and bearing temperature rise, motor current climbs,
  and suction pressure drops (cavitation-like behavior) — worsening
  nonlinearly as the failure approaches.
- **`failure_flag` labeling**: set to 1 only for readings between 24
  and 72 hours before an actual failure (per the spec) — the window
  where a maintenance alert is genuinely actionable. Readings in the
  final 24 hours before failure, and all normal operation, are labeled 0.
- **Realistic class imbalance**: ~2.7% positive rate after feature
  engineering, in the specified 2-5% range.
- Both **failure events** (unplanned, anomaly-preceded, resets runtime)
  and **scheduled preventive maintenance** (routine, no anomaly, also
  resets runtime) occur, so `runtime_hours_since_maintenance` and
  `hours_since_last_failure` (engineered later) are genuinely distinct
  signals rather than duplicates of each other.
- Guaranteed **no missing timestamps** per pump (asserted in code).

## Feature Engineering (`engineer_features.py`)

All features are computed **causally** (only past + current values,
grouped per pump), so there is no leakage from the future:

- **Rolling statistics**: 1-hour mean/std of vibration, 1-hour mean
  temperature, 6-hour max temperature, 1-hour mean current, 6-hour min
  suction pressure.
- **Rate-of-change**: vibration/temperature/current change over the
  last hour.
- **Lag features**: vibration 1 step and 6 steps (1 hour) ago.
- **Pump-relative baseline features** (the key generalization fix —
  see "Design Decisions" below): each pump's current vibration/
  temperature/suction pressure as a *ratio* to that same pump's own
  trailing 30-day baseline (computed from data more than 5 days old,
  so the baseline itself isn't contaminated by the anomaly it's meant
  to detect).
- **Operating context**: `hours_since_last_failure` (time since that
  pump's most recent actual failure, using only failures strictly
  before the current timestamp) and `maintenance_due_flag` (runtime
  since last maintenance exceeding ~29 days).
- Rows at the start of each pump's history without enough data for the
  longest rolling window are dropped (documented in the console output)
  rather than filled with fabricated values.

## Modeling (`train_model.py`)

- **Time-based split** (no shuffling): first 65% of the time range for
  training, next 10% for validation, final 25% for testing.
- **Class imbalance**: handled with `class_weight="balanced"` in
  RandomForestClassifier (`n_estimators=200`, `max_depth=8`).
- **Threshold tuning**: the default 0.5 probability cutoff is a poor
  fit for a ~3% positive rate. The classification threshold is instead
  chosen by maximizing F1 on the **validation** slice (never used to
  fit the model, and kept separate from the test set the results are
  reported on) via `precision_recall_curve`.
- **Evaluation on the untouched test set**: precision/recall/F1 for the
  failure class, ROC-AUC, and a full confusion matrix.
- **Plots**: confusion matrix heatmap, ROC curve, top-15 feature
  importances, and an example pump timeline with the labeled
  failure-risk windows shaded.
- **Persistence**: the trained model, its feature list, the
  train/validation split cutoffs, and the tuned threshold are all
  saved together in `models/pump_failure_model.joblib` so `predict.py`
  can reproduce identical predictions later without retraining.

### Optional: XGBoost

Set `MODEL_TYPE = "xgboost"` in `train_model.py` to use
`XGBClassifier` instead. If `xgboost` isn't installed, the script
prints a notice and automatically falls back to RandomForest — it
never raises an import error out of the box.

## Key Results (this synthetic dataset)

| Model | Test period | Precision (failure) | Recall (failure) | F1 (failure) | ROC-AUC |
|-------|-------------|----------------------|-------------------|--------------|---------|
| RandomForestClassifier | 2024-05-18 to 2024-06-30 | 0.80 | 0.91 | 0.85 | 0.997 |

Confusion matrix (test set): TN=48,359, FP=321, FN=131, TP=1,309.

**Interpretation**: the model correctly identifies ~91% of impending
failures (recall) with a false-alarm rate of ~0.7% among normal
operating periods — enough lead time and precision for a maintenance
team to act on the alerts without being overwhelmed by false positives.

Exact numbers will vary slightly if you change the random seed, pump
count, or date range in `generate_data.py`.

## Design Decisions Worth Knowing

- **Pump-relative baseline features were essential for generalization.**
  An earlier version of this model used only absolute sensor values and
  achieved near-perfect in-sample fit but **zero recall** on a pump that
  had no failure examples in the training period — the trees had learned
  pump-specific absolute thresholds rather than a transferable "how far
  above this pump's own normal level" signal. Adding
  `vibration_ratio_to_baseline` (and the temperature/suction equivalents)
  fixed this, and it's now the single most important feature in the
  model (see `figures/feature_importance.png`).
- **Every pump needs failure examples somewhere in the training data**
  for a tree-based model to generalize to it at all. The simulation
  parameters (8 pumps, 6 months, ~65% failure probability per operating
  cycle) were tuned so every pump has at least one failure event within
  the observation period.
- **The 24-72h label window (not 0-24h) is intentional**: readings in
  the final 24 hours before failure are labeled 0, matching the spec's
  framing of "predict failure with enough lead time to act," rather
  than flagging failures that are already too close to prevent.
- **Model persistence is lightweight by design.** Unlike statsmodels
  results objects, a fitted scikit-learn classifier pickles compactly
  on its own (~1.6 MB here), so no special compression trick was needed.

## Notes & Limitations

- This is synthetic data with an intentionally learnable signal
  (multiplicative anomaly ramps tied directly to the failure event).
  Real refinery sensor data will have noisier, more varied failure
  modes and will likely need additional feature engineering, more
  historical failure examples, and careful threshold tuning per pump
  or per failure mode.
- `hours_since_last_failure` uses a real historical failure log
  (`pump_failure_events.csv`), which is legitimate for causal historical
  features (it reflects what maintenance teams would already know at
  the time), not a leak of future information.
- The threshold is tuned once globally; a production system might tune
  per-pump or per-criticality thresholds, or use a cost-sensitive
  objective reflecting the true cost of a missed failure vs. a false
  alarm.
<img width="975" height="825" alt="roc_curve" src="https://github.com/user-attachments/assets/4a007f55-b60d-4cc5-b537-2bfefb1bfbfb" />
<img width="1350" height="1050" alt="feature_importance" src="https://github.com/user-attachments/assets/aa077d91-4120-4c02-95b2-7b3fc6f47561" />
<img width="2100" height="900" alt="example_pump_timeline" src="https://github.com/user-attachments/assets/c8b47c1a-e44d-41e1-942d-6ad035b924d0" />
<img width="900" height="750" alt="confusion_matrix" src="https://github.com/user-attachments/assets/2b0a287b-82b8-444d-8ced-f2a6575f994f" />



