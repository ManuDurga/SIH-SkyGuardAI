"""
model.py
Isolation Forest anomaly detection over the engineered features — the
"AI/ML Anomaly Detection" stage. Trained on baseline (normal) conditions,
scored across the full multi-station dataset, so it generalises to unseen
readings the way it would on live AWS feeds.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

MODEL_FEATURES = [
    "temperature_zscore", "temperature_rate", "temperature_roll_std",
    "pressure_zscore", "pressure_rate", "pressure_roll_std",
    "humidity_zscore", "humidity_rate", "humidity_roll_std",
]


def train_and_score(feat: pd.DataFrame, contamination=0.05, seed=7):
    feat = feat.copy()
    X = feat[MODEL_FEATURES].fillna(0).values

    baseline_mask = feat["scenario"].eq("normal").values
    if baseline_mask.sum() < 20:
        baseline_mask = np.ones(len(feat), dtype=bool)  # fallback: fit on everything

    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
    )
    clf.fit(X[baseline_mask])

    # decision_function: higher = more normal, lower/negative = more anomalous
    feat["if_score"] = clf.decision_function(X)
    feat["if_flag"] = (clf.predict(X) == -1).astype(int)
    # normalise anomaly score to an intuitive 0-100 "abnormality" scale
    s = feat["if_score"]
    feat["abnormality"] = ((s.max() - s) / (s.max() - s.min() + 1e-9) * 100).clip(0, 100)
    # combined evidence flag: IF catches statistical/pattern anomalies (spikes,
    # storm/flood drift); the deterministic rule flag catches flatlines that
    # IF's variance-based features can under-weight (a frozen sensor looks
    # "calm", not extreme). Fusing both is what "Evidence Fusion" means here.
    if "rule_flag" in feat.columns:
        feat["combined_flag"] = ((feat["if_flag"] == 1) | (feat["rule_flag"] == 1)).astype(int)
    else:
        feat["combined_flag"] = feat["if_flag"]
    return clf, feat


if __name__ == "__main__":
    from simulate import generate_dataset
    from features import build_features
    from rules import apply_rules

    df = apply_rules(build_features(generate_dataset()))
    _, scored = train_and_score(df)
    print(scored.groupby("scenario")[["if_flag", "abnormality"]].mean().round(2))
