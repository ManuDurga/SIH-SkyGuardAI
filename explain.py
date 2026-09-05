"""
explain.py
SHAP explainability layer — the "SHAP: provides explainable reasons for
detected anomalies" stage. We explain the model's decision_function output
(higher = more normal) using a permutation explainer against a background
of baseline/normal readings, then roll the per-feature attributions up to
per-variable (temperature/pressure/humidity) attribution, which is what the
root-cause narrative in the dashboard is built from.
"""
import numpy as np
import pandas as pd
import shap

from model import MODEL_FEATURES

VARS = ["temperature", "pressure", "humidity"]


def explain_anomalies(clf, feat: pd.DataFrame, background_size=40, max_explain_per_station=40, seed=7):
    rng = np.random.default_rng(seed)
    feat = feat.copy()
    flag_col = "combined_flag" if "combined_flag" in feat.columns else "if_flag"

    X_all = feat[MODEL_FEATURES].fillna(0).values
    baseline_idx = feat.index[feat["scenario"].eq("normal")].to_numpy()
    if len(baseline_idx) < 5:
        baseline_idx = feat.index.to_numpy()
    bg_idx = rng.choice(baseline_idx, size=min(background_size, len(baseline_idx)), replace=False)
    background = X_all[bg_idx]

    # cap the number of rows explained *per station* rather than globally,
    # so every anomalous station gets some SHAP-attributed evidence instead
    # of a few noisy stations eating the whole global budget.
    anomaly_idx = []
    for _, g in feat.groupby("station_id", sort=False):
        idx = g.index[g[flag_col] == 1].to_numpy()
        if len(idx) > max_explain_per_station:
            idx = rng.choice(idx, size=max_explain_per_station, replace=False)
        anomaly_idx.extend(idx.tolist())
    anomaly_idx = np.array(anomaly_idx)

    if len(anomaly_idx) == 0:
        for v in VARS:
            feat[f"{v}_contribution"] = 0.0
        feat["root_cause_var"] = None
        return feat

    feat["root_cause_var"] = None

    explainer = shap.Explainer(clf.decision_function, background, algorithm="permutation")
    to_explain = X_all[anomaly_idx]
    sv = explainer(to_explain, max_evals=2 * len(MODEL_FEATURES) + 3)
    shap_vals = np.abs(sv.values)  # (n_rows, n_features)

    # roll up 3 features per variable into one attribution score per variable
    for v in VARS:
        cols = [i for i, c in enumerate(MODEL_FEATURES) if c.startswith(v)]
        contrib = shap_vals[:, cols].sum(axis=1)
        feat.loc[anomaly_idx, f"{v}_contribution"] = contrib
    for v in VARS:
        if f"{v}_contribution" not in feat.columns:
            feat[f"{v}_contribution"] = 0.0
        feat[f"{v}_contribution"] = feat[f"{v}_contribution"].fillna(0.0)

    contrib_matrix = feat.loc[anomaly_idx, [f"{v}_contribution" for v in VARS]].values
    top_var_idx = contrib_matrix.argmax(axis=1)
    feat.loc[anomaly_idx, "root_cause_var"] = [VARS[i] for i in top_var_idx]
    return feat


if __name__ == "__main__":
    from simulate import generate_dataset
    from features import build_features
    from rules import apply_rules
    from model import train_and_score

    df = apply_rules(build_features(generate_dataset()))
    clf, scored = train_and_score(df)
    explained = explain_anomalies(clf, scored)
    print(explained.loc[explained.if_flag == 1,
          ["station_id", "scenario", "root_cause_var", "temperature_contribution",
           "pressure_contribution", "humidity_contribution"]].head(15))
