"""
main.py
Runs the full SkyGuard AI pipeline end to end, mirroring the implementation
flow diagram:

  AWS Sensor Data -> Data Cleaning & Preprocessing -> Rule-Based Quality
  Checks -> AI/ML Anomaly Detection -> Temporal+Multivariate+Spatial
  Analysis -> Evidence Fusion & Classification -> Confidence+Severity+Root
  Cause -> Sensor Health Assessment -> Dashboard + Real-Time Alerts

Usage:
    python main.py
"""
import json
import sys

import pandas as pd

from simulate import generate_dataset
from features import build_features
from rules import apply_rules
from model import train_and_score
from explain import explain_anomalies
from fusion import fuse, sensor_health_summary


def run_pipeline(n_points=300, freq_minutes=15, scenarios=None, verbose=True):
    if verbose:
        print("→ Ingesting AWS sensor data ...")
    raw = generate_dataset(n_points=n_points, freq_minutes=freq_minutes, scenarios=scenarios)

    if verbose:
        print("→ Cleaning + building temporal/multivariate features ...")
    feat = build_features(raw)

    if verbose:
        print("→ Running rule-based quality checks ...")
    ruled = apply_rules(feat)

    if verbose:
        print("→ Scoring with Isolation Forest (AI/ML anomaly detection) ...")
    clf, scored = train_and_score(ruled)

    if verbose:
        print("→ Explaining anomalies with SHAP ...")
    explained = explain_anomalies(clf, scored)

    if verbose:
        print("→ Fusing evidence -> classification, confidence, severity, root cause ...")
    fused = fuse(explained)

    if verbose:
        print("→ Assessing sensor health ...")
    health = sensor_health_summary(fused)

    return {"stations": fused, "health": health, "raw": raw, "scored": explained}


def report(result):
    fused = result["stations"]
    health = result["health"]
    print("\n" + "=" * 72)
    print("SKYGUARD AI — PIPELINE OUTPUT")
    print("=" * 72)
    for _, r in fused.iterrows():
        print(f"\n[{r.station_id}] {r.state}")
        print(f"  Classification : {r.predicted.upper()}  (severity: {r.severity})")
        print(f"  Confidence     : {r.confidence}%")
        print(f"  Support        : {r.support}")
        print(f"  RCA            : {r.rca}")
        if pd.notna(r.health):
            print(f"  Sensor health  : {r.health:.0f}%")
    print("\n" + "-" * 72)
    print(f"Fleet sensor-health prediction score : {health['prediction_score']}%"
          f"  (mild={health['mild']}, severe={health['severe']})")
    if health["alert"]:
        print("  ⚠ ALERT: sensor-health prediction score <= 70% — maintenance crew notified.")
    print("=" * 72)


def export_json(result, path="pipeline_output.json"):
    fused = result["stations"].copy()
    fused["health"] = fused["health"].astype(object).where(fused["health"].notna(), None)
    payload = {
        "stations": fused.to_dict(orient="records"),
        "fleet_health": result["health"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved machine-readable output to {path}")


if __name__ == "__main__":
    result = run_pipeline()
    report(result)
    export_json(result)
