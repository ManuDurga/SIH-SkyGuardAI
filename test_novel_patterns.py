"""
test_novel_patterns.py
Regression check for the fusion.py redesign: feeds the pipeline synthetic
patterns that do NOT match any of the four trained archetypes, and asserts
they land in an unknown_* bucket instead of being force-fit into a known
label. Run directly:

    python test_novel_patterns.py
"""
import numpy as np
import pandas as pd

from simulate import BASE, _normal_walk
from features import build_features
from rules import apply_rules
from model import train_and_score
from explain import explain_anomalies
from fusion import fuse, KNOWN_LABELS

N = 300
FREQ = "15min"


def _heatwave(rng):
    """Temperature UP, humidity DOWN — the opposite sign of the storm/flood
    signature (which is temp/pressure down, humidity up). Should not be
    force-fit into storm or flood."""
    t = _normal_walk(N, BASE["temperature"], 0.35, rng)
    p = _normal_walk(N, BASE["pressure"], 0.6, rng)
    h = _normal_walk(N, BASE["humidity"], 1.1, rng)
    start = N // 6
    for i in range(start, N):
        prog = (i - start) / (N - start)
        t[i] = BASE["temperature"] * (1 + 0.22 * prog) + rng.normal(0, 0.4)
        h[i] = BASE["humidity"] * (1 - 0.35 * prog) + rng.normal(0, 0.8)
    return t, p, h


def _slow_drift(rng):
    """A pressure sensor that drifts steadily off-calibration — no flatline,
    no burst. Should not be force-fit into frozen or spike."""
    t = _normal_walk(N, BASE["temperature"], 0.35, rng)
    p = _normal_walk(N, BASE["pressure"], 0.6, rng)
    h = _normal_walk(N, BASE["humidity"], 1.1, rng)
    start = N // 4
    p[start:] += np.linspace(0, 14, N - start)
    return t, p, h


def build_test_frame(seed=42):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-09-01", periods=N, freq=FREQ)
    frames = []

    t, p, h = _heatwave(rng)
    frames.append(pd.DataFrame({"station_id": "TEST-HEATWAVE", "state": "TestState",
                                 "timestamp": ts, "temperature": t, "pressure": p,
                                 "humidity": h, "scenario": "novel_weather"}))

    t, p, h = _slow_drift(rng)
    frames.append(pd.DataFrame({"station_id": "TEST-DRIFT", "state": "TestState",
                                 "timestamp": ts, "temperature": t, "pressure": p,
                                 "humidity": h, "scenario": "novel_fault"}))

    t = _normal_walk(N, BASE["temperature"], 0.35, rng)
    p = _normal_walk(N, BASE["pressure"], 0.6, rng)
    h = _normal_walk(N, BASE["humidity"], 1.1, rng)
    frames.append(pd.DataFrame({"station_id": "TEST-NORMAL", "state": "TestState",
                                 "timestamp": ts, "temperature": t, "pressure": p,
                                 "humidity": h, "scenario": "normal"}))
    return pd.concat(frames, ignore_index=True)


def run():
    df = build_test_frame()
    feat = apply_rules(build_features(df))
    clf, scored = train_and_score(feat)
    explained = explain_anomalies(clf, scored)
    fused = fuse(explained)

    print(fused[["station_id", "predicted", "confidence", "severity"]].to_string(index=False))
    print()
    for _, r in fused.iterrows():
        print(f"[{r.station_id}] {r.predicted}\n  {r.rca}\n")

    heatwave_pred = fused.loc[fused.station_id == "TEST-HEATWAVE", "predicted"].iat[0]
    drift_pred = fused.loc[fused.station_id == "TEST-DRIFT", "predicted"].iat[0]

    assert heatwave_pred == "other_weather_event", (
        f"FAIL: expected heatwave-like pattern -> other_weather_event, got '{heatwave_pred}'."
    )
    assert drift_pred == "other_sensor_fault", (
        f"FAIL: expected slow pressure drift -> other_sensor_fault, got '{drift_pred}'."
    )
    print("PASS: unseen heatwave -> OTHER_WEATHER_EVENT and slow drift -> OTHER_SENSOR_FAULT.")


if __name__ == "__main__":
    run()
