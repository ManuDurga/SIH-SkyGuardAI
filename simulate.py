"""
simulate.py
Generates synthetic Automatic Weather Station (AWS) time-series data for
temperature, pressure and humidity across multiple stations, covering the
five scenario classes the dashboard supports: normal, frozen, spike, storm,
flood. This stands in for the real AWS_Sensor_Data feed in the pipeline
diagram.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

STATIONS = [
    ("AWS-PB-07", "Punjab"),
    ("AWS-HP-12", "Himachal Pradesh"),
    ("AWS-UK-04", "Uttarakhand"),
    ("AWS-UP-31", "Uttar Pradesh"),
    ("AWS-RJ-22", "Rajasthan"),
]

BASE = {"temperature": 25.0, "pressure": 1013.0, "humidity": 55.0}


def _normal_walk(n, base, step, rng):
    vals = np.empty(n)
    v = base
    for i in range(n):
        v += rng.normal(0, step) + (base - v) * 0.03
        vals[i] = v
    return vals


def _make_series(n, scenario, rng):
    """Return dict of arrays temperature/pressure/humidity/label for one station."""
    t = _normal_walk(n, BASE["temperature"], 0.35, rng)
    p = _normal_walk(n, BASE["pressure"], 0.6, rng)
    h = _normal_walk(n, BASE["humidity"], 1.1, rng)
    label = np.array(["normal"] * n, dtype=object)

    if scenario == "normal":
        pass

    elif scenario == "frozen":
        start = n // 3
        length = n - start
        frozen_series = rng.choice(["temperature", "pressure", "humidity"],
                                    size=rng.integers(1, 3), replace=False)
        arrs = {"temperature": t, "pressure": p, "humidity": h}
        for s in frozen_series:
            freeze_val = arrs[s][start]
            arrs[s][start:] = freeze_val
        label[start:] = "frozen"

    elif scenario == "spike":
        spike_series = rng.choice(["temperature", "pressure", "humidity"],
                                   size=rng.integers(1, 3), replace=False)
        arrs = {"temperature": t, "pressure": p, "humidity": h}
        mag = {"temperature": 9, "pressure": 18, "humidity": 35}
        for s in spike_series:
            idxs = rng.choice(np.arange(n // 4, n), size=max(3, n // 12), replace=False)
            arrs[s][idxs] += rng.choice([-1, 1], size=len(idxs)) * mag[s] * rng.uniform(0.6, 1.4, len(idxs))
            label[idxs] = "spike"

    elif scenario == "storm":
        start = n // 6
        dur = n - start
        for i in range(start, n):
            prog = (i - start) / dur
            t[i] = BASE["temperature"] * (1 - 0.15 * prog) + rng.normal(0, 0.4)
            p[i] = BASE["pressure"] * (1 - 0.07 * prog) + rng.normal(0, 0.8)
            if prog <= 0.5:
                h[i] = 55 + (95 - 55) * (prog / 0.5) + rng.normal(0, 0.8)
            else:
                h[i] = 95 - 0.30 * 95 * ((prog - 0.5) / 0.5) + rng.normal(0, 0.8)
        label[start:] = "storm"

    elif scenario == "flood":
        start = n // 8
        for i in range(start, n):
            ramp = min(1.0, (i - start) / max(1, (n - start) * 0.15))
            t[i] = BASE["temperature"] * (1 - 0.10 * ramp) + rng.normal(0, 0.3)
            p[i] = BASE["pressure"] * (1 - 0.03 * ramp) + rng.normal(0, 0.6)
            h[i] = 90 + rng.uniform(0, 10)
        label[start:] = "flood"

    return {"temperature": t, "pressure": p, "humidity": h, "label": label}


def generate_dataset(n_points=300, freq_minutes=15, scenarios=None, seed=7):
    """
    Build a labelled multi-station dataset.
    scenarios: optional dict station_code -> scenario name. Defaults to
    spreading the 5 scenarios across the 5 stations, one each, so the
    resulting dataset always contains a working example of every class.
    """
    rng = np.random.default_rng(seed)
    if scenarios is None:
        scenarios = {
            "AWS-PB-07": "normal",
            "AWS-HP-12": "frozen",
            "AWS-UK-04": "spike",
            "AWS-UP-31": "storm",
            "AWS-RJ-22": "flood",
        }

    start_ts = pd.Timestamp("2026-09-01 00:00:00")
    ts_index = pd.date_range(start_ts, periods=n_points, freq=f"{freq_minutes}min")

    frames = []
    for code, state in STATIONS:
        scenario = scenarios.get(code, "normal")
        series = _make_series(n_points, scenario, rng)
        frames.append(pd.DataFrame({
            "station_id": code,
            "state": state,
            "timestamp": ts_index,
            "temperature": series["temperature"],
            "pressure": series["pressure"],
            "humidity": series["humidity"],
            "true_label": series["label"],
            "scenario": scenario,
        }))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = generate_dataset()
    print(df.groupby("station_id")["true_label"].value_counts())
    df.to_csv("synthetic_aws_data.csv", index=False)
    print(f"\nSaved {len(df)} rows to synthetic_aws_data.csv")
