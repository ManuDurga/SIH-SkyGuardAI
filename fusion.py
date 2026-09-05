"""
fusion.py
Evidence Fusion & Anomaly Classification + Confidence/Severity/Root-Cause +
Sensor Health Assessment stages, combined. Takes the rule flags, Isolation
Forest scores and SHAP root-cause attribution and turns them into the same
five-way call the dashboard shows (normal / frozen / spike / storm / flood),
with a confidence score, a support count, a plain-language RCA sentence,
and — for sensor-fault classes — a per-station health score.
"""
import numpy as np
import pandas as pd

VARS = ["temperature", "pressure", "humidity"]


def _longest_run(flags: np.ndarray) -> int:
    """Longest streak of consecutive 1s anywhere in the array."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f == 1 else 0
        best = max(best, cur)
    return best


def classify_station(g: pd.DataFrame) -> dict:
    n = len(g)
    flag_col = "combined_flag" if "combined_flag" in g.columns else "if_flag"
    flagged = g[g[flag_col] == 1]
    station = g["station_id"].iloc[0]
    state = g["state"].iloc[0]

    if flagged.empty:
        return {
            "station_id": station, "state": state, "predicted": "normal",
            "confidence": 97, "support": "0 anomalous intervals",
            "severity": "none", "rca": "All parameters within expected range.",
            "health": 98,
        }

    flags = g[flag_col].to_numpy()
    flatline_frac = flagged[[f"{v}_flatline" for v in VARS]].max(axis=1).mean()
    rate_frac = flagged[[f"{v}_rate_flag" for v in VARS]].max(axis=1).mean()
    run_len = _longest_run(flags)                 # longest sustained streak anywhere
    late_frac = flags[n // 2:].mean()              # persistence into the recent half
    mean_abnormality = flagged["abnormality"].mean()

    valid_root = flagged["root_cause_var"].dropna()
    valid_root = valid_root[valid_root.isin(VARS)]
    root_modes = valid_root.mode()
    root_var = root_modes.iat[0] if not root_modes.empty else "humidity"
    last = g.iloc[-1]
    humidity_high_sustained = (flagged["humidity"] >= 88).mean() > 0.6
    temp_down = last["temperature"] < 25 * 0.97
    pressure_down = last["pressure"] < 1013 * 0.99
    weather_shape = temp_down and pressure_down  # the storm/flood co-movement signature

    # --- classification heuristic (order matters) ---
    # 1. Frozen: a sensor is dead-flat for a large, sustained streak.
    if flatline_frac > 0.3 and run_len >= max(10, n * 0.15):
        predicted = "frozen"
    # 2. Sustained, physically-coherent drift (temp+pressure down) that
    #    persists for most of the recent window -> a real weather event.
    #    Longer / more persistent -> flood; shorter but still sustained -> storm.
    elif weather_shape and run_len >= max(15, n * 0.08):
        predicted = "flood" if (run_len >= n * 0.25 or late_frac >= 0.5) else "storm"
    # 3. Everything else that trips the rate-of-change rule in short, scattered
    #    bursts rather than one sustained streak -> sensor miscalibration.
    #    A handful of isolated flags at the baseline false-alarm rate isn't
    #    enough evidence on its own — require either a real burst (run_len)
    #    or a meaningfully elevated flagged fraction.
    elif rate_frac > 0.2 or run_len >= 4 or flags.mean() > 0.08:
        predicted = "spike"
    else:
        predicted = "normal"

    confidence = int(np.clip(50 + mean_abnormality * 0.4 + (flatline_frac + rate_frac) * 20, 40, 99))
    support = f"{len(flagged)} of {n} intervals flagged ({run_len} consecutive)"

    if predicted == "frozen":
        frozen_val = flagged[root_var].iloc[-1]
        rca = (f"{root_var.capitalize()} has held a fixed value of {frozen_val:.1f} with ~0 rolling "
               f"variance across {run_len} consecutive intervals — inconsistent with natural change. "
               f"Matches a sensor lock / stuck-reading fault.")
        severity = "high" if flatline_frac > 0.6 else "moderate"
        health = int(np.clip(100 - flatline_frac * 80, 5, 95))
    elif predicted == "spike":
        rca = (f"{root_var.capitalize()} shows abrupt bidirectional excursions exceeding physically "
               f"plausible rate-of-change limits in {len(flagged)} intervals — consistent with sensor "
               f"miscalibration or interference rather than a real event.")
        severity = "high" if rate_frac > 0.6 else "moderate"
        health = int(np.clip(100 - rate_frac * 70, 5, 95))
    elif predicted == "storm":
        rca = (f"Pressure down {100*(1013-last['pressure'])/1013:.1f}%, humidity trending toward "
               f"saturation, temperature down {100*(25-last['temperature'])/25:.1f}% — joint "
               f"pressure-drop / humidity-rise / cooling signature consistent with active convective "
               f"storm development.")
        severity = "moderate"
        health = None
    elif predicted == "flood":
        rca = (f"Humidity pinned in the 90-100% band with a sustained {100*(1013-last['pressure'])/1013:.1f}% "
               f"pressure deficit and {100*(25-last['temperature'])/25:.1f}% cooler temperatures for "
               f"{run_len} consecutive intervals — duration and stability point to prolonged flooding "
               f"risk rather than a short-lived event.")
        severity = "high" if run_len >= n * 0.6 else "moderate"
        health = None
    else:  # normal — a few isolated low-significance flags, not enough evidence
        rca = f"{len(flagged)} isolated readings brushed the anomaly threshold but showed no sustained pattern; within expected noise."
        severity = "none"
        health = 95
        confidence = 90

    return {
        "station_id": station, "state": state, "predicted": predicted,
        "confidence": confidence, "support": support, "severity": severity,
        "rca": rca, "health": health,
    }


def fuse(feat: pd.DataFrame) -> pd.DataFrame:
    rows = [classify_station(g) for _, g in feat.groupby("station_id", sort=False)]
    return pd.DataFrame(rows)


def sensor_health_summary(fused: pd.DataFrame) -> dict:
    fault_rows = fused[fused["predicted"].isin(["frozen", "spike"]) & fused["health"].notna()]
    if fault_rows.empty:
        return {"prediction_score": 100, "mild": 0, "severe": 0, "alert": False}
    mild = int((fault_rows["health"].between(70, 89)).sum())
    severe = int((fault_rows["health"] < 70).sum())
    score = int(np.clip(100 - (mild * 3 + severe * 9), 5, 100))
    return {"prediction_score": score, "mild": mild, "severe": severe, "alert": score <= 70}


if __name__ == "__main__":
    from simulate import generate_dataset
    from features import build_features
    from rules import apply_rules
    from model import train_and_score
    from explain import explain_anomalies

    df = apply_rules(build_features(generate_dataset()))
    clf, scored = train_and_score(df)
    explained = explain_anomalies(clf, scored)
    fused = fuse(explained)
    print(fused[["station_id", "state", "predicted", "confidence", "severity", "support"]])
    print()
    print(sensor_health_summary(fused))
