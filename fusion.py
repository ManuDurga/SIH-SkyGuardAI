"""
fusion.py
Evidence Fusion & Anomaly Classification + Confidence/Severity/Root-Cause +
Sensor Health Assessment stages, combined.

DESIGN NOTE (v3): classification is now genuinely hierarchical, matching
the required decision structure:

    anomalous? -> broad type (sensor-like / weather-like / insufficient
    evidence for either) -> only within the winning broad type, check
    whether a specific subtype (frozen/spike, or storm/flood) is a strong
    enough match; otherwise fall back within that branch.

The four archetypes never compete against each other directly across
branches — "storm" and "spike" are never compared to decide the winner,
because they answer different questions (weather vs. sensor). This
replaces the v2 approach, which scored all four continuously (an
improvement over the original hard-coded if/elif) but still let all four
compete in one flat ranking, which meant a weak weather-side score could
still "win" over an even weaker sensor-side score and get reported by
name. Outputs are now:

    NORMAL
    Sensor fault:  FROZEN, SPIKE, else OTHER_SENSOR_FAULT
    Weather event: STORM, FLOOD, else OTHER_WEATHER_EVENT
    Neither branch has enough evidence: UNKNOWN_ANOMALY

DESIGN NOTE (v2, still true): the original version of this file was a hard if/elif
cascade with fixed thresholds that could only ever emit one of four
anomaly archetypes (frozen / spike / storm / flood) or "normal". Anything
that didn't cleanly fit one of those four got force-fit into the closest
one anyway, with a confidence score that implied certainty it did not
have. That's a real overfitting risk: in production there will always be
more distinct technical-sensor-fault modes and more distinct
extreme-weather patterns than the four we happened to simulate.

This version instead:
  1. Scores each archetype independently and continuously (0-100), from
     smooth similarity functions rather than boolean gates chained with
     elif. A station's evidence is compared against all four hypotheses,
     not funneled down a decision tree that stops at the first match.
  2. Requires the *winning* archetype to clear a minimum confidence floor
     AND beat the runner-up by a margin before it's reported by name. If
     nothing clears the floor, the station is reported as an explicit
     "unknown_sensor_fault" or "unknown_weather_event" (or
     "unknown_anomaly" if it's ambiguous which side it's on) instead of
     being mislabeled with false confidence.
  3. Computes physical baselines per-station from that station's own
     recent data, rather than hard-coding the synthetic simulator's base
     values (25 degC / 1013 hPa) into the classification and RCA text —
     the old code would silently misjudge any station with a different
     climate baseline (e.g. Guwahati vs. Jaipur).
"""
import numpy as np
import pandas as pd

VARS = ["temperature", "pressure", "humidity"]

KNOWN_LABELS = ["frozen", "spike", "storm", "flood"]
FAULT_LABELS = ["frozen", "spike"]
WEATHER_LABELS = ["storm", "flood"]
FALLBACK_LABELS = ["other_sensor_fault", "other_weather_event", "unknown_anomaly"]

# --- tunable gates -----------------------------------------------------
# Stage 1 (broad type): how much evidence is needed before we commit to
# "this looks sensor-like" or "this looks weather-like" at all, and how
# clearly one side has to beat the other to avoid a contested call.
BROAD_MIN_SCORE = 20
BROAD_MARGIN = 8
# Stage 2 (subtype, applied ONLY within the winning broad branch): the
# specific archetype (frozen vs spike, or storm vs flood) must clear this
# score and beat its in-branch runner-up by this margin, or the result
# falls back to OTHER_SENSOR_FAULT / OTHER_WEATHER_EVENT instead.
CONFIDENCE_FLOOR = 45
CONTESTED_MARGIN = 8
# Below this fraction-flagged / run-length combination, don't even run the
# archetype scorer -- it's isolated noise, not a pattern worth naming.
# NOTE: Isolation Forest is fit with contamination=0.05, so a truly normal
# station will still have ~5-8% of its points flagged by baseline false
# positives alone (see model.py). NOISE_FRAC sits just above that expected
# background rate so genuinely normal stations aren't escalated into an
# "unknown" bucket purely from the detector's own calibrated noise floor.
NOISE_FRAC = 0.08
NOISE_RUN_FRAC = 0.02


def _longest_run(flags: np.ndarray) -> int:
    """Longest streak of consecutive 1s anywhere in the array."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f == 1 else 0
        best = max(best, cur)
    return best


def _closeness(observed: float, expected: float) -> float:
    """
    Smooth similarity in [0, 1] between an observed magnitude and an
    archetype's reference magnitude. 1.0 at a perfect match, decaying
    smoothly (not a cliff-edge threshold) as the observed value over- or
    under-shoots the reference. Used so "storm" / "flood" match strength
    is a continuous grade, not a pass/fail check against one hard number.
    """
    if expected <= 0:
        return 0.0
    ratio = observed / expected
    return float(np.clip(1 - abs(1 - ratio), 0, 1))


def _station_baseline(g: pd.DataFrame, n: int) -> dict:
    """
    Per-station baseline for each variable, estimated from that station's
    own early readings (not a global constant) so percentage-change
    comparisons generalise across stations with different climates.
    """
    lead = max(5, int(n * 0.2))
    return {v: float(g[v].iloc[:lead].median()) for v in VARS}


def _archetype_scores(g, flagged, flags, n, baseline, run_len, late_frac,
                       flatline_frac, rate_frac, frac_flagged, last):
    """Continuous 0-100 evidence score for each of the four archetypes."""
    run_frac = run_len / n

    # --- Frozen: sustained flat variance -----------------------------
    frozen_score = 100 * (0.6 * flatline_frac +
                           0.4 * min(1.0, run_len / max(10, n * 0.15)))

    # --- Spike: rate-limit violations in short, scattered bursts -----
    # (down-weighted as run_len grows toward a large fraction of the
    # window, since a long sustained streak looks like a weather event
    # or a stuck sensor, not scattered miscalibration)
    burstiness = 1 - min(1.0, run_frac / 0.3)
    spike_score = 100 * (0.5 * rate_frac +
                          0.3 * min(1.0, frac_flagged / 0.15) +
                          0.2 * burstiness)

    # --- Shared weather-event physics --------------------------------
    # temp/pressure drop is measured against this station's own recent
    # baseline, not a hard-coded constant.
    temp_drop_pct = max(0.0, (baseline["temperature"] - last["temperature"])
                        / baseline["temperature"] * 100)
    pressure_drop_pct = max(0.0, (baseline["pressure"] - last["pressure"])
                            / baseline["pressure"] * 100)
    peak_humidity_rise = g["humidity"].max() - baseline["humidity"]
    saturation_frac = (g["humidity"] >= 88).mean()  # over the whole window

    # both temp AND pressure have to actually be trending down for this to
    # look like storm/flood physics at all -- otherwise storm/flood score
    # is gated toward zero regardless of how the humidity behaves.
    coherence_gate = min(1.0, temp_drop_pct / 2) * min(1.0, pressure_drop_pct / 1)

    # --- Storm: sharper drop, humidity peaks then recedes, shorter run
    storm_match = (0.35 * _closeness(temp_drop_pct, 15) +
                   0.25 * _closeness(pressure_drop_pct, 7) +
                   0.25 * _closeness(peak_humidity_rise, 30) +
                   0.15 * (1 - min(1.0, run_frac / 0.3)))
    storm_score = 100 * storm_match * coherence_gate

    # --- Flood: gentler drop, humidity pinned high for most of the window
    flood_match = (0.25 * _closeness(temp_drop_pct, 10) +
                   0.20 * _closeness(pressure_drop_pct, 3) +
                   0.35 * saturation_frac +
                   0.20 * min(1.0, run_frac / 0.4))
    flood_score = 100 * flood_match * coherence_gate

    return {
        "frozen": float(np.clip(frozen_score, 0, 100)),
        "spike": float(np.clip(spike_score, 0, 100)),
        "storm": float(np.clip(storm_score, 0, 100)),
        "flood": float(np.clip(flood_score, 0, 100)),
    }, temp_drop_pct, pressure_drop_pct, peak_humidity_rise



def _trend_strength(series: pd.Series) -> tuple[float, float]:
    """Return (normalised end-to-end change, smoothness) for a signal.

    Smoothness is the fraction of smoothed first-differences that agree with
    the overall trend direction. It is deliberately generic: it describes a
    sustained change without assuming that the change is a storm, flood, etc.
    """
    x = series.astype(float).to_numpy()
    if len(x) < 8 or not np.isfinite(x).all():
        return 0.0, 0.0
    k = max(3, len(x) // 10)
    start = float(np.median(x[:k]))
    end = float(np.median(x[-k:]))
    scale = max(abs(start), 1e-6)
    change = (end - start) / scale

    smooth = pd.Series(x).rolling(5, min_periods=1).mean().to_numpy()
    diffs = np.diff(smooth)
    direction = np.sign(end - start)
    if direction == 0 or len(diffs) == 0:
        return change, 0.0
    meaningful = diffs[np.abs(diffs) > max(np.std(diffs) * 0.10, 1e-9)]
    if len(meaningful) == 0:
        return change, 0.0
    smoothness = float((np.sign(meaningful) == direction).mean())
    return change, smoothness


def _broad_evidence(g: pd.DataFrame, flagged: pd.DataFrame, flags: np.ndarray,
                    n: int, flatline_frac: float, rate_frac: float) -> tuple[float, float, dict]:
    """Independent broad evidence for sensor-like vs weather-like behaviour.

    IMPORTANT: these scores do *not* use the Frozen/Spike/Storm/Flood subtype
    scores.  That prevents an unseen weather pattern from becoming a sensor
    fault merely because it resembles Storm/Flood poorly (and vice versa).
    """
    run_frac = _longest_run(flags) / max(n, 1)
    frac_flagged = float(flags.mean())

    # Hard/near-hard sensor evidence from generic QC behaviour.
    out_cols = [f"{v}_out_of_range" for v in VARS if f"{v}_out_of_range" in g.columns]
    rate_cols = [f"{v}_rate_flag" for v in VARS if f"{v}_rate_flag" in g.columns]
    flat_cols = [f"{v}_flatline" for v in VARS if f"{v}_flatline" in g.columns]
    out_frac = float(g[out_cols].max(axis=1).mean()) if out_cols else 0.0
    rate_all = float(g[rate_cols].max(axis=1).mean()) if rate_cols else rate_frac
    flat_all = float(g[flat_cols].max(axis=1).mean()) if flat_cols else flatline_frac

    # Generic sustained trends, independent of named weather archetypes.
    changes = {}
    smoothness = {}
    for v in VARS:
        ch, sm = _trend_strength(g[v])
        changes[v] = ch
        smoothness[v] = sm

    # Significance thresholds are only for broad behavioural evidence, not
    # for naming a weather type. Humidity uses absolute %-points because a
    # relative percentage can exaggerate changes at low humidity.
    hum_start = float(np.median(g["humidity"].iloc[:max(3, n // 10)]))
    hum_end = float(np.median(g["humidity"].iloc[-max(3, n // 10):]))
    humidity_points = hum_end - hum_start
    significant = {
        "temperature": abs(changes["temperature"]) >= 0.04,
        "pressure": abs(changes["pressure"]) >= 0.01,
        "humidity": abs(humidity_points) >= 8.0,
    }
    significant_vars = [v for v, ok in significant.items() if ok]
    n_sig = len(significant_vars)
    mean_smooth = float(np.mean([smoothness[v] for v in significant_vars])) if significant_vars else 0.0

    # Weather-like: sustained, smooth, multivariable atmospheric movement,
    # especially when hard sensor-fault indicators are absent.
    persistence = min(1.0, 0.55 * (frac_flagged / 0.20) + 0.45 * (run_frac / 0.20))
    multivar = min(1.0, n_sig / 2.0)  # two changing variables are meaningful
    clean_sensor_behaviour = 1.0 - min(1.0, flat_all * 3 + rate_all * 2 + out_frac * 5)
    weather_score = 100 * (
        0.30 * persistence
        + 0.30 * multivar
        + 0.25 * mean_smooth
        + 0.15 * clean_sensor_behaviour
    )
    # Do not call a one-variable trend weather-like solely because IF flagged
    # it. Real weather evidence should normally involve multiple variables.
    if n_sig < 2:
        weather_score *= 0.55

    # Sensor-like: flatline, abrupt rate/range violations, or a persistent
    # smooth drift isolated to one variable while the other variables remain
    # comparatively stable. Linear-fit R² is useful here because a slow
    # calibration drift can be highly directional without producing any
    # single large step/rate violation.
    linear_trends = {}
    strong_linear = []
    x = np.arange(n, dtype=float)
    for v in VARS:
        y = g[v].astype(float).to_numpy()
        if n < 8 or not np.isfinite(y).all() or np.allclose(y, y[0]):
            linear_trends[v] = {"change_pct": 0.0, "r2": 0.0}
            continue
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 0.0 if ss_tot <= 1e-12 else max(0.0, 1 - ss_res / ss_tot)
        base = max(abs(float(np.mean(y[:max(5, n // 10)]))), 1e-6)
        change_pct = float(slope * (n - 1) / base * 100)
        linear_trends[v] = {"change_pct": change_pct, "r2": r2}
        threshold = 0.8 if v == "pressure" else 4.0
        if r2 >= 0.70 and abs(change_pct) >= threshold:
            strong_linear.append(v)

    # Require at least two independently smooth/strong variables before the
    # generic WEATHER score is allowed to become strong. This suppresses
    # random-walk endpoint drift in otherwise normal data, while retaining
    # genuine multivariable events such as heatwaves/storms/floods.
    if (len(strong_linear) >= 2
            or (n_sig >= 2 and frac_flagged >= 0.20 and clean_sensor_behaviour >= 0.70)):
        # Smooth multivariable trend (e.g. heatwave) OR a strongly persistent
        # multivariable event (e.g. flood, whose humidity may plateau rather
        # than remain linear) is valid broad weather evidence.
        weather_score *= 1.0
    elif len(strong_linear) == 1:
        weather_score *= 0.35
    else:
        weather_score *= 0.15

    isolated_drift = 0.0
    if len(strong_linear) == 1:
        v = strong_linear[0]
        isolated_drift = min(1.0, 0.65 + 0.35 * linear_trends[v]["r2"])

    sensor_score = 100 * (
        0.30 * min(1.0, flat_all * 4)
        + 0.22 * min(1.0, rate_all * 5)
        + 0.12 * min(1.0, out_frac * 8)
        + 0.36 * isolated_drift
    )

    detail = {
        "flatline_fraction": flat_all,
        "rate_violation_fraction": rate_all,
        "out_of_range_fraction": out_frac,
        "significant_trend_variables": significant_vars,
        "trend_smoothness": mean_smooth,
        "isolated_drift": isolated_drift,
        "linear_trends": linear_trends,
        "temperature_change_pct": changes["temperature"] * 100,
        "pressure_change_pct": changes["pressure"] * 100,
        "humidity_change_points": humidity_points,
    }
    return float(np.clip(sensor_score, 0, 100)), float(np.clip(weather_score, 0, 100)), detail

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
            "health": 98, "scores": {k: 0.0 for k in KNOWN_LABELS},
        }

    flags = g[flag_col].to_numpy()
    flatline_frac = flagged[[f"{v}_flatline" for v in VARS]].max(axis=1).mean()
    rate_frac = flagged[[f"{v}_rate_flag" for v in VARS]].max(axis=1).mean()
    run_len = _longest_run(flags)
    late_frac = flags[n // 2:].mean()
    frac_flagged = flags.mean()
    mean_abnormality = flagged["abnormality"].mean()

    valid_root = flagged["root_cause_var"].dropna()
    valid_root = valid_root[valid_root.isin(VARS)]
    root_modes = valid_root.mode()
    root_var = root_modes.iat[0] if not root_modes.empty else "humidity"
    last = g.iloc[-1]

    # Compute broad evidence before the noise shortcut so a slow, smooth
    # calibration drift is not dismissed merely because few individual
    # points cross the IF/rule threshold.
    fault_evidence, weather_evidence, broad_detail = _broad_evidence(
        g, flagged, flags, n, flatline_frac, rate_frac
    )

    # --- isolated noise: not enough evidence to name any pattern --------
    # Keep the shortcut only when there is no independent broad evidence.
    if (frac_flagged < NOISE_FRAC and run_len < max(3, int(n * NOISE_RUN_FRAC))
            and fault_evidence < BROAD_MIN_SCORE and weather_evidence < BROAD_MIN_SCORE):
        return {
            "station_id": station, "state": state, "predicted": "normal",
            "confidence": 88, "support": f"{len(flagged)} of {n} intervals flagged ({run_len} consecutive)",
            "severity": "none",
            "rca": (f"{len(flagged)} isolated readings brushed the anomaly threshold but showed no "
                    f"sustained pattern; within expected noise."),
            "health": 95, "scores": {k: 0.0 for k in KNOWN_LABELS},
        }

    baseline = _station_baseline(g, n)
    scores, temp_drop_pct, pressure_drop_pct, peak_humidity_rise = _archetype_scores(
        g, flagged, flags, n, baseline, run_len, late_frac,
        flatline_frac, rate_frac, frac_flagged, last,
    )

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    # === Stage 1: broad type — decide sensor-like vs weather-like vs   ===
    # === neither, BEFORE looking at which specific archetype matches.  ===
    if fault_evidence < BROAD_MIN_SCORE and weather_evidence < BROAD_MIN_SCORE:
        broad = "insufficient"
    elif fault_evidence >= weather_evidence + BROAD_MARGIN:
        broad = "sensor"
    elif weather_evidence >= fault_evidence + BROAD_MARGIN:
        broad = "weather"
    else:
        broad = "insufficient"  # contested between sensor-like and weather-like

    # === Stage 2: within the winning broad type only, check whether a  ===
    # === specific subtype is a strong enough match.                   ===
    if broad == "insufficient":
        predicted = "unknown_anomaly"
        best_score, margin = max(fault_evidence, weather_evidence), 0.0
    elif broad == "sensor":
        if scores["frozen"] >= scores["spike"]:
            sub_label, best_score, second_score = "frozen", scores["frozen"], scores["spike"]
        else:
            sub_label, best_score, second_score = "spike", scores["spike"], scores["frozen"]
        margin = best_score - second_score
        predicted = sub_label if (best_score >= CONFIDENCE_FLOOR and margin >= CONTESTED_MARGIN) \
            else "other_sensor_fault"
    else:  # broad == "weather"
        if scores["storm"] >= scores["flood"]:
            sub_label, best_score, second_score = "storm", scores["storm"], scores["flood"]
        else:
            sub_label, best_score, second_score = "flood", scores["flood"], scores["storm"]
        margin = best_score - second_score
        predicted = sub_label if (best_score >= CONFIDENCE_FLOOR and margin >= CONTESTED_MARGIN) \
            else "other_weather_event"

    support = (f"{len(flagged)} of {n} intervals flagged ({run_len} consecutive) — "
               f"broad evidence: sensor {fault_evidence:.0f}, weather {weather_evidence:.0f}; "
               f"subtype match: frozen {scores['frozen']:.0f}, spike {scores['spike']:.0f}, "
               f"storm {scores['storm']:.0f}, flood {scores['flood']:.0f}")

    if predicted in KNOWN_LABELS:
        confidence = int(np.clip(45 + best_score * 0.45 + margin * 0.3, 40, 98))
    elif predicted == "other_sensor_fault":
        # confident this is broadly sensor-like (that's what got it into
        # this branch), just not confident which specific fault it is.
        confidence = int(np.clip(40 + fault_evidence * 0.4, 35, 80))
    elif predicted == "other_weather_event":
        confidence = int(np.clip(40 + weather_evidence * 0.4, 35, 80))
    else:  # unknown_anomaly -- not even confident which broad type it is
        confidence = int(np.clip(25 + max(fault_evidence, weather_evidence) * 0.3, 20, 60))

    if predicted == "frozen":
        frozen_val = flagged[root_var].iloc[-1]
        rca = (f"{root_var.capitalize()} remained effectively unchanged with ~0 rolling variance across "
               f"{run_len} consecutive intervals (held a fixed value of {frozen_val:.1f}) while normal "
               f"sensor variation disappeared, indicating a possible frozen sensor "
               f"(match score {best_score:.0f}/100).")
        severity = "high" if flatline_frac > 0.6 else "moderate"
        health = int(np.clip(100 - flatline_frac * 80, 5, 95))

    elif predicted == "spike":
        rca = (f"{root_var.capitalize()} shows abrupt bidirectional excursions exceeding physically "
               f"plausible rate-of-change limits in {len(flagged)} intervals — consistent with sensor "
               f"miscalibration or interference rather than a real event (match score {best_score:.0f}/100).")
        severity = "high" if rate_frac > 0.6 else "moderate"
        health = int(np.clip(100 - rate_frac * 70, 5, 95))

    elif predicted == "storm":
        rca = (f"Pressure down {pressure_drop_pct:.1f}% and temperature down {temp_drop_pct:.1f}% from "
               f"this station's recent baseline, with humidity peaking {peak_humidity_rise:.0f} points "
               f"above baseline before receding — joint pressure-drop / humidity-spike / cooling "
               f"signature consistent with active convective storm development "
               f"(match score {best_score:.0f}/100).")
        severity = "moderate"
        health = None

    elif predicted == "flood":
        rca = (f"Humidity pinned in a high-saturation band for {run_len} consecutive intervals, with a "
               f"sustained {pressure_drop_pct:.1f}% pressure deficit and {temp_drop_pct:.1f}% cooler "
               f"temperatures relative to baseline — duration and stability point to prolonged flooding "
               f"risk rather than a short-lived event (match score {best_score:.0f}/100).")
        severity = "high" if run_len >= n * 0.6 else "moderate"
        health = None

    elif predicted == "other_sensor_fault":
        drift_note = "; isolated gradual drift evidence present" if broad_detail["isolated_drift"] > 0 else ""
        rca = (f"Abnormal sensor-like behaviour was detected (sensor-evidence "
               f"{fault_evidence:.0f}/100{drift_note}), but the temporal pattern does not sufficiently "
               f"match the supported Frozen or Spike signatures (best subtype match "
               f"{best_score:.0f}/100, {CONFIDENCE_FLOOR} required). It is reported as an unclassified "
               f"sensor fault rather than forced into a known type. Recommend manual review.")
        severity = "unconfirmed"
        health = int(np.clip(100 - fault_evidence * 0.6, 10, 95))

    elif predicted == "other_weather_event":
        rca = (f"A sustained, weather-like multivariable anomaly was detected "
               f"(temperature change {broad_detail['temperature_change_pct']:+.1f}%, "
               f"pressure change {broad_detail['pressure_change_pct']:+.1f}%, "
               f"humidity change {broad_detail['humidity_change_points']:+.1f} points; "
               f"weather-evidence {weather_evidence:.0f}/100), but it does not sufficiently match "
               f"the currently supported Storm or Flood signatures (best subtype match "
               f"{best_score:.0f}/100, {CONFIDENCE_FLOOR} required). It is therefore reported as an "
               f"unclassified weather event rather than mislabeled. Recommend manual review.")
        severity = "unconfirmed"
        health = None

    else:  # unknown_anomaly
        rca = (f"{len(flagged)} intervals were flagged as anomalous by the rule/ML layer, but available "
               f"evidence is insufficient to reliably determine whether this represents a weather event "
               f"or a sensor fault (fault-evidence {fault_evidence:.0f}/100, weather-evidence "
               f"{weather_evidence:.0f}/100 — neither clears the {BROAD_MIN_SCORE} floor by a clear "
               f"{BROAD_MARGIN}-point margin). This could be a genuinely novel failure mode or event "
               f"type outside the current archetype set. Recommend manual review before dismissing or "
               f"actioning.")
        severity = "unconfirmed"
        health = None

    return {
        "station_id": station, "state": state, "predicted": predicted,
        "confidence": confidence, "support": support, "severity": severity,
        "rca": rca, "health": health, "scores": scores,
        "broad_evidence": {"sensor": fault_evidence, "weather": weather_evidence, **broad_detail},
    }


def fuse(feat: pd.DataFrame) -> pd.DataFrame:
    rows = [classify_station(g) for _, g in feat.groupby("station_id", sort=False)]
    return pd.DataFrame(rows)


def sensor_health_summary(fused: pd.DataFrame) -> dict:
    # health-tracked rows now include the "other_sensor_fault" fallback
    # bucket too, since those stations still warrant equipment follow-up
    # even though we couldn't name the exact fault mode.
    fault_rows = fused[fused["predicted"].isin(["frozen", "spike", "other_sensor_fault"])
                        & fused["health"].notna()]
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
