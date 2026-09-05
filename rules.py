"""
rules.py
Deterministic, physically-grounded quality checks — the "Rule-Based Quality
Checks" stage. These catch obviously invalid readings cheaply, and also
supply interpretable flags (flatline / spike / out-of-range) that the fusion
stage uses to help name *why* the ML model flagged a point.
"""
import numpy as np
import pandas as pd

PHYSICAL_RANGE = {
    "temperature": (-10, 55),   # deg C, plausible range for Indian AWS network
    "pressure": (950, 1060),    # hPa at station level
    "humidity": (0, 100),       # %
}

RATE_LIMIT = {  # max plausible change between consecutive 15-min reports
    "temperature": 3.0,
    "pressure": 6.0,
    "humidity": 12.0,
}


def apply_rules(feat: pd.DataFrame) -> pd.DataFrame:
    feat = feat.copy()
    for v, (lo, hi) in PHYSICAL_RANGE.items():
        feat[f"{v}_out_of_range"] = (~feat[v].between(lo, hi)).astype(int)

    for v, lim in RATE_LIMIT.items():
        feat[f"{v}_rate_flag"] = (feat[f"{v}_rate"].abs() > lim).astype(int)

    # a point is "rule flagged" if any variable trips flatline, rate, or range
    flag_cols = []
    for v in PHYSICAL_RANGE:
        flag_cols += [f"{v}_flatline", f"{v}_rate_flag", f"{v}_out_of_range"]
    feat["rule_flag"] = (feat[flag_cols].sum(axis=1) > 0).astype(int)
    return feat


if __name__ == "__main__":
    from simulate import generate_dataset
    from features import build_features
    df = build_features(generate_dataset())
    ruled = apply_rules(df)
    print(ruled.groupby("scenario")["rule_flag"].mean())
