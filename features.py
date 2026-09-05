"""
features.py
Temporal + multivariate feature engineering per station, feeding both the
rule-based quality checks and the Isolation Forest model. This is the
"Data Cleaning & Preprocessing" + "Temporal / Multivariate / Spatial
Analysis" stage of the pipeline.
"""
import numpy as np
import pandas as pd

VARS = ["temperature", "pressure", "humidity"]
WINDOW = 6  # rolling window, in points


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    out_frames = []

    for station, g in df.groupby("station_id", sort=False):
        g = g.copy()
        for v in VARS:
            roll_mean = g[v].rolling(WINDOW, min_periods=2).mean()
            roll_std = g[v].rolling(WINDOW, min_periods=2).std().fillna(0)
            g[f"{v}_roll_std"] = roll_std
            g[f"{v}_zscore"] = ((g[v] - roll_mean) / roll_std.replace(0, np.nan)).fillna(0)
            g[f"{v}_rate"] = g[v].diff().fillna(0)
            # flatline signal: rolling std ~ 0 over the window (dead sensor)
            g[f"{v}_flatline"] = (roll_std < 1e-6).astype(int)
        out_frames.append(g)

    feat = pd.concat(out_frames, ignore_index=True)
    return feat


FEATURE_COLUMNS = [f"{v}_{suffix}" for v in VARS
                   for suffix in ("roll_std", "zscore", "rate", "flatline")]


if __name__ == "__main__":
    from simulate import generate_dataset
    df = generate_dataset()
    feat = build_features(df)
    print(feat[["station_id", "timestamp"] + FEATURE_COLUMNS].tail(10))
