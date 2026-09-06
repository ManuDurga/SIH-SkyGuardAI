# SkyGuard AI — working anomaly-detection model

A runnable implementation of the pipeline from the architecture diagram:

```
AWS Sensor Data → Data Cleaning & Preprocessing → Rule-Based Quality Checks
→ AI/ML Anomaly Detection → Temporal + Multivariate + Spatial Analysis
→ Evidence Fusion & Anomaly Classification → Confidence + Severity + Root Cause
→ Sensor Health Assessment → Dashboard + Real-Time Alerts
```

It's not a mockup — it simulates realistic AWS temperature/pressure/humidity
data for 7 stations (Dehradun, Jaipur, Thiruvananthapuram, Delhi, Bhopal,
Patna, Guwahati), spread across the five scenario classes (2× normal, 1×
frozen, 1× spike, 1× storm, 2× flood — so you also get to see two
independent floods and two normals classified correctly at once), then
actually detects and classifies each one from the raw signal — no
ground-truth labels are used at inference time. Change which station gets
which scenario via the `scenarios` dict at the top of
`simulate.generate_dataset()`, or pass your own `scenarios={...}` mapping.

## Files

| File | Pipeline stage |
|---|---|
| `simulate.py` | Synthetic AWS sensor feed (stand-in for the real feed) |
| `features.py` | Rolling z-score / rate-of-change / flatline features per variable |
| `rules.py` | Deterministic quality checks (physical range, rate limits, flatline) |
| `model.py` | Isolation Forest, trained on baseline-normal data, scored on everything |
| `explain.py` | SHAP (permutation explainer) attributes each anomaly to temperature / pressure / humidity |
| `fusion.py` | Combines rule flags + IF score + SHAP root cause into a classification, confidence, severity, RCA sentence, and per-station sensor health |
| `main.py` | Orchestrates the full pipeline and prints/exports a report |
| `app.py` | FastAPI wrapper exposing `/analyze` |

## Run it

```bash
pip install -r requirements.txt
python main.py                 # runs the pipeline once, prints a report,
                                # writes pipeline_output.json

uvicorn app:app --reload       # or serve it — GET /analyze for live JSON
```

## How the classification actually works

- **Frozen** is caught mainly by the *rule* layer: a rolling standard
  deviation of ~0 for a long, sustained streak. Isolation Forest alone
  under-weights this because "very calm" doesn't look statistically extreme
  in z-score/rate space — this is a good example of why the rule-based layer
  exists alongside the ML layer, not instead of it.
- **Spike** shows up as short, scattered bursts that trip the rate-of-change
  rule and the IF score, but never form one sustained streak.
- **Storm / Flood** both show the same physical signature (pressure down,
  temperature down, humidity up toward saturation) — they're told apart by
  *how long and how completely* that signature persists: a short sustained
  run → storm, a longer one covering most of the recent window → flood.
  This boundary is inherently fuzzy (that's realistic — the difference
  really is duration, not shape), so expect occasional storm/flood
  disagreement at the margin; frozen/spike/normal are clean.
- **Confidence** blends the Isolation Forest's abnormality magnitude with
  how strongly the rule layer agrees. **Support** reports how many
  intervals were flagged and the longest consecutive run.
- **Sensor health** (frozen/spike stations only) and the **fleet-wide
  prediction score** are derived from how many stations show mild
  (70-89%) vs severe (<70%) fault signatures; the alert trigger fires
  at ≤70%, matching the dashboard.

## Using real historical data instead of synthetic data

Everything downstream of `simulate.py` — `features.py`, `rules.py`,
`model.py`, `explain.py`, `fusion.py` — only cares about getting a pandas
DataFrame with this shape, regardless of where it came from:

| column | type | notes |
|---|---|---|
| `station_id` | str | unique AWS code, e.g. `AWS-UK-DDN` |
| `state` | str | display name, e.g. `Dehradun` |
| `timestamp` | datetime | ascending, per station |
| `temperature` | float | °C |
| `pressure` | float | hPa |
| `humidity` | float | % |
| `scenario` | str | only used to pick the Isolation Forest's *training* baseline (see below) and to print accuracy in the demo scripts — **not** required at real inference time |

So swapping in real data is a matter of replacing the call to
`simulate.generate_dataset()` with your own loader that returns a frame in
that shape. A minimal version, reading from a CSV export of your AWS
archive (or a `pandas.read_sql()` against the PostgreSQL store the stack
diagram already calls for):

```python
# load_real.py
import pandas as pd

def load_real_data(path_or_query, from_db=False, conn=None):
    if from_db:
        df = pd.read_sql(path_or_query, conn)
    else:
        df = pd.read_csv(path_or_query, parse_dates=["timestamp"])

    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    # If you don't have labelled anomaly periods, mark everything "normal" —
    # the Isolation Forest is unsupervised, it doesn't need labels to run.
    # It only uses the "normal" rows to pick its baseline (see below).
    if "scenario" not in df.columns:
        df["scenario"] = "normal"
    return df
```

Then the pipeline runs exactly as before:

```python
from features import build_features
from rules import apply_rules
from model import train_and_score
from explain import explain_anomalies
from fusion import fuse, sensor_health_summary
from load_real import load_real_data

raw = load_real_data("historic_aws_readings.csv")
feat = apply_rules(build_features(raw))
clf, scored = train_and_score(feat)          # see baseline note below
explained = explain_anomalies(clf, scored)
fused = fuse(explained)
health = sensor_health_summary(fused)
```

Three things worth adjusting once the data is real rather than synthetic:

1. **Baseline for training.** `train_and_score()` fits the Isolation Forest
   on rows where `scenario == "normal"`. With real data you won't have that
   label in advance — instead, pick a stretch of known-calm weather (e.g.
   a fair-weather week with no advisories in your records) per station,
   tag *those* rows `"normal"`, and leave the rest unlabelled. The model
   only needs a trustworthy "this is what normal looks like" sample; it
   doesn't need every row labelled.
2. **Resolution.** `features.py`'s rolling window (`WINDOW = 6`) and
   `rules.py`'s rate-of-change limits (`RATE_LIMIT`) assume ~15-minute
   reporting. If your historic archive reports at a different interval,
   rescale `RATE_LIMIT` proportionally (it's a *per-interval* limit) and
   adjust `WINDOW` so it still spans roughly the same wall-clock time.
3. **Physical ranges.** `rules.PHYSICAL_RANGE` is set for the Indian AWS
   network generally. Tighten it per station/region if you have
   station-specific instrument specs — e.g. a hill station's plausible
   temperature floor is lower than a plains station's.

Nothing in `fusion.py`'s classification thresholds (flatline fraction, run
length, rate fraction) is synthetic-data-specific — they're built from the
*physics* of each fault type, not from the shape of the simulator's random
walk — so they should carry over reasonably to real readings, but treat
them as a starting point to re-tune once you can check them against real
labelled incidents, the same way you'd validate any anomaly detector.

## Known limitation

The storm/flood boundary is heuristic and threshold-based (see `fusion.py`,
`classify_station`) — tuned against this synthetic generator, not a labelled
real-world dataset. For the hackathon, that's the one number worth being
ready to defend to judges; everything else (frozen/spike/normal) classifies
cleanly across random seeds.
