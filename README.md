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
| `test_novel_patterns.py` | Regression check: novel patterns must not be force-fit into a known archetype |

## Run it

```bash
pip install -r requirements.txt
python main.py                 # runs the pipeline once, prints a report,
                                # writes pipeline_output.json

uvicorn app:app --reload       # or serve it — GET /analyze for live JSON
```

## How the classification actually works

**v3 design change: hierarchical classification.** Earlier versions
scored all four archetypes (frozen/spike/storm/flood) continuously
instead of using a hard `if/elif` cascade — an improvement over the
original, which force-fit everything into one of four labels — but still
let all four compete in one flat ranking. That meant a weak weather-side
score could still technically "win" over an even weaker sensor-side
score. Classification is now genuinely hierarchical, in two stages:

1. **Broad type first.** Before naming any specific archetype, decide
   whether the evidence looks sensor-like, weather-like, or insufficient
   for either — using `max(frozen_score, spike_score)` vs.
   `max(storm_score, flood_score)`, gated by a minimum evidence floor and
   a margin between the two sides. Storm and spike, for example, are
   never compared head-to-head to pick a "winner" — they answer different
   questions (weather vs. sensor) and only ever compete within their own
   branch.
2. **Subtype only within the winning branch.** Once the broad type is
   decided, the specific archetype (frozen vs. spike, or storm vs. flood)
   still has to clear its own confidence floor and margin over its
   in-branch runner-up. If it doesn't, the result falls back within that
   branch — it's never re-routed across branches.

The full output hierarchy:

```text
NORMAL

Sensor fault:  FROZEN, SPIKE, else OTHER_SENSOR_FAULT
Weather event: STORM, FLOOD, else OTHER_WEATHER_EVENT
Neither branch has enough evidence: UNKNOWN_ANOMALY
```

- **Baselines are computed per-station, not hard-coded.** An early
  version compared every station's temperature/pressure against the
  synthetic simulator's constants (25°C / 1013 hPa) directly in the
  classification logic and RCA text — which would silently misjudge any
  station with a different climate baseline (Guwahati vs. Jaipur, for
  instance). Now each station's own recent readings set its baseline
  before any percentage change is computed.
- **The four main archetypes stay the priority.** `FROZEN`, `SPIKE`,
  `STORM`, and `FLOOD` are still what the project is built to detect and
  demonstrate — `OTHER_SENSOR_FAULT`, `OTHER_WEATHER_EVENT`, and
  `UNKNOWN_ANOMALY` are fallback states, not additional primary
  scenarios. They exist so the four main outputs stay trustworthy: a
  known label is only returned when there's positive evidence for it,
  never just because it happened to be the least-weak option among four
  weak matches.
  `test_novel_patterns.py` checks this directly: it feeds the pipeline a
  synthetic "heatwave" (temperature up, humidity down — the opposite sign
  of the storm/flood signature) and a slow-drifting pressure sensor
  (neither flat nor bursty), and asserts neither gets force-fit into
  storm/flood or frozen/spike respectively. Run `python
  test_novel_patterns.py` to see it pass — and feel free to add your own
  novel synthetic scenarios there to stress-test the boundary further.
- **Storm vs. flood is still the fuzziest known-archetype boundary** (same
  underlying physics, told apart mainly by duration) — occasionally this
  now reports `OTHER_WEATHER_EVENT` at the margin instead of forcing a
  pick between the two. That's the intended trade-off: less
  complete-looking output, fewer confidently wrong labels.
- **Confidence** for a named archetype (`FROZEN`/`SPIKE`/`STORM`/`FLOOD`)
  blends its subtype match score and its margin over its in-branch
  runner-up. For `OTHER_SENSOR_FAULT` / `OTHER_WEATHER_EVENT`, confidence
  reflects how sure the model is of the *broad* type, not the specific
  archetype. For `UNKNOWN_ANOMALY`, confidence is capped lowest of all —
  it means "confident something is anomalous," not "confident which
  broad type it even is." **Support** reports how many intervals were
  flagged, the longest consecutive run, and the full four-way score
  breakdown.
- **Sensor health** now also covers `OTHER_SENSOR_FAULT` stations (they
  still warrant equipment follow-up even though the exact fault mode
  couldn't be named), while `OTHER_WEATHER_EVENT` / `UNKNOWN_ANOMALY` get
  no health score (not confirmed to be equipment-related). The
  **fleet-wide prediction score** and the ≤70% alert trigger are
  unchanged.
- **Known limitation:** the underlying rule/ML detection layer (not
  `fusion.py`) is tuned for the fault/event durations in this dataset. A
  *very* slow, gradual sensor drift (small per-interval change,
  accumulating over hundreds of intervals) can fall under both the
  rolling z-score and rate-of-change thresholds and simply not get
  flagged at all — a false negative rather than a false positive, but
  worth knowing about and tuning `features.py`/`rules.py` for if
  slow-drift faults matter for your deployment.


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


## Hierarchical anomaly classification

SkyGuard keeps four primary supported anomaly outputs: **Frozen**, **Spike**, **Storm**, and **Flood**.
It does not force every abnormal pattern into one of those four labels. The fusion layer first decides whether the evidence is broadly **sensor-like**, **weather-like**, or **ambiguous**, using generic behaviour rather than the four subtype scores. Only then does it attempt subtype classification.

Fallback outputs are:

- `other_sensor_fault` — sensor-like anomaly that does not confidently match Frozen or Spike
- `other_weather_event` — weather-like anomaly that does not confidently match Storm or Flood
- `unknown_anomaly` — anomaly with insufficient/conflicting broad evidence

Examples used in regression testing:

- a sustained heatwave-like temperature rise with falling humidity -> `other_weather_event`
- a smooth isolated pressure calibration drift -> `other_sensor_fault`

This separation is intentional: **failure to match Storm/Flood is not evidence of a sensor fault, and failure to match Frozen/Spike is not evidence of a weather event.**
