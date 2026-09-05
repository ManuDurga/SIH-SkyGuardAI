# SkyGuard AI — working anomaly-detection model

A runnable implementation of the pipeline from the architecture diagram:

```
AWS Sensor Data → Data Cleaning & Preprocessing → Rule-Based Quality Checks
→ AI/ML Anomaly Detection → Temporal + Multivariate + Spatial Analysis
→ Evidence Fusion & Anomaly Classification → Confidence + Severity + Root Cause
→ Sensor Health Assessment → Dashboard + Real-Time Alerts
```

It's not a mockup — it simulates realistic AWS temperature/pressure/humidity
data for 5 stations (Punjab, Himachal Pradesh, Uttarakhand, Uttar Pradesh,
Rajasthan), one in each scenario class (normal / frozen / spike / storm /
flood), then actually detects and classifies each one from the raw signal —
no ground-truth labels are used at inference time.

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

## Known limitation

The storm/flood boundary is heuristic and threshold-based (see `fusion.py`,
`classify_station`) — tuned against this synthetic generator, not a labelled
real-world dataset. For the hackathon, that's the one number worth being
ready to defend to judges; everything else (frozen/spike/normal) classifies
cleanly across random seeds.
