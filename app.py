"""
app.py
Thin FastAPI layer over the pipeline — the "FastAPI: connects real-time
data, AI model and dashboard" box in the stack. Run with:

    uvicorn app:app --reload --port 8000

Then GET /analyze to run a fresh simulated batch through the full pipeline
and get back the same station-level classification the dashboard renders.
In a real deployment this endpoint would read live AWS readings (via the
MQTT/WebSocket ingestion box) instead of simulate.generate_dataset().
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from main import run_pipeline

app = FastAPI(title="SkyGuard AI", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/analyze")
def analyze(n_points: int = 300, freq_minutes: int = 15):
    """
    Run the full pipeline (simulate -> features -> rules -> Isolation Forest
    -> SHAP -> fusion -> sensor health) and return per-station results plus
    the fleet-wide sensor-health prediction score.
    """
    result = run_pipeline(n_points=n_points, freq_minutes=freq_minutes, verbose=False)
    fused = result["stations"].copy()
    fused["health"] = fused["health"].astype(object).where(fused["health"].notna(), None)
    return {
        "stations": fused.to_dict(orient="records"),
        "fleet_health": result["health"],
    }
