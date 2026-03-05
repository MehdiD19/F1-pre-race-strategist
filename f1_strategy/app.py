"""FastAPI web server for the F1 Pre-Race Strategy Optimizer.

Run with:
    python app.py
or:
    uvicorn app:app --reload
Then open http://localhost:8000 in your browser.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import uuid
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fastf1
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import CACHE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

fastf1.Cache.enable_cache(CACHE_DIR)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

app = FastAPI(title="F1 Strategy Optimizer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store  {job_id: {status, log, result, error}}
_jobs: Dict[str, Dict[str, Any]] = {}


# ── Schemas ────────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    year: int
    gp: str
    drivers: Optional[List[str]] = None
    max_stops: int = 2


# ── API routes ─────────────────────────────────────────────────────────────

@app.get("/api/events/{year}")
def get_events(year: int):
    """Return all race events for a given season year."""
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    events = []
    for _, row in schedule.iterrows():
        round_num = int(row.get("RoundNumber", 0))
        if round_num <= 0:
            continue
        events.append({
            "round": round_num,
            "name": str(row.get("EventName", "")),
            "country": str(row.get("Country", "")),
            "location": str(row.get("Location", "")),
            "date": str(row.get("EventDate", ""))[:10],
        })
    return sorted(events, key=lambda e: e["round"])


@app.post("/api/analyze")
def start_analysis(request: AnalysisRequest):
    """Start a background analysis job. Returns job_id for polling."""
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "running", "log": [], "result": None, "error": None}

    def _run():
        from api_runner import run_analysis

        def _progress(msg: str):
            _jobs[job_id]["log"].append(msg)

        try:
            result = run_analysis(
                request.year,
                request.gp,
                request.drivers,
                request.max_stops,
                progress=_progress,
            )
            _jobs[job_id]["result"] = result
            _jobs[job_id]["status"] = "done"
        except Exception as exc:
            logger.exception("Analysis job %s failed", job_id)
            _jobs[job_id]["error"] = str(exc)
            _jobs[job_id]["status"] = "error"

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    """Poll the status and progress log of a running job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _jobs[job_id]
    return {
        "status": job["status"],
        "log": job["log"],
        "error": job.get("error"),
    }


@app.get("/api/results/{job_id}")
def get_results(job_id: str):
    """Retrieve the full results dict for a completed job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job is '{job['status']}', not done yet")
    return job["result"]


# ── Frontend serving ───────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index):
        return {"message": "Frontend not found. Run the build or check frontend/ directory."}
    return FileResponse(index)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
