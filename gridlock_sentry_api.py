"""
Gridlock Sentry — FastAPI Backend
===================================
Serves all intelligence outputs from SQLite to the dashboard.

Endpoints:
  GET /api/stats                           — overall dataset summary
  GET /api/eps                             — full EPS ranked list
  GET /api/eps/critical                    — CRITICAL + HIGH junctions only
  GET /api/hotspots                        — DBSCAN cluster centroids
  GET /api/anomalies                       — flagged junction-hour spikes
  GET /api/forecast/stations/list          — available stations
  GET /api/forecast/{station}              — hourly forecast for a zone
  GET /api/junctions                       — junction-hour capacity loss table
  GET /api/zones                           — police station zone summaries
  GET /api/root-causes                     — root cause classification
  GET /api/heatmap-data                    — lat/lon + score for map rendering
  GET /api/yellow-zones                    — low density high impact zones

Run:
  pip install fastapi uvicorn
  python gridlock_sentry_api.py
  -> http://localhost:8000
  -> http://localhost:8000/docs  (auto Swagger UI)
"""

import sqlite3
import os
from typing import Optional
import numpy as np

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "gridlock_sentry.db")

app = FastAPI(
    title="Gridlock Sentry API",
    description="AI-Driven Parking Intelligence — Flipkart GRiD 6.0",
    version="1.0.0",
)

@app.get("/")
def get_dashboard():
    """Serves the main dashboard page."""
    index_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(index_path)


# Allow all origins so the HTML dashboard can call locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_conn():
    if not os.path.exists(DB_PATH):
        raise HTTPException(
            status_code=503,
            detail="Database not found. Run gridlock_sentry_intelligence.py first."
        )
    return sqlite3.connect(DB_PATH)


def query(sql: str, params: tuple = ()) -> list[dict]:
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    cur = conn.execute(sql, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
def get_stats():
    """Overall dataset and system summary — shown on dashboard header."""
    total_viol  = query("SELECT SUM(violation_count) AS n FROM eps_rankings")[0]["n"] or 0
    stations    = query("SELECT COUNT(DISTINCT police_station) AS n FROM zone_summaries")[0]["n"]
    eps_rows    = query("SELECT COUNT(*) AS n FROM eps_rankings")[0]["n"]
    critical    = query("SELECT COUNT(*) AS n FROM eps_rankings WHERE action_tier='CRITICAL'")[0]["n"]
    high        = query("SELECT COUNT(*) AS n FROM eps_rankings WHERE action_tier='HIGH'")[0]["n"]
    clusters    = query("SELECT COUNT(*) AS n FROM hotspot_clusters")[0]["n"]
    anomalies   = query("SELECT COUNT(*) AS n FROM anomaly_detections WHERE is_anomaly=1")[0]["n"]
    return {
        "total_violations_approved": total_viol,
        "total_junctions_monitored": eps_rows,
        "critical_zones": critical,
        "high_zones": high,
        "hotspot_clusters": clusters,
        "anomalies_detected": anomalies,
        "police_stations_covered": stations,
        "months_of_data": 5,
    }


@app.get("/api/eps")
def get_eps(
    limit: int = Query(default=50, le=200),
    tier: Optional[str] = Query(default=None),
):
    """
    Full EPS ranked action list — the core enforcement dispatch table.
    Filter by tier: CRITICAL | HIGH | MEDIUM | LOW
    """
    if tier:
        rows = query(
            "SELECT * FROM eps_rankings WHERE action_tier=? ORDER BY eps DESC LIMIT ?",
            (tier.upper(), limit)
        )
    else:
        rows = query(
            "SELECT * FROM eps_rankings ORDER BY eps DESC LIMIT ?",
            (limit,)
        )
    return rows


@app.get("/api/eps/critical")
def get_critical_junctions():
    """CRITICAL and HIGH junctions — immediate dispatch directives."""
    rows = query(
        "SELECT * FROM eps_rankings WHERE action_tier IN ('CRITICAL','HIGH') ORDER BY eps DESC"
    )
    return rows


@app.get("/api/hotspots")
def get_hotspots(tier: Optional[str] = Query(default=None)):
    """DBSCAN cluster centroids for the two-layer heatmap."""
    if tier:
        rows = query(
            "SELECT * FROM hotspot_clusters WHERE hotspot_tier=? ORDER BY mean_impact_score DESC",
            (tier.upper(),)
        )
    else:
        rows = query(
            "SELECT * FROM hotspot_clusters ORDER BY mean_impact_score DESC"
        )
    return rows


@app.get("/api/anomalies")
def get_anomalies(only_flagged: bool = Query(default=True)):
    """Junction-hour anomalies — spikes 3x above baseline."""
    if only_flagged:
        rows = query(
            "SELECT * FROM anomaly_detections WHERE is_anomaly=1 ORDER BY spike_ratio DESC LIMIT 100"
        )
    else:
        rows = query(
            "SELECT * FROM anomaly_detections ORDER BY spike_ratio DESC LIMIT 200"
        )
    return rows


@app.get("/api/forecast/stations/list")
def list_stations():
    """All available police station names for the forecast dropdown."""
    rows = query("SELECT DISTINCT police_station FROM temporal_forecast ORDER BY police_station")
    return [r["police_station"] for r in rows]


@app.get("/api/forecast/{station}")
def get_forecast(station: str, day: Optional[int] = Query(default=None)):
    """
    Hourly violation forecast for a police station zone.
    day: 0=Monday … 6=Sunday (optional filter)
    """
    if day is not None:
        rows = query(
            "SELECT * FROM temporal_forecast WHERE police_station=? AND day=? ORDER BY hour",
            (station, day)
        )
    else:
        rows = query(
            "SELECT * FROM temporal_forecast WHERE police_station=? ORDER BY day, hour",
            (station,)
        )
    if not rows:
        raise HTTPException(status_code=404, detail=f"Station '{station}' not found.")
    return rows


@app.get("/api/junctions")
def get_junction_scores(
    limit: int = Query(default=100, le=500),
    hour: Optional[int] = Query(default=None),
):
    """Junction-hour effective capacity loss table."""
    if hour is not None:
        rows = query(
            "SELECT * FROM junction_hour_scores WHERE hour=? ORDER BY effective_capacity_loss_index DESC LIMIT ?",
            (hour, limit)
        )
    else:
        rows = query(
            "SELECT * FROM junction_hour_scores ORDER BY effective_capacity_loss_index DESC LIMIT ?",
            (limit,)
        )
    return rows


@app.get("/api/zones")
def get_zones(limit: int = Query(default=50, le=200)):
    """Police station zone summaries — ranked by zone impact index."""
    rows = query(
        "SELECT * FROM zone_summaries ORDER BY zone_impact_index DESC LIMIT ?",
        (limit,)
    )
    return rows


@app.get("/api/root-causes")
def get_root_causes(cause: Optional[str] = Query(default=None)):
    """Root cause classification per junction."""
    if cause:
        rows = query(
            "SELECT * FROM root_cause_report WHERE root_cause=? ORDER BY violation_count DESC",
            (cause.upper(),)
        )
    else:
        rows = query(
            "SELECT * FROM root_cause_report ORDER BY violation_count DESC LIMIT 100"
        )
    return rows


@app.get("/api/heatmap-data")
def get_heatmap_data(
    layer: str = Query(default="impact", description="'impact' or 'density'")
):
    """
    Lat/lon + score for Leaflet heatmap rendering.
    layer=impact  -> congestion impact score per junction (PP2 layer)
    layer=density -> violation count per junction (density layer)
    Returns list of [lat, lon, intensity] triples.
    """
    if layer == "impact":
        rows = query(
            """SELECT lat, lon, mean_congestion_impact_score AS score
               FROM junction_hour_scores
               WHERE lat IS NOT NULL AND lon IS NOT NULL"""
        )
        field = "score"
    else:
        rows = query(
            """SELECT lat, lon, violation_count AS score
               FROM junction_hour_scores
               WHERE lat IS NOT NULL AND lon IS NOT NULL"""
        )
        field = "score"

    # Normalise score to [0, 1] for Leaflet heatmap intensity
    scores = [r[field] for r in rows if r[field] is not None]
    max_score = max(scores) if scores else 1
    data = [
        [r["lat"], r["lon"], round((r[field] or 0) / max_score, 4)]
        for r in rows
        if r["lat"] and r["lon"]
    ]
    return {"layer": layer, "count": len(data), "points": data}


@app.get("/api/yellow-zones")
def get_yellow_zones():
    """
    Yellow zones: low violation DENSITY but HIGH congestion IMPACT.
    These are enforcement blind spots — the key innovation of this system.
    """
    rows = query(
        """
        SELECT e.junction_name, e.eps, e.mean_impact_score, e.violation_count,
               e.lat, e.lon, e.root_cause
        FROM eps_rankings e
        """
    )
    if not rows:
        return {"description": "Low density, HIGH impact — enforcement blind spots", "zones": []}
    
    # Calculate percentiles dynamically
    impacts = [r["mean_impact_score"] for r in rows if r["mean_impact_score"] is not None]
    counts = [r["violation_count"] for r in rows if r["violation_count"] is not None]
    
    impact_cutoff = np.percentile(impacts, 75) if impacts else 0.25
    count_cutoff = np.percentile(counts, 50) if counts else 50.0
    
    # Filter the rows
    yellow_zones = [
        r for r in rows
        if r["mean_impact_score"] is not None and r["mean_impact_score"] > impact_cutoff
        and r["violation_count"] is not None and r["violation_count"] < count_cutoff
    ]
    
    # Sort by mean_impact_score descending and limit to 50
    yellow_zones.sort(key=lambda x: x["mean_impact_score"], reverse=True)
    return {
        "description": f"Low density (violation count < {count_cutoff:.1f}), HIGH impact (mean impact score > {impact_cutoff:.4f}) — enforcement blind spots",
        "zones": yellow_zones[:50]
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Gridlock Sentry API starting...")
    print("Dashboard API: http://localhost:8000")
    print("Swagger docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
