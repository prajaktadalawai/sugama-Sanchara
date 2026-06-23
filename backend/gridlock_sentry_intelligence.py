"""
Gridlock Sentry — Core Intelligence Engine
==========================================
Implements all intelligence modules from the architecture diagram:
  1. Hotspot detection      — DBSCAN spatial clustering on lat/lon
  2. Impact scoring         — uses congestion_impact_score from pipeline
  3. Anomaly detection      — junctions scoring 3x above their own baseline
  4. Temporal forecast      — per-zone hour-of-week violation density (per-zone normalised)
  5. Root cause classifier  — WHY this spot keeps generating violations

EPS formula (proactive — includes live forecast intensity):
  impact_norm        × 0.30
  capacity_norm      × 0.25
  anomaly_norm       × 0.15
  forecast_intensity × 0.20   ← proactive component: what will get bad NOW
  root_cause_weight  × 0.10

All inputs from pipeline output (CSV-derived only). No external data.
"""

import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# ── Paths as constants at the top — change here, propagates everywhere ────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "jan to may police violation_anonymized791b166.csv")
DB_PATH   = os.path.join(BASE_DIR, "gridlock_sentry.db")

from gridlock_sentry_pipeline import (
    run_pipeline,
    junction_hour_capacity_loss,
    police_station_zone_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — HOTSPOT DETECTION (DBSCAN spatial clustering)
# ─────────────────────────────────────────────────────────────────────────────

def detect_hotspots(df: pd.DataFrame,
                    eps_km: float = 0.3,
                    min_samples: int = 3) -> pd.DataFrame:
    """
    Cluster violation records by lat/lon using DBSCAN.
    eps_km      : neighbourhood radius in kilometres
    min_samples : minimum violations to form a hotspot cluster
    Returns df with 'hotspot_cluster' column (-1 = noise).
    """
    # Cluster on junction centroids, not raw records
    junction_centroids = df[df["junction_flag"]==1].groupby("junction_name")[["latitude", "longitude"]].mean().reset_index()
    coords  = junction_centroids[["latitude", "longitude"]].values
    eps_rad = eps_km / 6371.0   # convert km to radians for haversine

    labels = DBSCAN(
        eps=eps_rad, min_samples=min_samples,
        algorithm="ball_tree", metric="haversine", n_jobs=-1,
    ).fit_predict(np.radians(coords))

    junction_centroids["hotspot_cluster"] = labels
    
    # Merge back to raw records via junction_name
    df = df.copy()
    df = df.merge(junction_centroids[["junction_name", "hotspot_cluster"]], on="junction_name", how="left")
    # Ensure unmatched (noise) junctions get -1
    df["hotspot_cluster"] = df["hotspot_cluster"].fillna(-1).astype(int)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"[Hotspot] {n_clusters} clusters | {n_noise:,} noise junctions | "
          f"{len(junction_centroids)-n_noise:,} clustered junctions")
    return df


def build_hotspot_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-cluster summary: centroid, size, avg impact, top vehicle type.
    hotspot_tier uses LOW/MED/HIGH — 'yellow' is reserved exclusively for
    the blind-spot concept in /api/yellow-zones to avoid naming confusion.
    """
    clustered = df[df["hotspot_cluster"] >= 0]
    summary = (
        clustered.groupby("hotspot_cluster")
        .agg(
            violation_count         =("congestion_impact_score", "size"),
            lat                     =("latitude",                "mean"),
            lon                     =("longitude",               "mean"),
            mean_impact_score       =("congestion_impact_score", "mean"),
            max_impact_score        =("congestion_impact_score", "max"),
            mean_carriageway_blocked=("carriageway_blocked_pct", "mean"),
            rush_hour_pct           =("rush_flag",               "mean"),
            at_junction_pct         =("junction_flag",           "mean"),
            top_road_type           =("road_type",               lambda x: x.mode()[0]),
            top_vehicle             =("vehicle_type",            lambda x: x.mode()[0]),
        )
        .reset_index()
        .sort_values("mean_impact_score", ascending=False)
    )
    # LOW / MED / HIGH — "yellow" is NOT used here (reserved for blind-spot zones)
    summary["hotspot_tier"] = pd.cut(
        summary["mean_impact_score"],
        bins=[0, 0.15, 0.30, 1.01],
        labels=["LOW", "MED", "HIGH"]
    )
    print(f"[Hotspot] Tiers — HIGH:{(summary['hotspot_tier']=='HIGH').sum()} "
          f"MED:{(summary['hotspot_tier']=='MED').sum()} "
          f"LOW:{(summary['hotspot_tier']=='LOW').sum()}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame, spike_factor: float = 3.0) -> pd.DataFrame:
    """
    Flags junction-hours where violation count >= spike_factor × the
    median count for that junction across its ACTIVE hours only.
    (Baseline = median among hours that had at least one violation.)
    Source: junction_name + created_datetime (hour) from CSV.
    """
    jh = (
        df[df["junction_flag"] == 1]
        .groupby(["junction_name", "hour"])["congestion_impact_score"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "violation_count", "mean": "avg_impact"})
    )
    junction_medians = (
        jh.groupby("junction_name")["violation_count"]
        .median()
        .rename("median_count")
    )
    jh = jh.join(junction_medians, on="junction_name")
    jh["spike_ratio"] = (jh["violation_count"] / jh["median_count"].clip(lower=1)).round(2)
    jh["is_anomaly"]  = (jh["spike_ratio"] >= spike_factor).astype(int)
    print(f"[Anomaly] {jh['is_anomaly'].sum()} junction-hour spikes "
          f"(>= {spike_factor}x active-hour baseline)")
    return jh.sort_values("spike_ratio", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — TEMPORAL FORECAST (per-zone normalised)
# ─────────────────────────────────────────────────────────────────────────────

def _zone_minmax(s: pd.Series) -> pd.Series:
    """Min-max scale within a single police station's time series."""
    mn, mx = s.min(), s.max()
    return ((s - mn) / (mx - mn + 1e-9)).round(4)


def build_temporal_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a (day x hour) violation density matrix per police_station zone.
    forecast_intensity is normalised WITHIN each zone (0=that zone's quietest
    hour, 1=that zone's busiest hour) — enabling meaningful cross-hour
    comparison inside one zone without volume differences swamping small zones.

    Source: created_datetime (day, hour) + police_station from CSV.
    """
    forecast = (
        df.groupby(["police_station", "day", "hour"])
        .agg(
            avg_violations=("congestion_impact_score", "size"),
            avg_impact    =("congestion_impact_score", "mean"),
        )
        .reset_index()
    )
    # Per-zone normalisation — quiet stations show meaningful peaks within themselves
    forecast["forecast_intensity"] = (
        forecast.groupby("police_station")["avg_violations"]
        .transform(_zone_minmax)
    )
    print(f"[Forecast] {len(forecast):,} zone*day*hour patterns across "
          f"{forecast['police_station'].nunique()} police stations")
    return forecast


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — ROOT CAUSE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

# Rules checked IN ORDER — first match wins.
# RUSH_HOUR_DEMAND_SPIKE is checked BEFORE NO_PARKING_ALTERNATIVE to prevent
# the broad primary-road condition from swallowing rush-hour junctions.
ROOT_CAUSE_RULES = [
    ("STRUCTURAL_DESIGN", {
        "desc": "Violations across 18+ distinct hours — present regardless of time of day; "
                "structural road design prevents safe stopping alternatives.",
        "condition": lambda r: r["active_hours"] >= 18 and r["violation_count"] >= 50,
    }),
    ("RUSH_HOUR_DEMAND_SPIKE", {
        "desc": "40%+ of violations occur in peak hours (8-10 AM / 5-8 PM) — "
                "demand for stopping/loading exceeds road capacity during commute.",
        "condition": lambda r: r["rush_hour_pct"] >= 0.4 and r["violation_count"] >= 15,
    }),
    ("ENFORCEMENT_BLIND_SPOT", {
        "desc": "High violation count but low congestion impact score — "
                "violations occur in low-priority zones that receive minimal patrol.",
        "condition": lambda r: r["violation_count"] >= 20 and r["mean_impact_score"] < 0.12,
    }),
    ("NO_PARKING_ALTERNATIVE", {
        "desc": "High volume on a primary road without consistent rush-hour concentration — "
                "likely a commercial stretch with no designated loading/parking zone nearby.",
        "condition": lambda r: r["violation_count"] >= 30 and r["top_road_type"] == "primary",
    }),
    ("GENERAL_VIOLATION_ZONE", {
        "desc": "Mixed or low-frequency pattern — standard periodic enforcement sufficient.",
        "condition": lambda r: True,
    }),
]


def classify_root_cause(row: dict) -> str:
    for cause, meta in ROOT_CAUSE_RULES:
        try:
            if meta["condition"](row):
                return cause
        except Exception:
            continue
    return "GENERAL_VIOLATION_ZONE"


def build_root_cause_report(df: pd.DataFrame) -> pd.DataFrame:
    """Junction-level root cause classification."""
    named = df[df["junction_flag"] == 1]
    junc_agg = (
        named.groupby("junction_name")
        .agg(
            violation_count  =("congestion_impact_score", "size"),
            mean_impact_score=("congestion_impact_score", "mean"),
            active_hours     =("hour",                    "nunique"),
            rush_hour_pct    =("rush_flag",               "mean"),
            top_road_type    =("road_type",               lambda x: x.mode()[0]),
            lat              =("latitude",                "mean"),
            lon              =("longitude",               "mean"),
        )
        .reset_index()
    )
    junc_agg["root_cause"] = junc_agg.apply(
        lambda r: classify_root_cause(r.to_dict()), axis=1
    )
    print(f"[RootCause] Distribution:\n{junc_agg['root_cause'].value_counts().to_string()}")
    return junc_agg.sort_values("violation_count", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — ENFORCEMENT PRIORITY SCORE (EPS) 0-100 per junction
# ─────────────────────────────────────────────────────────────────────────────

def compute_eps(
    anomaly_df:      pd.DataFrame,
    root_cause_df:   pd.DataFrame,
    jh_capacity:     pd.DataFrame,
    forecast_df:     pd.DataFrame,    # needed for proactive component
    df:              pd.DataFrame,    # raw records — needed to map junction -> station
    current_day:  int = None,
    current_hour: int = None,
) -> pd.DataFrame:
    """
    Combines all intelligence signals into a single EPS (0-100) per junction.

    EPS = impact_norm        * 0.30   (historical severity)
        + capacity_norm      * 0.25   (road blockage accumulation)
        + anomaly_norm       * 0.15   (spike detection)
        + forecast_intensity * 0.20   (PROACTIVE: how busy this zone is RIGHT NOW)
        + root_cause_weight  * 0.10   (structural urgency)

    The forecast component is what makes EPS proactive — it answers
    "where should I send officers in the next hour?" not just "where was it bad."
    """
    if current_day is None or current_hour is None:
        now = datetime.now()
        current_day, current_hour = now.weekday(), now.hour

    eps_df = root_cause_df[[
        "junction_name", "violation_count", "mean_impact_score",
        "rush_hour_pct", "lat", "lon", "root_cause"
    ]].copy()
    
    eps_df = eps_df[eps_df["violation_count"] >= 10]

    # Map junction -> dominant police_station (for forecast lookup)
    junc_station = (
        df[df["junction_flag"] == 1]
        .groupby("junction_name")["police_station"]
        .agg(lambda x: x.mode().iat[0])
        .reset_index()
        .rename(columns={"police_station": "police_station"})
    )
    eps_df = eps_df.merge(junc_station, on="junction_name", how="left")

    # Proactive: look up that zone's forecast_intensity for current day+hour
    fc_now = forecast_df[
        (forecast_df["day"] == current_day) &
        (forecast_df["hour"] == current_hour)
    ][["police_station", "forecast_intensity"]].rename(
        columns={"forecast_intensity": "current_forecast_intensity"}
    )
    eps_df = eps_df.merge(fc_now, on="police_station", how="left")
    eps_df["current_forecast_intensity"] = eps_df["current_forecast_intensity"].fillna(0.0)

    # Capacity loss (sum across all hours per junction)
    jh_cap = (
        jh_capacity.groupby("junction_name")["effective_capacity_loss_index"]
        .sum().reset_index()
        .rename(columns={"effective_capacity_loss_index": "total_capacity_loss"})
    )
    eps_df = eps_df.merge(jh_cap, on="junction_name", how="left")
    eps_df["total_capacity_loss"] = eps_df["total_capacity_loss"].fillna(0)

    # Anomaly: max spike ratio per junction
    anomaly_max = (
        anomaly_df[anomaly_df["is_anomaly"] == 1]
        .groupby("junction_name")["spike_ratio"]
        .max().reset_index()
        .rename(columns={"spike_ratio": "max_spike_ratio"})
    )
    eps_df = eps_df.merge(anomaly_max, on="junction_name", how="left")
    eps_df["max_spike_ratio"] = eps_df["max_spike_ratio"].fillna(1.0)

    # Root cause severity weight (bounded 0.0-1.0, consistent scale)
    rc_weight = {
        "STRUCTURAL_DESIGN":      1.0,
        "RUSH_HOUR_DEMAND_SPIKE": 0.8,
        "NO_PARKING_ALTERNATIVE": 0.6,
        "ENFORCEMENT_BLIND_SPOT": 0.4,
        "GENERAL_VIOLATION_ZONE": 0.0,
    }
    eps_df["root_cause_weight"] = eps_df["root_cause"].map(rc_weight).fillna(0.0)

    def norm(s):
        mn, mx = s.min(), s.max()
        return ((s - mn) / (mx - mn + 1e-9)).clip(0, 1)

    eps_df["impact_norm"]   = norm(eps_df["mean_impact_score"])
    eps_df["capacity_norm"] = norm(eps_df["total_capacity_loss"])
    eps_df["anomaly_norm"]  = norm(eps_df["max_spike_ratio"])
    # forecast_intensity already [0,1] per-zone — no further scaling needed

    eps_df["eps_raw"] = (
        eps_df["impact_norm"]                  * 0.30
        + eps_df["capacity_norm"]              * 0.25
        + eps_df["anomaly_norm"]               * 0.15
        + eps_df["current_forecast_intensity"] * 0.20
        + eps_df["root_cause_weight"]          * 0.10
    )
    eps_df["eps"] = (norm(eps_df["eps_raw"]) * 100).clip(0, 100).round(1)

    # Percentile-based tiers — CRITICAL/HIGH always populated
    p50 = eps_df["eps"].quantile(0.50)
    p75 = eps_df["eps"].quantile(0.75)
    p90 = eps_df["eps"].quantile(0.90)

    def assign_tier(score):
        if score >= p90: return "CRITICAL"
        if score >= p75: return "HIGH"
        if score >= p50: return "MEDIUM"
        return "LOW"

    eps_df["action_tier"] = eps_df["eps"].apply(assign_tier)
    print(f"\n[EPS] Day={current_day} Hour={current_hour} (proactive window)")
    print(f"[EPS] Tiers: CRITICAL>={p90:.1f}  HIGH>={p75:.1f}  MEDIUM>={p50:.1f}")
    print(f"[EPS] Distribution:\n{eps_df['action_tier'].value_counts().to_string()}")
    return eps_df.sort_values("eps", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SQLITE STORAGE
# ─────────────────────────────────────────────────────────────────────────────

def save_to_sqlite(
    eps_df:          pd.DataFrame,
    hotspot_df:      pd.DataFrame,
    anomaly_df:      pd.DataFrame,
    forecast_df:     pd.DataFrame,
    root_cause_df:   pd.DataFrame,
    jh_capacity_df:  pd.DataFrame,
    zone_summary_df: pd.DataFrame,
    db_path: str = DB_PATH,
):
    """Save all intelligence outputs to SQLite for FastAPI to serve."""
    conn = sqlite3.connect(db_path)
    tables = {
        "eps_rankings":         eps_df,
        "hotspot_clusters":     hotspot_df,
        "anomaly_detections":   anomaly_df,
        "temporal_forecast":    forecast_df,
        "root_cause_report":    root_cause_df,
        "junction_hour_scores": jh_capacity_df,
        "zone_summaries":       zone_summary_df,
    }
    for name, frame in tables.items():
        frame_out = frame.copy()
        for col in frame_out.select_dtypes(include="category").columns:
            frame_out[col] = frame_out[col].astype(str)
        frame_out.to_sql(name, conn, if_exists="replace", index=False)
        print(f"  [SQLite] {name}: {len(frame_out):,} rows")
    conn.close()
    print(f"[SQLite] All tables saved -> {db_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_intelligence_pipeline():
    print("\n" + "=" * 60)
    print("GRIDLOCK SENTRY - INTELLIGENCE ENGINE")
    print("=" * 60)

    df = run_pipeline(INPUT_CSV)

    print("\n[Step 2] Hotspot Detection...")
    df = detect_hotspots(df, eps_km=0.3, min_samples=3)
    hotspot_summary = build_hotspot_summary(df)

    print("\n[Step 3] Junction-Hour Capacity Loss...")
    jh_capacity = junction_hour_capacity_loss(df)
    print(f"  Top: {jh_capacity.iloc[0]['junction_name']} "
          f"h={jh_capacity.iloc[0]['hour']} "
          f"loss={jh_capacity.iloc[0]['effective_capacity_loss_index']}")

    zone_summary = police_station_zone_summary(df)

    print("\n[Step 4] Anomaly Detection...")
    anomaly_df = detect_anomalies(df, spike_factor=3.0)

    print("\n[Step 5] Temporal Forecast (per-zone normalised)...")
    forecast_df = build_temporal_forecast(df)

    print("\n[Step 6] Root Cause Classification...")
    root_cause_df = build_root_cause_report(df)

    print("\n[Step 7] Enforcement Priority Score (with forecast)...")
    eps_df = compute_eps(
        anomaly_df, root_cause_df,
        jh_capacity, forecast_df, df          # forecast + df now passed in
    )

    print("\n--- TOP 20 JUNCTIONS BY EPS ---")
    display = ["junction_name", "eps", "action_tier", "root_cause",
               "violation_count", "mean_impact_score",
               "current_forecast_intensity", "lat", "lon"]
    print(eps_df[display].head(20).to_string(index=False))

    print("\n[Step 8] Saving to SQLite...")
    save_to_sqlite(
        eps_df, hotspot_summary, anomaly_df,
        forecast_df, root_cause_df, jh_capacity, zone_summary
    )

    print("\n" + "=" * 60)
    print("Intelligence pipeline complete. Ready for FastAPI.")
    print("=" * 60)
    return eps_df, df


if __name__ == "__main__":
    run_intelligence_pipeline()
