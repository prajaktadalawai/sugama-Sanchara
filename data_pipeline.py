"""
Gridlock Sentry — Data Pipeline
Tier 1: Data Foundation | Tier 2: Quantification Engine

COMPLIANCE:
  Uses only the provided CSV. No external datasets, APIs, or databases.
  Complies with: "Participants must use only the datasets provided by
  HackerEarth for Problem Statements 1 and 2."

DATASET CONSTRAINTS (verified on 298,450 rows):
  action_taken_timestamp : 0% populated — no enforcement resolution signal.
  closed_datetime        : 0% populated — no case-closure signal.
  CONSEQUENCE: No duration-based metrics are computed. This is stated
  explicitly and reflected throughout. The system quantifies congestion
  from observable, present attributes only.

TIER 1 — outputs per record:
  hour, day, month, rush_flag         ← created_datetime
  road_type, lanes, road_priority     ← location + junction_name (text)
  junction_flag                       ← junction_name (text)
  violation_footprint_m               ← vehicle_type
  carriageway_blocked_pct             ← violation_footprint_m / road_width_m
  violation_severity                  ← violation_type (JSON array)

TIER 2 — outputs per record + per junction-hour aggregate:
  congestion_impact_score             ← composite of all Tier 1 features
  junction_hour_violation_count       ← frequency at this junction-hour
  junction_hour_capacity_loss_pct     ← cumulative carriageway blockage
                                         (the money metric — no timestamps)
"""

import json
import os

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# TIER 1 HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def infer_road_attributes(location: str, junction: str) -> tuple[str, int, float]:
    """
    Infer road_type, lanes, road_priority from CSV text fields.
    Source: location column + junction_name column.
    No external data — pure keyword match on address strings.

    Returns: (road_type, lanes, road_priority)
    """
    combined = (str(location) + " " + str(junction)).lower()

    if any(k in combined for k in ["flyover", "ring road", "expressway", "outer ring", "bypass"]):
        return "motorway", 3, 1.00
    if any(k in combined for k in ["main road", "double road", "arterial"]):
        return "primary", 2, 0.85
    if "road" in combined:
        return "primary", 2, 0.75
    if any(k in combined for k in ["cross", "street", "avenue"]):
        return "secondary", 2, 0.60
    if any(k in combined for k in ["lane", "layout"]):
        return "secondary", 1, 0.50
    return "tertiary", 1, 0.40


def infer_junction_flag(junction: str) -> int:
    """1 if a named junction, 0 if 'No Junction'. Source: junction_name."""
    s = str(junction).strip().lower()
    return 0 if s in ("", "nan", "no junction", "none") else 1


# Severity weights for violation types present in the CSV
_VIOLATION_SEVERITY: dict[str, float] = {
    "double parking":                              1.00,
    "parking opposite to another parked vehicle":  0.90,
    "parking in a main road":                      0.80,
    "parking near traffic light":                  0.70,
    "parking near road crossing":                  0.70,
    "parking near bustop":                         0.60,
    "wrong parking":                               0.50,
    "no parking":                                  0.50,
    "parking on footpath":                         0.30,
}

def get_violation_severity(raw: str) -> float:
    """
    Parse the JSON array in violation_type and return the max severity weight.
    Source: violation_type column.
    """
    try:
        items = json.loads(str(raw))
        if not isinstance(items, list):
            items = [str(items)]
    except Exception:
        items = [str(raw)]

    score = 0.30  # minimum — some violation was recorded
    for item in items:
        low = str(item).lower().strip()
        for keyword, weight in _VIOLATION_SEVERITY.items():
            if keyword in low:
                score = max(score, weight)
    return score


# Physical footprint widths derived from vehicle class
_FOOTPRINT_M: dict[str, float] = {
    "HGV": 3.0, "BUS": 3.0, "TRUCK": 3.0,
    "LGV": 2.2,
    "CAR": 1.9,
    "PASSENGER AUTO": 1.5, "AUTO": 1.5,
    "MOTOR CYCLE": 0.8, "MOTORCYCLE": 0.8, "SCOOTER": 0.8,
}

def get_vehicle_footprint(vtype: str) -> float:
    """Vehicle physical width in metres. Source: vehicle_type column."""
    v = str(vtype).upper()
    for key, width in _FOOTPRINT_M.items():
        if key in v:
            return width
    return 1.5  # default


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

_LANE_WIDTH_M = 3.5          # standard lane width — engineering constant


def run_pipeline(input_path: str, nrows: int | None = None) -> pd.DataFrame:
    """
    Execute Tier 1 (Data Foundation) + Tier 2 (Quantification Engine).
    Returns enriched DataFrame ready for Tier 3 intelligence engines.
    """

    # ── Load ──────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("TIER 1 — DATA FOUNDATION")
    print("=" * 60)
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path, nrows=nrows)
    print(f"Raw rows: {len(df):,}")

    # ── Filter: approved records only ─────────────────────────────────────────
    # Source: validation_status
    df = df[df["validation_status"].astype(str).str.lower() == "approved"].copy()
    df = df.reset_index(drop=True)
    print(f"Approved records: {len(df):,}")

    # ── Time features ─────────────────────────────────────────────────────────
    # Source: created_datetime
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], utc=True)
    df["hour"]  = df["created_datetime"].dt.hour
    df["day"]   = df["created_datetime"].dt.dayofweek   # 0=Mon, 6=Sun
    df["month"] = df["created_datetime"].dt.month
    df["rush_flag"] = (
        df["hour"].between(8, 10) | df["hour"].between(17, 20)
    ).astype(int)
    print(f"rush_flag: {df['rush_flag'].sum():,} peak-hour violations "
          f"({df['rush_flag'].mean()*100:.1f}%)")

    # ── Road context ──────────────────────────────────────────────────────────
    # Source: location + junction_name (text keyword heuristics)
    attrs = df.apply(
        lambda r: infer_road_attributes(r["location"], r["junction_name"]),
        axis=1, result_type="expand"
    )
    attrs.columns = ["road_type", "lanes", "road_priority"]
    df = pd.concat([df, attrs], axis=1)

    df["junction_flag"] = df["junction_name"].apply(infer_junction_flag)
    print(f"road_type distribution:\n"
          f"  motorway={( df['road_type']=='motorway').sum():,}  "
          f"primary={(df['road_type']=='primary').sum():,}  "
          f"secondary={(df['road_type']=='secondary').sum():,}  "
          f"tertiary={(df['road_type']=='tertiary').sum():,}")
    print(f"at_junction: {df['junction_flag'].sum():,} / {len(df):,} records")

    # ── Carriageway blockage ──────────────────────────────────────────────────
    # Source: vehicle_type → footprint; location/junction → lanes
    df["violation_footprint_m"] = df["vehicle_type"].apply(get_vehicle_footprint)
    df["road_width_m"]          = df["lanes"] * _LANE_WIDTH_M
    df["carriageway_blocked_pct"] = (
        df["violation_footprint_m"] / df["road_width_m"]
    ).clip(upper=1.0)

    # ── Violation severity ────────────────────────────────────────────────────
    # Source: violation_type (JSON array column)
    df["violation_severity"] = df["violation_type"].apply(get_violation_severity)

    print("\n" + "=" * 60)
    print("TIER 2 — QUANTIFICATION ENGINE")
    print("=" * 60)

    # ── Congestion Impact Score (per-record) ──────────────────────────────────
    #
    #   score = carriageway_blocked_pct   [how much road is blocked]
    #         × violation_severity        [how disruptive the violation type is]
    #         × road_priority             [how critical the road is]
    #         × temporal_multiplier       [rush-hour amplification]
    #         × junction_multiplier       [intersection cross-flow penalty]
    #
    # All inputs are CSV-derived. No duration, no external coordinates.
    # Normalised to [0, 1] across the dataset.
    #
    temporal_mult = 1.0 + df["rush_flag"]   * 0.75
    junction_mult = 1.0 + df["junction_flag"] * 0.50

    raw = (
        df["carriageway_blocked_pct"]
        * df["violation_severity"]
        * df["road_priority"]
        * temporal_mult
        * junction_mult
    )
    df["congestion_impact_score"] = (raw / raw.max()).round(4)

    print(f"Congestion Impact Score per record:")
    print(f"  min={df['congestion_impact_score'].min():.4f}  "
          f"mean={df['congestion_impact_score'].mean():.4f}  "
          f"max={df['congestion_impact_score'].max():.4f}")

    # ── Junction-hour Effective Capacity Loss (aggregate) ─────────────────────
    #
    #   THE MONEY METRIC — no timestamps required.
    #
    #   At each (junction × hour) pair, compute:
    #     violation_count         = how many violations occurred
    #     mean_blockage_pct       = avg fraction of road blocked per violation
    #     effective_capacity_loss = mean_blockage_pct × violation_count
    #                             = cumulative fraction of road-capacity lost
    #                               over that junction-hour window
    #
    #   A single scooter blocking 11% of a lane = capacity_loss 0.11
    #   Ten scooters at the same spot = capacity_loss 1.14 (>1 full lane lost)
    #   One bus at a junction during rush hour can hit capacity_loss 0.43 alone
    #
    # Source: junction_name + created_datetime + vehicle_type + location
    #
    df["junction_hour_key"] = (
        df["junction_name"].astype(str) + "||h" + df["hour"].astype(str)
    )

    jh_agg = (
        df.groupby("junction_hour_key")
        .agg(
            junction_hour_violation_count=("carriageway_blocked_pct", "count"),
            junction_hour_mean_blockage  =("carriageway_blocked_pct", "mean"),
            junction_hour_max_severity   =("violation_severity",      "max"),
        )
        .reset_index()
    )
    jh_agg["junction_hour_capacity_loss"] = (
        jh_agg["junction_hour_mean_blockage"]
        * jh_agg["junction_hour_violation_count"]
    ).round(4)

    # Merge aggregate back onto per-record dataframe
    df = df.merge(jh_agg, on="junction_hour_key", how="left")

    top = jh_agg.nlargest(5, "junction_hour_capacity_loss")
    print(f"\nJunction-hour Effective Capacity Loss (top 5):")
    print(top[["junction_hour_key",
               "junction_hour_violation_count",
               "junction_hour_mean_blockage",
               "junction_hour_capacity_loss"]].to_string(index=False))

    print(f"\nFull shape: {df.shape}")
    print("Tier 1 + Tier 2 complete — ready for Tier 3.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INPUT_CSV = "d:/gridlock/jan to may police violation_anonymized791b166.csv"

    if not os.path.exists(INPUT_CSV):
        print("ERROR: CSV not found.")
    else:
        result = run_pipeline(INPUT_CSV, nrows=20000)

        show = [
            "location", "vehicle_type", "road_type",
            "rush_flag", "junction_flag",
            "carriageway_blocked_pct", "violation_severity",
            "congestion_impact_score",
            "junction_hour_violation_count",
            "junction_hour_capacity_loss",
        ]
        print("\n─── Sample output (10 rows) ───")
        print(result[show].head(10).to_string(index=False))

        print("\n─── Top 10 by congestion_impact_score ───")
        print(result[show].nlargest(10, "congestion_impact_score").to_string(index=False))

        print("\n─── Top 10 by junction_hour_capacity_loss ───")
        print(result[show].nlargest(10, "junction_hour_capacity_loss").to_string(index=False))
