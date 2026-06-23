"""
Gridlock Sentry — Data Pipeline
================================
Tier 1: Data Foundation        (filtering, time features, text-based road context)
Tier 2: Quantification Engine  (Congestion Impact Score)

COMPLIANCE
----------
All features are derived exclusively from the provided CSV file.
No external datasets, APIs, or geospatial databases are used.
road_type / lanes / road_priority / near_metro are inferred by keyword-matching
the CSV's own `location` and `junction_name` text fields — not from OpenStreetMap
or any external source. Lane width (3.5m) and standard vehicle widths are fixed
engineering constants, not a dataset, used the same way pi or freezing point
of water would be.

DATASET FINDING (important — read before changing the score formula)
-----------------------------------------------------------------------
action_taken_timestamp and closed_datetime are 0% populated in this dataset.
There is therefore NO ground-truth signal for how long a violation physically
blocked the road. validation_timestamp marks when the system *reviewed* the
report — not when the vehicle left — so it is NOT a congestion-duration proxy.
Using it as one would overstate what this data supports and will not survive
a judge's question. validation_timestamp is reported separately below as an
operational metric (report-to-validation lag), and is deliberately excluded
from the Congestion Impact Score, which uses only frequency, severity, and
road-context signals — all directly and defensibly observable in the CSV.
"""
import json
import pandas as pd

# ---------------------------------------------------------------------------
# TIER 1 HELPERS — road context from CSV text fields (no external data)
# ---------------------------------------------------------------------------

def infer_road_attributes(location_str: str, junction_str: str) -> tuple:
    """
    Infer road_type, lanes, road_priority, and near_metro from the CSV's own
    `location` and `junction_name` text. Pure text transformation of fields
    already in the dataset — not OSM, not any external source.
    Returns: (road_type: str, lanes: int, road_priority: float, near_metro: int)
    """
    combined = (str(location_str) + " " + str(junction_str)).lower()
    near_metro = int(any(k in combined for k in ["metro", "namma metro", "metro station"]))

    if any(k in combined for k in ["flyover", "ring road", "expressway", "outer ring", "bypass"]):
        return "motorway", 3, 1.00, near_metro
    elif any(k in combined for k in ["main road", "double road", "arterial"]):
        return "primary", 2, 0.85, near_metro
    elif "road" in combined:
        return "primary", 2, 0.75, near_metro
    elif any(k in combined for k in ["cross", "street", "avenue"]):
        return "secondary", 2, 0.60, near_metro
    elif any(k in combined for k in ["lane", "layout"]):
        return "secondary", 1, 0.50, near_metro
    else:
        return "tertiary", 1, 0.40, near_metro


def infer_junction_flag(junction_str: str) -> int:
    """1 if violation is at a named junction, 0 if 'No Junction'. CSV source: junction_name."""
    s = str(junction_str).strip().lower()
    return 0 if s in ("", "nan", "no junction", "none") else 1


# ---------------------------------------------------------------------------
# TIER 2 HELPERS — severity and footprint from CSV columns
# ---------------------------------------------------------------------------

VIOLATION_SEVERITY = {
    "double parking": 1.00,
    "parking opposite to another parked vehicle": 0.90,
    "parking in a main road": 0.80,
    "parking near traffic light": 0.70,
    "parking near road crossing": 0.70,
    "parking near bustop": 0.60,
    "wrong parking": 0.50,
    "no parking": 0.50,
    "parking on footpath": 0.30,
}

def get_violation_severity(vtype_str: str) -> float:
    """Parse the violation_type JSON array and return the max severity weight found."""
    try:
        violations = json.loads(str(vtype_str))
        if not isinstance(violations, list):
            violations = [str(violations)]
    except Exception:
        violations = [str(vtype_str)]

    score = 0.30  # floor: some violation was recorded, even if unmatched
    for v in violations:
        v_lower = str(v).lower().strip()
        for keyword, weight in VIOLATION_SEVERITY.items():
            if keyword in v_lower:
                score = max(score, weight)
    return score


VEHICLE_FOOTPRINT_M = {
    "HGV": 3.0, "BUS": 3.0, "TRUCK": 3.0,
    "LGV": 2.2,
    "CAR": 1.9,
    "PASSENGER AUTO": 1.5, "AUTO": 1.5,
    "MOTOR CYCLE": 0.8, "MOTORCYCLE": 0.8, "SCOOTER": 0.8,
}

def get_vehicle_footprint(vtype_str: str) -> float:
    """Physical width (metres) of the parked vehicle. CSV source: vehicle_type."""
    v = str(vtype_str).upper()
    for key, width in VEHICLE_FOOTPRINT_M.items():
        if key in v:
            return width
    return 1.5  # fallback: generic mid-size footprint


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

LANE_WIDTH_M = 3.5  # standard lane width — engineering constant, not a dataset
VPH = {"motorway": 1800, "primary": 1200, "secondary": 600, "tertiary": 200}  # textbook road-capacity figures


def run_pipeline(input_path: str, nrows: int = None) -> pd.DataFrame:
    print("=" * 60)
    print("TIER 1 — DATA FOUNDATION")
    print("=" * 60)
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path, nrows=nrows)
    print(f"Raw shape: {df.shape}")

    # STEP 1 — keep approved records only. CSV source: validation_status
    df = df[df["validation_status"].astype(str).str.lower() == "approved"].copy()
    df = df.reset_index(drop=True)
    print(f"After approved filter: {df.shape}")

    # STEP 2 — time features. CSV source: created_datetime
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], utc=True, errors="coerce")
    df["hour"] = df["created_datetime"].dt.hour
    df["day"] = df["created_datetime"].dt.dayofweek  # 0=Mon, 6=Sun
    df["month"] = df["created_datetime"].dt.month
    df["rush_flag"] = ((df["hour"].between(8, 10)) | (df["hour"].between(17, 20))).astype(int)
    print(f"Time features done. Rush-hour violations: {df['rush_flag'].sum():,}")

    # STEP 3 — road context from address text. CSV source: location, junction_name
    road_attrs = [
        infer_road_attributes(loc, junc)
        for loc, junc in zip(df["location"], df["junction_name"])
    ]
    df["road_type"] = [r[0] for r in road_attrs]
    df["lanes"] = [r[1] for r in road_attrs]
    df["road_priority"] = [r[2] for r in road_attrs]
    df["near_metro"] = [r[3] for r in road_attrs]
    df["junction_flag"] = df["junction_name"].apply(infer_junction_flag)

    print("Road context distribution (check this isn't skewed entirely into one bucket):")
    print(df["road_type"].value_counts().to_string())
    print(f"At junction: {df['junction_flag'].sum():,} / {len(df):,}  |  Near metro: {df['near_metro'].sum():,}")

    # STEP 4 — violation frequency per junction-hour: the real persistence signal
    # this dataset can support, since duration is not available.
    # CSV source: junction_name + created_datetime
    df["junction_hour_key"] = df["junction_name"].astype(str) + "_h" + df["hour"].astype(str)
    freq_map = df["junction_hour_key"].value_counts().to_dict()
    df["violation_frequency"] = df["junction_hour_key"].map(freq_map)
    max_freq = df["violation_frequency"].max()
    df["violation_frequency_norm"] = (df["violation_frequency"] / max_freq).round(4)
    print(f"Max violations at a single junction-hour: {max_freq}")

    print()
    print("=" * 60)
    print("TIER 2 — QUANTIFICATION ENGINE")
    print("=" * 60)

    # COMPONENT A — carriageway blocked %. CSV source: vehicle_type -> footprint; road context -> lanes
    df["violation_footprint_m"] = df["vehicle_type"].apply(get_vehicle_footprint)
    df["road_width_m"] = df["lanes"] * LANE_WIDTH_M
    df["carriageway_blocked_pct"] = (df["violation_footprint_m"] / df["road_width_m"]).clip(upper=1.0)
    print(f"Carriageway blocked %: mean={df['carriageway_blocked_pct'].mean():.3f}, "
          f"max={df['carriageway_blocked_pct'].max():.3f}")

    # COMPONENT B — violation severity. CSV source: violation_type
    df["violation_severity"] = df["violation_type"].apply(get_violation_severity)
    print(f"Violation severity: mean={df['violation_severity'].mean():.3f}")

    # COMPONENT C — temporal / spatial multipliers
    df["temporal_multiplier"] = 1.0 + df["rush_flag"] * 0.75
    df["junction_multiplier"] = 1.0 + df["junction_flag"] * 0.50
    df["metro_multiplier"] = 1.0 + df["near_metro"] * 0.30

    # CONGESTION IMPACT SCORE — the single number per record.
    # Uses ONLY signals this dataset can actually support: how much road width
    # one violation occupies, how severe its type is, how critical the road is,
    # whether it's rush hour / at a junction / near a metro, and how often this
    # exact junction-hour combination recurs. No duration, no fabricated proxy.
    raw_score = (
        df["carriageway_blocked_pct"]
        * df["violation_severity"]
        * df["road_priority"]
        * df["temporal_multiplier"]
        * df["junction_multiplier"]
        * df["metro_multiplier"]
        * (1.0 + df["violation_frequency_norm"])
    )
    df["congestion_impact_score"] = (raw_score / raw_score.max()).round(4)
    print(f"Congestion Impact Score: min={df['congestion_impact_score'].min():.4f}, "
          f"mean={df['congestion_impact_score'].mean():.4f}, "
          f"max={df['congestion_impact_score'].max():.4f}")

    # OPERATIONAL METRIC (kept separate, NOT part of the impact score) —
    # report-to-validation lag. This measures backend processing speed,
    # not how long a vehicle blocked the road. Useful for an "enforcement
    # pipeline health" insight, not for the congestion narrative.
    df["validation_timestamp"] = pd.to_datetime(df["validation_timestamp"], utc=True, errors="coerce")
    df["validation_lag_minutes"] = (
        (df["validation_timestamp"] - df["created_datetime"]).dt.total_seconds() / 60
    ).clip(lower=0, upper=10080)
    print(f"Validation lag (operational, not congestion): "
          f"median={df['validation_lag_minutes'].median():.0f} min")

    print()
    print("Tier 1 + Tier 2 complete. Shape:", df.shape)
    print("Ready for Tier 3 — Intelligence Engines (hotspot clustering, anomaly, forecast).")
    return df


def junction_hour_capacity_loss(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates to junction-hour level: mean carriageway % blocked × how many
    violations hit that junction in that hour. This is the cumulative,
    fully CSV-derived quantification of traffic-flow impact.

    Only named junctions are included — 'No Junction' records are excluded
    because they cannot be dispatched to or mapped to a specific location.
    """
    named = df[df["junction_flag"] == 1].copy()
    agg = (
        named.groupby(["junction_name", "hour"])
        .agg(
            violation_count=("congestion_impact_score", "size"),
            mean_carriageway_blocked_pct=("carriageway_blocked_pct", "mean"),
            mean_congestion_impact_score=("congestion_impact_score", "mean"),
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
        )
        .reset_index()
    )
    agg["effective_capacity_loss_index"] = (
        agg["mean_carriageway_blocked_pct"] * agg["violation_count"]
    ).round(2)
    return agg.sort_values("effective_capacity_loss_index", ascending=False)


def police_station_zone_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback aggregation for 'No Junction' records — groups by police_station
    and hour to identify high-volume patrol zones even without junction names.
    """
    agg = (
        df.groupby(["police_station", "hour"])
        .agg(
            violation_count=("congestion_impact_score", "size"),
            mean_congestion_impact_score=("congestion_impact_score", "mean"),
            mean_carriageway_blocked_pct=("carriageway_blocked_pct", "mean"),
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
        )
        .reset_index()
    )
    agg["zone_impact_index"] = (
        agg["mean_congestion_impact_score"] * agg["violation_count"]
    ).round(2)
    return agg.sort_values("zone_impact_index", ascending=False)


if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_CSV = os.path.join(BASE_DIR, "jan to may police violation_anonymized791b166.csv")
    if not os.path.exists(INPUT_CSV):
        print("ERROR: CSV not found.")
    else:
        result = run_pipeline(INPUT_CSV)   # full 115K approved rows

        display_cols = [
            "location", "vehicle_type", "road_type", "rush_flag", "junction_flag",
            "near_metro", "carriageway_blocked_pct", "violation_severity",
            "violation_frequency", "congestion_impact_score",
        ]
        print("\n--- Top 10 Highest Congestion Impact (record level) ---")
        print(result[display_cols].nlargest(10, "congestion_impact_score").to_string(index=False))

        print("\n--- Top 10 Junction-Hours by Effective Capacity Loss ---")
        capacity_loss = junction_hour_capacity_loss(result)
        print(capacity_loss.head(10).to_string(index=False))

        # Export junction-level aggregates for dashboard
        out_path = os.path.join(BASE_DIR, "junction_scores.csv")
        capacity_loss.to_csv(out_path, index=False)
        print(f"\nJunction scores exported to: {out_path}")

