import pytest
import pandas as pd
from gridlock_sentry_pipeline import (
    infer_road_attributes,
    get_violation_severity,
    get_vehicle_footprint
)
from gridlock_sentry_intelligence import classify_root_cause

def test_infer_road_attributes():
    # Test 1: Metro and Expressway
    rt, lanes, prio, metro = infer_road_attributes("Namma Metro Station near ring road", "Outer Ring Road Junction")
    assert rt == "motorway"
    assert lanes == 3
    assert prio == 1.00
    assert metro == 1

    # Test 2: Standard cross street
    rt, lanes, prio, metro = infer_road_attributes("10th Cross", "No Junction")
    assert rt == "secondary"
    assert lanes == 2
    assert prio == 0.60
    assert metro == 0

def test_violation_severity():
    # Test 1: Array string parsing
    score = get_violation_severity('["Double parking", "No parking"]')
    assert score == 1.00  # Double parking maxes it out

    # Test 2: String mismatch fallback
    score = get_violation_severity("Random new violation")
    assert score == 0.30  # Floor severity

def test_vehicle_footprint():
    # Test 1: Heavy Goods
    assert get_vehicle_footprint("HGV") == 3.0
    # Test 2: Passenger Car
    assert get_vehicle_footprint("CAR") == 1.9
    # Test 3: Motorcycle
    assert get_vehicle_footprint("MOTOR CYCLE") == 0.8

def test_root_cause_classifier():
    # Test 1: Structural Design (High volume + high active hours)
    row = {
        "active_hours": 20,
        "violation_count": 60,
        "rush_hour_pct": 0.2,
        "mean_impact_score": 0.5,
        "top_road_type": "primary"
    }
    assert classify_root_cause(row) == "STRUCTURAL_DESIGN"

    # Test 2: Enforcement Blind Spot (High count, low impact)
    row2 = {
        "active_hours": 10,
        "violation_count": 25,
        "rush_hour_pct": 0.2,
        "mean_impact_score": 0.10,
        "top_road_type": "secondary"
    }
    assert classify_root_cause(row2) == "ENFORCEMENT_BLIND_SPOT"

    # Test 3: Rush Hour Demand Spike
    row3 = {
        "active_hours": 5,
        "violation_count": 40,
        "rush_hour_pct": 0.50, # 50% in rush hour
        "mean_impact_score": 0.5,
        "top_road_type": "tertiary"
    }
    assert classify_root_cause(row3) == "RUSH_HOUR_DEMAND_SPIKE"

if __name__ == "__main__":
    print("Running Core Unit Tests (Test Cases)...")
    test_infer_road_attributes()
    test_violation_severity()
    test_vehicle_footprint()
    test_root_cause_classifier()
    print("ALL TEST CASES PASSED SUCCESSFULLY!")
