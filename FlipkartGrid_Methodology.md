# Sugama Sanchara — Methodology & How It Works

**Flipkart GRiD 6.0 | Smart City Traffic Management**

---

## Pipeline: Data → Model → Output

---

### Stage 1: DATA

**Source:** Bengaluru Traffic Police violation records (anonymized)
- **298,907 records** spanning January to May (5 months)
- Fields: Junction ID, Police Station Zone, Vehicle Type, Timestamp, GPS Coordinates, Violation Type

**Preprocessing:**
- Geocoded 155 unique junctions to lat/lon using BTP station codes
- Extracted hour-of-day, day-of-week temporal features
- Filtered "No Junction" entries for spatial analysis
- Split: **Jan–Apr = Training data** | **May = Blind test data** (strict temporal holdout)

---

### Stage 2: MODEL

Three AI/ML components working together:

#### A. Spatial Hotspot Detection (DBSCAN Clustering)
- Algorithm: Density-Based Spatial Clustering of Applications with Noise
- Groups geographically adjacent junctions with similar violation patterns
- Output: 6 DBSCAN clusters identified as high-density enforcement zones
- **Why DBSCAN?** Unlike k-means, it doesn't require pre-specifying clusters and naturally identifies irregular Bengaluru street grid patterns

#### B. Enforcement Priority Score (EPS) — Core Innovation
- Formula: `EPS = f(mean_congestion_impact_score × violation_count × carriageway_block_pct)`
- Normalized 0–100 scale; reweights enforcement priority away from raw violation count
- **Key insight:** A single LGV blocking a carriageway scores higher than 10 scooters on a footpath
- Scooters = 56% of violations but only 12% of congestion impact → EPS deprioritizes them
- Result: 4 action tiers: CRITICAL (16), HIGH (23), MEDIUM, LOW

#### C. Temporal Violation Forecasting
- Historical average intensity by hour and day-of-week per police station zone
- Generates 24-hour intensity profiles for each of 54 zones
- Enables **proactive pre-positioning** of officers before peak hours occur

#### D. Anomaly Detection
- Identifies junction-hours where violation intensity spikes >3× above the baseline
- 160 anomalous spikes detected — these are "flash congestion" events needing rapid response

---

### Stage 3: OUTPUT

The Sugama Sanchara dashboard delivers five actionable intelligence layers:

| Panel | What It Shows | Who Acts |
|---|---|---|
| **Live Hotspots** | Top 20 junctions ranked by EPS with tow-truck dispatch priority | Traffic enforcement officers |
| **Hidden Violations** | 14 junctions with HIGH impact but LOW enforcement presence (enforcement blind spots) | Station commanders |
| **Future Predictions** | 24-hour forecast of violation intensity per police zone | Shift planners |
| **Why is it blocked?** | Root cause classification + junction-level action recommendation | Policy makers |
| **Overview** | City-wide KPIs: 60,858 violations, 155 junctions, 16 critical zones | Senior leadership |

---

### Generalization Proof

- **Training:** Jan–Apr data
- **Validation:** May data (completely unseen during model development)
- **Result:** EPS scores computed from Jan–Apr training data strongly correlate with May's actual congestion patterns
- This proves the model **generalizes** — it's not just memorizing past data, it's learning structural patterns in Bengaluru's traffic behavior

---

### Root Cause Classification (5 Categories)

| Category | Meaning | Example Junction | Recommended Action |
|---|---|---|---|
| Poor Road Design | Physical bottleneck — road too narrow for demand | Elite Junction | Infrastructure redesign |
| No Parking Alternative | Drivers park illegally because no legal option exists | Safina Plaza | Build multi-level parking |
| Enforcement Blind Spot | High congestion but historically zero enforcement | Police Quarters, Sultan Road | Increase patrol frequency |
| Demand-Supply Mismatch | Rush hour spike overwhelming road capacity | Subbanna Junction | Pre-position officers at peak |
| Repeat Commercial Offenders | Delivery trucks, vendors repeatedly blocking junctions | KR Market | Issue repeat-offender notices |
