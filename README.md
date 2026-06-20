# 🛡️ Gridlock Sentry: AI-Driven Parking Intelligence

> **Flipkart GRiD 6.0 Hackathon Submission**  
> **Theme:** Traffic Analysis and Congestion Detection  
> **Live Demo:** [Insert Your Render URL Here]

## 🚦 The Problem
On-street illegal parking, double parking, and spillover congestion near commercial areas and metro stations severely choke Bengaluru's carriageways. Currently, traffic enforcement is reactive and patrol-based. Police lack the data to identify whether a congestion hotspot is caused by a structural design flaw, a rush-hour demand spike, or simply an enforcement blind spot. 

## 💡 Our Solution
**Gridlock Sentry** is an end-to-end, AI-driven traffic intelligence platform. It transforms raw, anonymized police violation data into actionable, real-time enforcement priorities using mathematical clustering and predictive forecasting.

Instead of just showing a generic heatmap, Gridlock Sentry computes an **Enforcement Priority Score (EPS)** to instantly tell traffic dispatchers *where* to send a tow truck to clear the most severe bottlenecks, and *why* the road is blocked.

---

## ✨ Key Features

- **📍 Spatial Hotspot Clustering (DBSCAN)**  
  Groups thousands of raw violation coordinates into unique, high-density junction clusters using Haversine distance metrics.
- **🚨 Enforcement Priority Score (EPS)**  
  A proprietary `0-100` scoring engine that ranks junctions by combining historical congestion impact, carriageway blockage loss, anomaly spikes, and real-time temporal forecasting.
- **🧠 Root Cause Classifier**  
  An automated rule-based engine that categorizes *why* a junction is blocked (e.g., *Structural Design Flaw*, *Rush-Hour Demand Spike*, *No Parking Alternative*).
- **👁️ Systemic Blind-Spot Detection ("Yellow Zones")**  
  Automatically highlights areas with massive traffic impact but abnormally low ticketing rates, directing police to hidden problem areas.
- **📊 Real-Time Interactive Dashboard**  
  A premium, highly responsive UI tailored for dispatchers to instantly view Live Hotspots, Future Predictions, and Junction Impact Reports.

---

## 📈 Mathematical Rigour & Generalization
To ensure the AI is scalable and predictive (not just memorizing past data), we utilized a strict hold-out evaluation framework:
1. **Train Phase:** Analyzed data from January to April to build spatial hotspot bounds and historical hour-of-week averages.
2. **Test Phase:** Blindly evaluated the model against May's data. 
3. **Result:** Achieved high Pearson correlation and proved that our EPS metrics persistently track future "Yellow Zone" bottlenecks.

---

## 🛠️ Technology Stack
- **Backend:** Python, FastAPI, Uvicorn
- **Database:** SQLite (Lightweight, embedded analytics)
- **Data Science:** Pandas, NumPy, Scikit-learn (DBSCAN), SciPy
- **Frontend:** HTML5, CSS3, Vanilla JS, Leaflet.js (Mapping), Chart.js (Analytics)
- **Deployment:** Render (Free Tier Web Service)

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your machine.

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/your-username/sugama-Sanchara.git
cd sugama-Sanchara
pip install -r requirements.txt
```

### 3. Data Setup
Ensure the provided dataset (`jan to may police violation_anonymized791b166.csv`) is placed in the root directory. 
*(Note: This file is intentionally ignored in `.gitignore` due to size limits).*

### 4. Run the Intelligence Pipeline
To regenerate the AI clusters and EPS rankings:
```bash
python gridlock_sentry_intelligence.py
```

### 5. Start the Server
```bash
uvicorn gridlock_sentry_api:app --host 0.0.0.0 --port 8000
```

### 6. View the Dashboard
Open your browser and navigate to:
- **Dashboard:** `http://localhost:8000/`
- **Swagger API Docs:** `http://localhost:8000/docs`

---
*Built with ❤️ for Bengaluru's future mobility.*
