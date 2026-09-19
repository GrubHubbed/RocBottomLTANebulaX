# LTA Smart Depot AI Condition Monitoring (CdM) System
## NebulaX Hackathon Problem Statement 3 (PS3) Dashboard

An Operations Control Center (OCC) dashboard designed for **Land Transport Authority (LTA) Singapore Smart Depot engineers and rolling stock technicians** to monitor train fleet health, run diagnostic models, store and inspect Excel datasets, and make active operational decisions on **what trains to pull out of mainline service for depot maintenance bay repair**.

---

## 🌟 Key Features

1. **Futuristic Cyber-HUD Operations Control Center**:
   - High-tech Singapore MRT depot theme with glowing status feeds, live depot clock (SGT UTC+8), fleet KPI decks, and glassmorphic telemetry cards.
2. **Customizable Multi-Solution Grid**:
   - Display **any combination of the 4 solutions simultaneously**:
     - **4-QUAD**: All 4 subsystems in a 2x2 grid for full tactical awareness.
     - **3-TRI**: 3 subsystems triage layout.
     - **2-DUAL**: Side-by-side or stacked split view.
     - **1-SOLO**: Maximized deep-dive diagnostic view.
   - Dynamic checkbox/pill toggles allow instantaneous filtering with layout preferences saved to `localStorage`.
3. **Official PS3 Subsystems & Parameter Headers**:
   - **Subsystem 1: Door Subsystem** (`PS3-SUB01-DOOR`)
     - Temporal segment detection & abnormal motor resistance classification.
     - Metric: **IoU-weighted F1 Score**.
     - Full parameter headers: `Datetime`, `Car Type`, `Car Number`, `Door Number`, `Motor current (mA)`, `Motor Voltage (10mV)`, `Motor back-EMF`, `Close command`, `Open command`, `DCSR`, `DCSL`, `DLSR`, `DLSL`, `Door Opened`, `Door Locked`, `Door leaf position`.
   - **Subsystem 2: ACV Subsystem** (`PS3-SUB02-ACV`)
     - Refrigerant leakage localization & car ranking across 8 carriages (`03|01|05|02|04|06|07|08`).
     - Metric: **Linear Rank-Decay Score**.
     - Full parameter headers: `Car model`, `Train number`, `Time`, per-car `Outside Temperature Sensor Reading`, `ACV Running Mode`, `Indoor Average Temperature`, `Load Halved`, `ACV Setting Mode`, etc.
   - **Subsystem 3: Rail Corrugation Subsystem** (`PS3-SUB03-RAIL`)
     - 3-class classification (`Normal`, `Side I`, `Side II`) from 64-channel axle-box vibration and shock.
     - Metric: **Macro F1 Score**.
     - Full parameter headers: Toothed-wheel train speed sensor pulse (Column 1) + 64 vibration & 64 shock channels across 8 cars x 8 axle boxes.
   - **Subsystem 4: SHM Subsystem** (`PS3-SUB04-SHM`)
     - Structural Health Monitoring dynamic stress cumulative fatigue damage regression.
     - Metric: **$\max(0, 1 - \text{MAPE})$**.
     - Full parameter headers: Dynamic stress time histories (MPa) for bogie frames and carbody bolsters under AW0 and AW4 tare/crush load conditions.
4. **Excel & CSV Storage Vault**:
   - Collects, uploads, validates, and permanently stores Excel (`.xlsx`, `.xls`) and CSV condition monitoring files.
   - In-app interactive **Data Grid Previewer**: inspect sheet names, column distributions, null values, and summary statistics without leaving the dashboard.
5. **Pluggable Python (`.py`) Model Runner**:
   - Pre-loaded with **4 stand-in baseline models**:
     - `standin_models/door_standin_model.py`
     - `standin_models/acv_standin_model.py`
     - `standin_models/rail_standin_model.py`
     - `standin_models/shm_standin_model.py`
   - Upload any custom `.py` script to run in a controlled subprocess (`--input <path> --output <path>`).
   - Streams stdout/stderr in real-time to in-card telemetry terminals.
   - **Hackathon Deliverables Packager**: One-click packaging of all generated prediction CSVs into `predictions.zip` as specified in the hackathon rubric.
6. **LTA Smart Depot Active Train Pull-Out Triage Board**:
   - Computes a composite **Train Health Index (THI)**:
     - `P1 - IMMEDIATE PULL-OUT`: Critical safety threshold exceeded; withdraw at next terminus.
     - `P2 - URGENT PULL-OUT (OFF-PEAK)`: High defect severity; route to maintenance bay after morning peak.
     - `P3 - SCHEDULED DEPOT CHECK`: Attention required during overnight servicing.
     - `P4 - FIT FOR SERVICE`: Nominal operational readiness.
   - Live **Depot Bay Allocation Matrix**: Tracks Bays 01 to 08 (Heavy Lift, HVAC Degas, Wheel Lathe, Structural NDT, etc.).
   - Generates official **LTA Depot Rolling Stock Maintenance Work Orders / Pull-Out Slips** with print/export capabilities.

---

## 🚀 Quickstart Guide

### 1. Requirements
Ensure Python 3.10+ is installed:
```bash
py -m pip install -r requirements.txt
```
*(Dependencies: `fastapi`, `uvicorn`, `pandas`, `openpyxl`, `python-multipart`)*

### 2. Launch the Application
Double-click `run_dashboard.bat` or run:
```bash
py app.py
```
Open your browser to: **[http://localhost:8000](http://localhost:8000)**

---

## 📂 Project Directory Structure

```
lta-smart-depot-dashboard/
├── app.py                         # FastAPI web application server & REST endpoints
├── requirements.txt               # Dependencies
├── run_dashboard.bat              # Windows batch launcher
├── README.md                      # Full documentation
├── backend/
│   ├── config.py                  # Storage paths and server configuration
│   ├── schemas_info.py            # Official PS3 parameter headers, metrics, and schemas
│   ├── file_manager.py            # Excel & CSV storage vault and sheet previewer
│   ├── runner_service.py          # Python .py execution engine & predictions packager
│   └── triage_engine.py           # LTA depot health scoring and bay allocation
├── standin_models/                # Stand-in baseline models conforming to PS3 specs
│   ├── door_standin_model.py      # Door segment & resistance classifier
│   ├── acv_standin_model.py       # ACV refrigerant leak localizer & ranker
│   ├── rail_standin_model.py      # Rail corrugation 3-class classifier
│   └── shm_standin_model.py       # SHM cumulative fatigue damage regressor
├── storage/                       # Persistent storage vault
│   ├── excel/                     # Uploaded & sample datasets (.xlsx, .csv)
│   ├── models/                    # Uploaded custom .py models
│   └── predictions/               # Generated predictions CSVs and predictions.zip
└── static/                        # Cyber-HUD frontend assets
    ├── index.html                 # Main dashboard UI
    ├── css/
    │   └── cyber_dashboard.css    # High-tech HUD styles, glassmorphism, responsive grids
    └── js/
        ├── app.js                 # Dashboard controller, layout switcher, HUD clock
        ├── file_vault.js          # Excel upload, previewer, sheet parser
        ├── model_runner.js        # Python runner, terminal streamer, zip packager
        └── triage.js              # Active train pull-out triage, bays, work orders
```
