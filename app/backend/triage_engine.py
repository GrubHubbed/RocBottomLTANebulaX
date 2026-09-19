"""
triage_engine.py - LTA Smart Depot Active Train Pull-Out Triage Engine
Analyzes condition monitoring data across the 4 subsystems to prioritize rolling stock
maintenance bay induction, depot track allocation, and generate operational work orders.
"""

from datetime import datetime
from typing import Dict, Any, List

class DepotTriageEngine:
    def __init__(self):
        # Simulated depot bays at Bishan / Tuas / Mandai / Kim Chuan
        self.depot_bays = [
            {"bay_id": "BAY-01", "type": "Heavy Lift / Bogie Drop", "status": "OCCUPIED", "train_id": "T-118", "eta_hours": 3.5},
            {"bay_id": "BAY-02", "type": "General Overhaul", "status": "AVAILABLE", "train_id": None, "eta_hours": 0.0},
            {"bay_id": "BAY-03", "type": "Door Mechanism & Actuator Bench", "status": "AVAILABLE", "train_id": None, "eta_hours": 0.0},
            {"bay_id": "BAY-04", "type": "HVAC Degas & Refrigerant Recharge", "status": "AVAILABLE", "train_id": None, "eta_hours": 0.0},
            {"bay_id": "BAY-05", "type": "Underfloor Wheel Lathe", "status": "OCCUPIED", "train_id": "T-204", "eta_hours": 1.2},
            {"bay_id": "BAY-06", "type": "Structural NDT & Weld Ultrasonic", "status": "AVAILABLE", "train_id": None, "eta_hours": 0.0},
            {"bay_id": "BAY-07", "type": "Routine 30-Day Inspection", "status": "OCCUPIED", "train_id": "T-329", "eta_hours": 4.0},
            {"bay_id": "BAY-08", "type": "Fast Turnaround Siding", "status": "AVAILABLE", "train_id": None, "eta_hours": 0.0}
        ]

    def get_depot_fleet_triage(self) -> Dict[str, Any]:
        """Generate active fleet health matrix and pull-out recommendation list for LTA depot controllers."""
        trains = [
            {
                "train_id": "T-315",
                "car_count": 8,
                "fleet_line": "East-West Line (EWL)",
                "depot": "Tuas Depot",
                "door_score": 42.0,
                "door_status": "CRITICAL",
                "door_detail": "Car 03 Door 4R: Motor current spike 2,450 mA during closing cycle (Obstruction / roller bearing breakdown)",
                "acv_score": 38.0,
                "acv_status": "CRITICAL",
                "acv_detail": "Car 03 HVAC: Refrigerant leak confirmed (Rank #1, Temp delta +4.8°C above saloon setpoint)",
                "rail_score": 78.0,
                "rail_status": "NOMINAL",
                "rail_detail": "Normal axle-box vibration signature across all 8 wheelsets",
                "shm_score": 82.0,
                "shm_status": "NOMINAL",
                "shm_detail": "Cumulative fatigue damage D = 0.22 (Well within safety threshold 1.0)",
                "composite_thi": 44.5,
                "triage_priority": "P1 - IMMEDIATE PULL-OUT",
                "recommended_action": "Withdraw from service at Tuas Link Terminus. Shunt to Bay 04 for immediate HVAC degassing/recharge and Bay 03 for Door 4R roller replacement.",
                "allocated_bay": "BAY-04 (HVAC Degas & Recharge)",
                "assigned_team": "Team Alpha (HVAC & Mechanical)"
            },
            {
                "train_id": "T-242",
                "car_count": 6,
                "fleet_line": "Circle Line (CCL)",
                "depot": "Kim Chuan Depot",
                "door_score": 88.0,
                "door_status": "NOMINAL",
                "door_detail": "All door leaf cycles nominal (open/close profiles within 0.1s tolerance)",
                "acv_score": 92.0,
                "acv_status": "NOMINAL",
                "acv_detail": "Saloon temperatures consistent at 22.4°C across Cars 01-06",
                "rail_score": 35.0,
                "rail_status": "CRITICAL",
                "rail_detail": "Severe Side I periodic corrugation contact resonance on Car 02 & Car 04 axle boxes (RMS vib 2.15 m/s²)",
                "shm_score": 64.0,
                "shm_status": "ATTENTION",
                "shm_detail": "Moderate cyclic stress amplitudes on bogie primary bolster (D = 0.58)",
                "composite_thi": 53.2,
                "triage_priority": "P2 - URGENT PULL-OUT (OFF-PEAK)",
                "recommended_action": "Withdraw after morning passenger peak. Direct to Underfloor Wheel Lathe (Bay 05) for wheel reprofiling to suppress corrugation excitation.",
                "allocated_bay": "BAY-05 (Underfloor Wheel Lathe)",
                "assigned_team": "Team Gamma (Bogie & Running Gear)"
            },
            {
                "train_id": "T-109",
                "car_count": 8,
                "fleet_line": "North-South Line (NSL)",
                "depot": "Bishan Depot",
                "door_score": 90.0,
                "door_status": "NOMINAL",
                "door_detail": "Nominal door operations; zero switch actuation latency",
                "acv_score": 85.0,
                "acv_status": "NOMINAL",
                "acv_detail": "Thermal variation within +/- 0.6°C across all carriages",
                "rail_score": 82.0,
                "rail_status": "NOMINAL",
                "rail_detail": "Vibration spectra nominal across both Side I and Side II",
                "shm_score": 31.0,
                "shm_status": "CRITICAL",
                "shm_detail": "High cumulative fatigue damage D = 0.89 detected on Car 05 bogie frame side beam weldment",
                "composite_thi": 58.6,
                "triage_priority": "P2 - URGENT PULL-OUT (OFF-PEAK)",
                "recommended_action": "Schedule bay induction for Non-Destructive Ultrasonic Weld Scan (NDT) to inspect micro-fissure propagation on Car 05 bogie.",
                "allocated_bay": "BAY-06 (Structural NDT & Weld Ultrasonic)",
                "assigned_team": "Team Delta (Structural & NDT)"
            },
            {
                "train_id": "T-405",
                "car_count": 4,
                "fleet_line": "Thomson-East Coast Line (TEL)",
                "depot": "Mandai Depot",
                "door_score": 68.0,
                "door_status": "ATTENTION",
                "door_detail": "Car 01 Door 2L closing stroke delayed by 0.4s; motor back-EMF slightly degraded",
                "acv_score": 74.0,
                "acv_status": "ATTENTION",
                "acv_detail": "Car 04 running at 100% cooling continuously with slight temperature divergence (+1.9°C)",
                "rail_score": 90.0,
                "rail_status": "NOMINAL",
                "rail_detail": "Axle-box vibration frequencies within baseline limits",
                "shm_score": 95.0,
                "shm_status": "NOMINAL",
                "shm_detail": "Dynamic stress low (D = 0.14)",
                "composite_thi": 76.5,
                "triage_priority": "P3 - SCHEDULED DEPOT CHECK",
                "recommended_action": "Route for routine overnight maintenance siding check. Inspect Door 2L guide rail lubricant and verify ACV refrigerant pressure gauges.",
                "allocated_bay": "BAY-03 (Door Mechanism & Actuator Bench)",
                "assigned_team": "Team Beta (Light Maintenance)"
            },
            {
                "train_id": "T-151",
                "car_count": 8,
                "fleet_line": "North-South Line (NSL)",
                "depot": "Bishan Depot",
                "door_score": 96.0,
                "door_status": "NOMINAL",
                "door_detail": "All switches (DCSR, DCSL, DLSR, DLSL) operating within millisecond benchmarks",
                "acv_score": 98.0,
                "acv_status": "NOMINAL",
                "acv_detail": "Optimum cooling efficiency across all 8 carriages",
                "rail_score": 94.0,
                "rail_status": "NOMINAL",
                "rail_detail": "Vibration harmonic ratio 1.02 (Optimal smooth track contact)",
                "shm_score": 97.0,
                "shm_status": "NOMINAL",
                "shm_detail": "Cumulative fatigue damage D = 0.08 (Pristine structural condition)",
                "composite_thi": 96.2,
                "triage_priority": "P4 - FIT FOR REVENUE SERVICE",
                "recommended_action": "Maintain full scheduled revenue service without restrictions.",
                "allocated_bay": "STABLING SIDING 14",
                "assigned_team": "Depot Ops"
            }
        ]

        # Calculate high-level summary statistics
        total_trains = len(trains) + 137 # Total 142 simulated fleet
        critical_count = sum(1 for t in trains if "P1" in t["triage_priority"])
        urgent_count = sum(1 for t in trains if "P2" in t["triage_priority"])
        scheduled_count = sum(1 for t in trains if "P3" in t["triage_priority"])
        healthy_count = total_trains - critical_count - urgent_count - scheduled_count

        return {
            "fleet_summary": {
                "total_fleet": total_trains,
                "in_revenue_service": total_trains - 24,
                "depot_stabled": 24,
                "critical_pullout_required": critical_count,
                "urgent_offpeak_pullout": urgent_count,
                "scheduled_check": scheduled_count,
                "healthy_trains": healthy_count,
                "bay_occupancy": "3 / 8 Bays Occupied (5 Available)",
                "last_sync": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "bays": self.depot_bays,
            "triage_queue": trains
        }

    def generate_work_order(self, train_id: str) -> Dict[str, Any]:
        """Generate an official LTA Depot Maintenance Work Order for a specific train."""
        triage_data = self.get_depot_fleet_triage()
        target_train = next((t for t in triage_data["triage_queue"] if t["train_id"] == train_id), None)
        
        if not target_train:
            target_train = triage_data["triage_queue"][0]

        now_str = datetime.now().strftime("%Y%m%d")
        wo_number = f"LTA-SMARTDEPOT-WO-{now_str}-{target_train['train_id'].replace('-', '')}"

        return {
            "work_order_id": wo_number,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S SGT"),
            "issuing_authority": "Land Transport Authority (LTA) Rail Asset Operations",
            "train_id": target_train["train_id"],
            "fleet_line": target_train["fleet_line"],
            "depot": target_train["depot"],
            "composite_thi": f"{target_train['composite_thi']}%",
            "priority": target_train["triage_priority"],
            "allocated_bay": target_train["allocated_bay"],
            "assigned_team": target_train["assigned_team"],
            "subsystem_diagnostics": {
                "door": {"score": target_train["door_score"], "status": target_train["door_status"], "finding": target_train["door_detail"]},
                "acv": {"score": target_train["acv_score"], "status": target_train["acv_status"], "finding": target_train["acv_detail"]},
                "rail": {"score": target_train["rail_score"], "status": target_train["rail_status"], "finding": target_train["rail_detail"]},
                "shm": {"score": target_train["shm_score"], "status": target_train["shm_status"], "finding": target_train["shm_detail"]}
            },
            "operational_directive": target_train["recommended_action"],
            "authorization": {
                "depot_master": "APPROVED - Automated Safety Interlock Cleared",
                "chief_rolling_stock_engineer": "PENDING PHYSICAL ACCEPTANCE"
            }
        }

# Global singleton
triage_engine = DepotTriageEngine()
