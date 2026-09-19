"""
verify_dashboard.py - Automated Verification Suite for LTA Smart Depot Dashboard
Validates schemas, file vault, stand-in model execution, predictions packaging, and triage engine.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from backend.config import BASE_DIR, STORAGE_DIR, EXCEL_DIR, PREDICTIONS_DIR
from backend.schemas_info import SUBSYSTEM_SCHEMAS
from backend.file_manager import file_manager
from backend.runner_service import runner_service
from backend.triage_engine import triage_engine

def test_schemas():
    print("[TEST 1/5] Verifying Subsystem Schemas & Official Headers...")
    assert len(SUBSYSTEM_SCHEMAS) == 4, f"Expected 4 subsystems, found {len(SUBSYSTEM_SCHEMAS)}"
    for sub_id in ["door", "acv", "rail", "shm"]:
        sub = SUBSYSTEM_SCHEMAS[sub_id]
        print(f"  ✔ Subsystem '{sub_id}': {sub['name']} | Metric: {sub['primary_metric']} | Headers: {len(sub['headers'])}")
        assert len(sub["headers"]) > 0, f"Headers missing for {sub_id}"
    print("  -> Schemas verified successfully!\n")

def test_file_vault():
    print("[TEST 2/5] Verifying Excel & CSV File Vault...")
    files = file_manager.list_files()
    print(f"  ✔ Stored files in vault: {len(files)}")
    for f in files:
        print(f"    - {f['filename']} ({f['size_kb']} KB, {f['subsystem'].upper()}, {f['sheet_count']} sheet(s))")
    
    # Test preview on an Excel file
    excel_file = next((f["filename"] for f in files if f["filename"].endswith(".xlsx")), None)
    if excel_file:
        preview = file_manager.preview_file(excel_file)
        assert preview["status"] == "success", f"Preview failed: {preview.get('message')}"
        print(f"  ✔ Previewed '{excel_file}': {preview['total_rows']} rows, {preview['total_cols']} cols, Sheets: {preview['sheets']}")
    print("  -> File vault verified successfully!\n")

def test_model_runner():
    print("[TEST 3/5] Verifying Python Model Script Runner & Stand-in Models...")
    models = runner_service.list_available_scripts()
    print(f"  ✔ Available models: {len(models)}")
    for m in models:
        print(f"    - {m['name']} ({m['type']}, Subsystem: {m['subsystem']})")

    # 1. Test Door stand-in execution
    print("  -> Running Door Stand-in Model...")
    res_door = runner_service.execute_script("door_standin_model.py", "door_telemetry_stream.csv", "door")
    assert res_door["status"] == "success", f"Door model failed: {res_door}"
    print(f"    ✔ Door model executed in {res_door['duration_sec']}s. Output: {res_door['output_file']}")

    # 2. Test ACV stand-in execution on Excel
    print("  -> Running ACV Stand-in Model...")
    res_acv = runner_service.execute_script("acv_standin_model.py", "acv_fleet_case_01.xlsx", "acv")
    assert res_acv["status"] == "success", f"ACV model failed: {res_acv}"
    print(f"    ✔ ACV model executed in {res_acv['duration_sec']}s. Output: {res_acv['output_file']}")

    # 3. Test Rail stand-in execution
    print("  -> Running Rail Stand-in Model...")
    res_rail = runner_service.execute_script("rail_standin_model.py", "rail_axlebox_vibration_sample.csv", "rail")
    assert res_rail["status"] == "success", f"Rail model failed: {res_rail}"
    print(f"    ✔ Rail model executed in {res_rail['duration_sec']}s. Output: {res_rail['output_file']}")

    # 4. Test SHM stand-in execution
    print("  -> Running SHM Stand-in Model...")
    res_shm = runner_service.execute_script("shm_standin_model.py", "shm_dynamic_stress_segment.csv", "shm")
    assert res_shm["status"] == "success", f"SHM model failed: {res_shm}"
    print(f"    ✔ SHM model executed in {res_shm['duration_sec']}s. Output: {res_shm['output_file']}")

    print("  -> All 4 stand-in models executed and produced PS3 prediction outputs!\n")

def test_zip_packaging():
    print("[TEST 4/5] Verifying Hackathon Predictions.zip Packager...")
    res = runner_service.package_predictions_zip()
    assert res["status"] == "success", f"Zip packaging failed: {res}"
    print(f"  ✔ Created predictions.zip ({res['size_kb']} KB) containing {res['total_files']} files: {res['files_included']}")
    assert (PREDICTIONS_DIR / "predictions.zip").exists()
    print("  -> Predictions zip packager verified successfully!\n")

def test_triage_engine():
    print("[TEST 5/5] Verifying LTA Depot Train Pull-Out Triage Engine...")
    triage_data = triage_engine.get_depot_fleet_triage()
    summary = triage_data["fleet_summary"]
    print(f"  ✔ Fleet Summary: Total {summary['total_fleet']}, Active {summary['in_revenue_service']}, Critical {summary['critical_pullout_required']}")
    assert len(triage_data["triage_queue"]) > 0
    
    top_train = triage_data["triage_queue"][0]
    print(f"  ✔ Top Pull-Out Recommendation: {top_train['train_id']} | Priority: {top_train['triage_priority']} | Health: {top_train['composite_thi']}%")
    print(f"    Allocated Bay: {top_train['allocated_bay']}")

    # Test Work Order generation
    wo = triage_engine.generate_work_order(top_train["train_id"])
    print(f"  ✔ Generated Work Order: {wo['work_order_id']} for {wo['train_id']}")
    assert wo["work_order_id"].startswith("LTA-SMARTDEPOT-WO-")
    print("  -> Triage engine and work orders verified successfully!\n")

if __name__ == "__main__":
    print("=====================================================================")
    print(" STARTING VERIFICATION: LTA SMART DEPOT OCC DASHBOARD (NEBULA-X PS3)")
    print("=====================================================================\n")
    try:
        test_schemas()
        test_file_vault()
        test_model_runner()
        test_zip_packaging()
        test_triage_engine()
        print("🎉 ALL 5 VERIFICATION SUITES PASSED FLAWLESSLY!")
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
