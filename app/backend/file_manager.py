"""
file_manager.py - Excel & CSV File Storage and Inspection Engine
Handles permanent file collection, metadata indexing, sheet parsing, and data grid extraction.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import openpyxl
import pandas as pd

from backend.config import EXCEL_DIR, STORAGE_DIR

class FileManager:
    def __init__(self, storage_dir: Path = EXCEL_DIR):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.seed_sample_datasets()

    def list_files(self) -> List[Dict[str, Any]]:
        """List all stored Excel and CSV files with rich operational metadata."""
        files = []
        for file_path in self.storage_dir.glob("*.*"):
            if file_path.suffix.lower() not in [".xlsx", ".xls", ".csv"]:
                continue
            
            stat = file_path.stat()
            size_kb = round(stat.st_size / 1024, 2)
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            # Infer subsystem association from filename
            name_lower = file_path.name.lower()
            if "door" in name_lower:
                subsystem = "door"
            elif "acv" in name_lower:
                subsystem = "acv"
            elif "rail" in name_lower or "corrugation" in name_lower:
                subsystem = "rail"
            elif "shm" in name_lower or "stress" in name_lower:
                subsystem = "shm"
            else:
                subsystem = "general"

            sheet_names = []
            if file_path.suffix.lower() in [".xlsx", ".xls"]:
                try:
                    wb = openpyxl.load_workbook(file_path, read_only=True)
                    sheet_names = wb.sheetnames
                    wb.close()
                except Exception:
                    sheet_names = ["Sheet1"]

            files.append({
                "filename": file_path.name,
                "path": str(file_path),
                "extension": file_path.suffix.lower(),
                "size_kb": size_kb,
                "modified": modified,
                "subsystem": subsystem,
                "sheets": sheet_names,
                "sheet_count": len(sheet_names) if sheet_names else 1
            })

        # Sort newest first
        files.sort(key=lambda x: x["modified"], reverse=True)
        return files

    def save_file(self, filename: str, content: bytes) -> Dict[str, Any]:
        """Save uploaded file into persistent storage."""
        safe_name = os.path.basename(filename)
        dest_path = self.storage_dir / safe_name
        with open(dest_path, "wb") as f:
            f.write(content)
        
        stat = dest_path.stat()
        return {
            "status": "success",
            "filename": safe_name,
            "path": str(dest_path),
            "size_kb": round(stat.st_size / 1024, 2),
            "message": f"File '{safe_name}' safely stored in depot vault."
        }

    def delete_file(self, filename: str) -> bool:
        """Delete a file from the vault."""
        target = self.storage_dir / os.path.basename(filename)
        if target.exists():
            target.unlink()
            return True
        return False

    def preview_file(self, filename: str, sheet_name: Optional[str] = None, max_rows: int = 15) -> Dict[str, Any]:
        """Preview sheet data, headers, column types, and statistics."""
        target = self.storage_dir / os.path.basename(filename)
        if not target.exists():
            return {"status": "error", "message": f"File '{filename}' not found."}

        ext = target.suffix.lower()
        sheets = []
        
        try:
            if ext in [".xlsx", ".xls"]:
                wb = openpyxl.load_workbook(target, read_only=True)
                sheets = wb.sheetnames
                wb.close()
                
                selected_sheet = sheet_name if sheet_name in sheets else sheets[0]
                df = pd.read_excel(target, sheet_name=selected_sheet, nrows=max_rows)
                total_cols = len(df.columns)
                
                # Get quick row count approximation
                wb_full = openpyxl.load_workbook(target, read_only=True)
                ws = wb_full[selected_sheet]
                total_rows = ws.max_row or len(df)
                wb_full.close()
            else:
                sheets = ["Default"]
                selected_sheet = "Default"
                df = pd.read_csv(target, nrows=max_rows)
                total_cols = len(df.columns)
                # Count total lines
                with open(target, "r", encoding="utf-8", errors="ignore") as f:
                    total_rows = sum(1 for _ in f) - 1

            # Sanitize dataframe for JSON output
            df_filled = df.fillna("")
            headers = [str(col) for col in df.columns]
            rows = df_filled.values.tolist()

            column_summary = []
            for col in df.columns:
                series = df[col]
                column_summary.append({
                    "name": str(col),
                    "dtype": str(series.dtype),
                    "null_count": int(series.isna().sum())
                })

            return {
                "status": "success",
                "filename": filename,
                "sheets": sheets,
                "current_sheet": selected_sheet,
                "headers": headers,
                "rows": rows,
                "preview_row_count": len(rows),
                "total_rows": total_rows,
                "total_cols": total_cols,
                "column_summary": column_summary
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to preview file: {str(e)}"
            }

    def seed_sample_datasets(self):
        """Seed representative sample datasets for each subsystem if storage is empty."""
        # 1. ACV Sample Dataset (Excel)
        acv_sample_path = self.storage_dir / "acv_fleet_case_01.xlsx"
        if not acv_sample_path.exists():
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "ACV_Telemetry"
            
            # Write headers conforming to PS3 specifications
            headers = ["Car model", "Train number", "Time"]
            for car_idx in range(1, 9):
                car_str = f"Car {car_idx:02d}"
                headers.extend([
                    f"{car_str} - Outside Temperature Sensor Reading",
                    f"{car_str} - ACV Running Mode",
                    f"{car_str} - Indoor Average Temperature",
                    f"{car_str} - Load Halved",
                    f"{car_str} - ACV Setting Mode",
                    f"{car_str} - ACV Information Valid"
                ])
            ws.append(headers)

            # Generate 20 realistic telemetry rows with Car 03 showing abnormal thermal divergence (refrigerant leak signature)
            base_time = datetime(2026, 9, 18, 14, 0, 0)
            for step in range(20):
                row = ["C751B", "315", (base_time + pd.Timedelta(seconds=30 * step)).strftime("%Y-%m-%d %H:%M:%S")]
                ambient_temp = 32.4 + (step * 0.05)
                for car_idx in range(1, 9):
                    if car_idx == 3: # Car 03 has refrigerant leak
                        indoor_temp = round(26.8 + (step * 0.18), 2) # Warming up
                        running_mode = 2 # Struggling at full cool
                    else:
                        indoor_temp = round(22.1 + (car_idx % 3) * 0.3, 2) # Nominal
                        running_mode = 1
                    
                    row.extend([
                        round(ambient_temp, 2),
                        running_mode,
                        indoor_temp,
                        0,
                        1,
                        1
                    ])
                ws.append(row)
            wb.save(acv_sample_path)

        # 2. Door Sample Dataset (CSV)
        door_sample_path = self.storage_dir / "door_telemetry_stream.csv"
        if not door_sample_path.exists():
            door_headers = [
                "Datetime", "Car Type", "Car Number", "Door Number",
                "Motor current (mA)", "Motor Voltage (10mV)", "Motor back electromotive force",
                "Door opening time (0.1 s)", "Door closing time (0.1 s)", "Close command",
                "Open command", "DCSR", "DCSL", "DLSR", "DLSL", "Door Opened", "Door Locked",
                "Door is opening", "Door is closing", "Door leaf position"
            ]
            door_rows = [door_headers]
            for i in range(15):
                # Simulated closing cycle with friction resistance on door 4R
                is_abnormal = (i >= 8 and i <= 12)
                curr = 2450.0 if is_abnormal else 1150.0 + (i * 25.0)
                pos = max(0.0, 850.0 - (i * 70.0))
                door_rows.append([
                    f"2026-09-18-14-05-{i:02d}-000", "M", "03", "4R",
                    str(curr), "2400.0", "120.0", "0.0", f"{i * 0.1:.1f}",
                    "1", "0", "0" if pos > 50 else "1", "0" if pos > 50 else "1",
                    "0" if pos > 0 else "1", "0" if pos > 0 else "1",
                    "0", "1" if pos == 0 else "0", "0", "1", f"{pos:.1f}"
                ])
            with open(door_sample_path, "w", encoding="utf-8") as f:
                for r in door_rows:
                    f.write(",".join(r) + "\n")

        # 3. Rail Corrugation Sample Dataset (CSV)
        rail_sample_path = self.storage_dir / "rail_axlebox_vibration_sample.csv"
        if not rail_sample_path.exists():
            rail_headers = ["rotational_speed"] + [f"Car{(i//8)+1}_Pos{(i%8)+1}_vib" for i in range(64)] + [f"Car{(i//8)+1}_Pos{(i%8)+1}_shock" for i in range(64)]
            with open(rail_sample_path, "w", encoding="utf-8") as f:
                f.write(",".join(rail_headers) + "\n")
                # 10 sample time slices
                for t in range(10):
                    speed_pulse = "1" if t % 2 == 0 else "0"
                    # Side I positions (1, 3, 5, 7) with elevated corrugation harmonic amplitude
                    vib_vals = [f"{1.85 if (i%8)%2==0 else 0.42:.2f}" for i in range(64)]
                    shock_vals = [f"{0.25:.2f}" for _ in range(64)]
                    f.write(",".join([speed_pulse] + vib_vals + shock_vals) + "\n")

        # 4. SHM Dynamic Stress Sample Dataset (CSV)
        shm_sample_path = self.storage_dir / "shm_dynamic_stress_segment.csv"
        if not shm_sample_path.exists():
            shm_headers = ["timestamp", "stress_bogie_weld_1_mpa", "stress_bogie_weld_2_mpa", "stress_carbody_bolster_mpa"]
            with open(shm_sample_path, "w", encoding="utf-8") as f:
                f.write(",".join(shm_headers) + "\n")
                for s in range(25):
                    s1 = 45.2 + (s % 7) * 4.3
                    s2 = 38.1 + (s % 5) * 3.1
                    s3 = 52.4 + (s % 9) * 5.8
                    f.write(f"2026-09-18T14:10:{s:02d},{s1:.2f},{s2:.2f},{s3:.2f}\n")

# Global singleton
file_manager = FileManager()
