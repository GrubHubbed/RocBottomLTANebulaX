"""
runner_service.py - Python Script & AI Model Execution Engine
Executes stand-in and user-uploaded .py models via subprocesses, streams telemetry logs,
and validates prediction CSV outputs conforming to NebulaX PS3 standards.
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from backend.config import MODELS_DIR, STANDIN_MODELS_DIR, PREDICTIONS_DIR, EXCEL_DIR

class ModelRunnerService:
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.standin_dir = STANDIN_MODELS_DIR
        self.predictions_dir = PREDICTIONS_DIR
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.standin_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def list_available_scripts(self) -> List[Dict[str, Any]]:
        """List all executable Python models (stand-in baseline + user-uploaded scripts)."""
        scripts = []
        
        # 1. Stand-in built-in scripts
        for script_path in self.standin_dir.glob("*.py"):
            stat = script_path.stat()
            name_lower = script_path.name.lower()
            if "door" in name_lower:
                subsystem = "door"
            elif "acv" in name_lower:
                subsystem = "acv"
            elif "rail" in name_lower:
                subsystem = "rail"
            elif "shm" in name_lower:
                subsystem = "shm"
            else:
                subsystem = "custom"

            scripts.append({
                "name": script_path.name,
                "path": str(script_path),
                "type": "standin_baseline",
                "subsystem": subsystem,
                "size_kb": round(stat.st_size / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })

        # 2. User-uploaded custom scripts
        for script_path in self.models_dir.glob("*.py"):
            stat = script_path.stat()
            scripts.append({
                "name": script_path.name,
                "path": str(script_path),
                "type": "user_uploaded",
                "subsystem": "custom",
                "size_kb": round(stat.st_size / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })

        return scripts

    def save_script(self, filename: str, code_content: str) -> Dict[str, Any]:
        """Save a new or uploaded Python model script."""
        safe_name = os.path.basename(filename)
        if not safe_name.endswith(".py"):
            safe_name += ".py"
            
        target = self.models_dir / safe_name
        with open(target, "w", encoding="utf-8") as f:
            f.write(code_content)

        return {
            "status": "success",
            "name": safe_name,
            "path": str(target),
            "size_kb": round(target.stat().st_size / 1024, 2),
            "message": f"Model script '{safe_name}' saved successfully."
        }

    def read_script(self, script_path_str: str) -> Dict[str, Any]:
        """Read Python script content for code inspection and editing."""
        path = Path(script_path_str)
        if not path.exists():
            # Try searching in standin or models dir
            cand1 = self.standin_dir / path.name
            cand2 = self.models_dir / path.name
            if cand1.exists():
                path = cand1
            elif cand2.exists():
                path = cand2
            else:
                return {"status": "error", "message": f"Script '{script_path_str}' not found."}

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return {
                "status": "success",
                "name": path.name,
                "path": str(path),
                "code": content
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def execute_script(
        self,
        input_filenames: List[str],
        subsystem: str,
        custom_args: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute the native Python model adapter against chosen input file(s) and capture execution telemetry."""
        from backend.adapters import run_subsystem
        
        # Verify inputs exist
        for input_filename in input_filenames:
            input_path = EXCEL_DIR / os.path.basename(input_filename)
            if not input_path.exists():
                return {"status": "error", "message": f"Input dataset '{input_filename}' not found in storage."}

        out_names = {
            "door": "door_predictions.csv",
            "acv": "acv_predictions.csv",
            "rail": "rail_predictions.csv",
            "shm": "shm_predictions.csv"
        }
        out_filename = out_names.get(subsystem.lower(), f"{subsystem}_predictions.csv")
        output_csv_path = self.predictions_dir / out_filename

        start_time = datetime.now()
        
        try:
            adapter_results = run_subsystem(subsystem, input_filenames)
            duration_sec = round((datetime.now() - start_time).total_seconds(), 2)
            
            # Check if output file was generated
            prediction_preview = None
            if output_csv_path.exists():
                try:
                    df = pd.read_csv(output_csv_path)
                    prediction_preview = {
                        "columns": list(df.columns),
                        "row_count": len(df),
                        "sample_rows": df.head(10).to_dict(orient="records")
                    }
                except Exception as ex:
                    prediction_preview = {"error": f"Failed to parse prediction output: {ex}"}

            return {
                "status": "success",
                "returncode": 0,
                "script": f"{subsystem}_adapter",
                "input_files": input_filenames,
                "output_file": out_filename,
                "duration_sec": duration_sec,
                "stdout": "Adapter execution successful.",
                "stderr": "",
                "prediction_preview": prediction_preview,
                "adapter_results": adapter_results,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Adapter execution error: {str(e)}"
            }

    def validate_single_prediction_file(self, filename: str) -> Dict[str, Any]:
        """Validate a single prediction CSV against official PS3 submission rubric."""
        csv_path = self.predictions_dir / filename
        if not csv_path.exists():
            return {"status": "missing", "filename": filename, "valid": False, "error": "File does not exist yet."}

        try:
            df = pd.read_csv(csv_path)
            cols = [c.strip().lower() for c in df.columns]
            fname = filename.lower()

            if "door" in fname:
                expected = ["start_time", "end_time", "prediction"]
                if cols != expected:
                    return {"status": "invalid_schema", "filename": filename, "valid": False, "error": f"Expected columns {expected}, got {list(df.columns)}"}
                # Check valid predictions (0/1 integer or standard labels)
                raw_vals = set(df["prediction"].dropna().astype(str).str.strip().str.lower().unique())
                allowed_door = {"0", "1", "0.0", "1.0", "normal", "abnormal resistance"}
                if not raw_vals.issubset(allowed_door):
                    return {"status": "invalid_values", "filename": filename, "valid": False, "error": f"Door predictions must be 0/1 or Normal/Abnormal resistance. Found: {raw_vals}"}

            elif "acv" in fname:
                expected = ["file_id", "ranked_cars"]
                if cols != expected:
                    return {"status": "invalid_schema", "filename": filename, "valid": False, "error": f"Expected columns {expected}, got {list(df.columns)}"}
                # Check format of ranked_cars (pipe-separated)
                sample = str(df["ranked_cars"].iloc[0]) if len(df) > 0 else ""
                if "|" not in sample:
                    return {"status": "invalid_format", "filename": filename, "valid": False, "error": f"ACV ranked_cars should be pipe-separated e.g. '03|01|05...'. Found: '{sample}'"}

            elif "rail" in fname:
                expected = ["file_id", "prediction"]
                if cols != expected:
                    return {"status": "invalid_schema", "filename": filename, "valid": False, "error": f"Expected columns {expected}, got {list(df.columns)}"}
                valid_vals = set(df["prediction"].dropna().unique())
                allowed = {"Normal", "Side I", "Side II", "normal", "side i", "side ii"}
                if not valid_vals.issubset(allowed):
                    return {"status": "invalid_values", "filename": filename, "valid": False, "error": f"Rail predictions must be Normal, Side I, or Side II. Found: {valid_vals}"}

            elif "shm" in fname:
                expected = ["file_id", "prediction"]
                if cols != expected:
                    return {"status": "invalid_schema", "filename": filename, "valid": False, "error": f"Expected columns {expected}, got {list(df.columns)}"}
                # Ensure numeric
                if not pd.to_numeric(df["prediction"], errors="coerce").notnull().all():
                    return {"status": "invalid_values", "filename": filename, "valid": False, "error": "SHM predictions must be numeric values."}

            return {
                "status": "valid",
                "filename": filename,
                "valid": True,
                "row_count": len(df),
                "columns": list(df.columns),
                "sample": df.head(5).to_dict(orient="records")
            }

        except Exception as e:
            return {"status": "error", "filename": filename, "valid": False, "error": str(e)}

    def validate_all_predictions(self) -> Dict[str, Any]:
        """Validate all prediction CSVs against official PS3 submission specifications."""
        targets = [
            ("door", "door_predictions.csv"),
            ("acv", "acv_predictions.csv"),
            ("rail", "rail_predictions.csv"),
            ("shm", "shm_predictions.csv")
        ]
        results = {}
        all_valid = True
        found_any = False

        for sub, fname in targets:
            report = self.validate_single_prediction_file(fname)
            results[sub] = report
            if report.get("valid"):
                found_any = True
            elif report.get("status") != "missing":
                all_valid = False

        return {
            "status": "success",
            "all_valid": all_valid and found_any,
            "has_predictions": found_any,
            "subsystems": results
        }

    def get_prediction_preview(self, subsystem: str) -> Dict[str, Any]:
        """Get full prediction preview and download metadata for a specific subsystem."""
        out_names = {
            "door": "door_predictions.csv",
            "acv": "acv_predictions.csv",
            "rail": "rail_predictions.csv",
            "shm": "shm_predictions.csv"
        }
        filename = out_names.get(subsystem.lower(), f"{subsystem}_predictions.csv")
        path = self.predictions_dir / filename
        if not path.exists():
            return {"status": "not_found", "message": f"No predictions generated for {subsystem} yet."}

        try:
            df = pd.read_csv(path)
            validation = self.validate_single_prediction_file(filename)
            return {
                "status": "success",
                "subsystem": subsystem,
                "filename": filename,
                "row_count": len(df),
                "columns": list(df.columns),
                "rows": df.head(50).values.tolist(),
                "validation": validation
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def package_predictions_zip(self) -> Dict[str, Any]:
        """Package all available *_predictions.csv files into predictions.zip conforming to PS3 rubric."""
        zip_path = self.predictions_dir / "predictions.zip"
        included_files = []
        validation_reports = {}
        
        # Zip only *_predictions.csv files directly at the top level of the zip (no subfolders)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for csv_file in sorted(self.predictions_dir.glob("*_predictions.csv")):
                zf.write(csv_file, arcname=csv_file.name)
                included_files.append(csv_file.name)
                validation_reports[csv_file.name] = self.validate_single_prediction_file(csv_file.name)

        is_all_valid = all(v.get("valid", False) for v in validation_reports.values()) if validation_reports else False

        return {
            "status": "success",
            "zip_filename": "predictions.zip",
            "path": str(zip_path),
            "files_included": included_files,
            "total_files": len(included_files),
            "size_kb": round(zip_path.stat().st_size / 1024, 2) if zip_path.exists() else 0,
            "validation": {
                "all_valid": is_all_valid,
                "reports": validation_reports,
                "top_level_confirmed": True
            }
        }

# Global singleton
runner_service = ModelRunnerService()
