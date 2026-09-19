import sys
from pathlib import Path
from backend.config import BASE_DIR, PREDICTIONS_DIR, EXCEL_DIR
import json
import pandas as pd

PS3_DIR = BASE_DIR / "NebulaX-Hackathon-ProblemStatement" / "PS3"

# Add subsystem directories to sys.path so we can import them natively
sys.path.append(str(PS3_DIR))
sys.path.append(str(PS3_DIR / "acv-depot-console"))
sys.path.append(str(PS3_DIR / "shm"))
sys.path.append(str(PS3_DIR / "doors"))
sys.path.append(str(PS3_DIR / "src"))

# Now import the subsystems
import door.predict as door_predict
import acv_core
import src.rail_model as rail_model
from src.rail_model import build_features
import joblib
import predictors.shm as shm_predictor

RAIL_MODEL_PATH = PS3_DIR / "src" / "outputs" / "final" / "rail_model.joblib"
rail_model_instance = None
if RAIL_MODEL_PATH.exists():
    rail_model_instance = joblib.load(RAIL_MODEL_PATH)

DOOR_MODEL_PATH = PS3_DIR / "doors" / "models" / "door_model.joblib"

class DoorAdapter:
    @staticmethod
    def run(input_filenames: list[str]):
        # Door typically expects one continuous file
        # We will just process the first file if multiple are selected,
        # or process them in a loop if needed. Let's process just the first for simplicity
        # since door usually expects a single continuous file.
        input_filename = input_filenames[0]
        input_path = EXCEL_DIR / input_filename
        output_path = PREDICTIONS_DIR / "door_predictions.csv"
        
        result = door_predict.predict_file(
            input_csv=input_path,
            output_csv=output_path,
            model_path=DOOR_MODEL_PATH,
            write_detailed=True,
            history_csv=None
        )
        
        return {
            "summary": result.summary,
            "events": result.events.to_dict(orient="records"),
            "segments": result.segments.to_dict(orient="records"),
            "trend": result.trend
        }

class ACVAdapter:
    @staticmethod
    def run(input_filenames: list[str]):
        results = []
        for input_filename in input_filenames:
            input_path = EXCEL_DIR / input_filename
            res = acv_core.analyze_workbook(str(input_path), file_id=input_filename)
            results.append(res)
        
        # Write submission CSV
        output_path = PREDICTIONS_DIR / "acv_predictions.csv"
        csv_content = acv_core.submission_csv(results)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(csv_content)
            
        # For the UI, we return a list of results
        ui_results = []
        for res in results:
            ui_results.append({
                "file_id": res["file_id"] if "file_id" in res else res.get("file", "unknown"),
                "ranked_cars": res["ranked_cars"],
                "ranking_basis": res["ranking_basis"],
                "basis_reason": res["basis_reason"],
                "alerts": res.get("alerts", []),
                "cars": res.get("cars", []),
                "ok": res.get("ok", True),
                "error": res.get("error", "")
            })
            
        return {
            "batch_results": ui_results
        }

class RailAdapter:
    @staticmethod
    def run(input_filenames: list[str]):
        global rail_model_instance
        if rail_model_instance is None:
            rail_model_instance = joblib.load(RAIL_MODEL_PATH)
            
        input_paths = [EXCEL_DIR / fname for fname in input_filenames]
        
        feats = build_features(input_paths, with_wl=(rail_model_instance.config == "REF+WL"), verbose=False)
        preds = rail_model_instance.predict(feats)
        
        out_df = pd.DataFrame({"file_id": input_filenames, "prediction": preds})
        output_path = PREDICTIONS_DIR / "rail_predictions.csv"
        out_df.to_csv(output_path, index=False)
        
        ui_results = out_df.to_dict(orient="records")
        
        counts = out_df["prediction"].value_counts().to_dict()
        
        return {
            "batch_results": ui_results,
            "counts": counts
        }

class SHMAdapter:
    @staticmethod
    def run(input_filenames: list[str]):
        class FileWrapper:
            def __init__(self, path, name):
                self.path = path
                self.name = name
                self.f = open(path, "rb")
            def __iter__(self):
                return iter(self.f)
            def read(self, *args, **kwargs):
                return self.f.read(*args, **kwargs)
            def readline(self, *args, **kwargs):
                return self.f.readline(*args, **kwargs)
            def close(self):
                self.f.close()
                
        fws = [FileWrapper(EXCEL_DIR / fname, fname) for fname in input_filenames]
        
        try:
            result_df = shm_predictor.predict(fws)
            
            output_path = PREDICTIONS_DIR / "shm_predictions.csv"
            sub_df = result_df[list(shm_predictor.OUTPUT_COLUMNS)].copy()
            sub_df.to_csv(output_path, index=False)
            
            interpretations = []
            for _, row in result_df.iterrows():
                interpretations.append(shm_predictor.interpret(row))
                
            ui_results = []
            for i, (_, row) in enumerate(result_df.iterrows()):
                d = row.to_dict()
                d["interpretation"] = interpretations[i]
                ui_results.append(d)
                
            return {
                "batch_results": ui_results
            }
        finally:
            for fw in fws:
                fw.close()

def run_subsystem(subsystem: str, input_filenames: list[str]):
    subsystem = subsystem.lower()
    if subsystem == "door":
        return DoorAdapter.run(input_filenames)
    elif subsystem == "acv":
        return ACVAdapter.run(input_filenames)
    elif subsystem == "rail":
        return RailAdapter.run(input_filenames)
    elif subsystem == "shm":
        return SHMAdapter.run(input_filenames)
    else:
        raise ValueError(f"Unknown subsystem: {subsystem}")
