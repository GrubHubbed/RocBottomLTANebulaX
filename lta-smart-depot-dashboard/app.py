"""
app.py - LTA Smart Depot AI Condition Monitoring (CdM) System
FastAPI Application Server for NebulaX Hackathon Problem Statement 3 (PS3)
"""

import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import APP_TITLE, APP_VERSION, HOST, PORT, BASE_DIR, EXCEL_DIR, PREDICTIONS_DIR
from backend.schemas_info import SUBSYSTEM_SCHEMAS
from backend.file_manager import file_manager
from backend.runner_service import runner_service
from backend.triage_engine import triage_engine

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="Singapore LTA Smart Depot Multi-Subsystem Rolling Stock Maintenance Triage Platform"
)

# Enable CORS for flexible integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = BASE_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def root():
    """Serve the futuristic Operations Control Center dashboard."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "LTA Smart Depot OCC Backend Active. Static assets initializing..."}

# ----------------- SCHEMAS & SPECIFICATIONS -----------------

@app.get("/api/schemas")
async def get_subsystem_schemas():
    """Return all official PS3 subsystem specifications, parameter headers, and metrics."""
    return {
        "status": "success",
        "subsystems": SUBSYSTEM_SCHEMAS
    }

# ----------------- EXCEL & CSV FILE VAULT -----------------

@app.get("/api/files")
async def list_files():
    """List all stored Excel and CSV files with metadata."""
    files = file_manager.list_files()
    return {"status": "success", "files": files, "total": len(files)}

@app.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload Excel (.xlsx, .xls) or CSV files into permanent storage."""
    allowed_extensions = [".xlsx", ".xls", ".csv"]
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Expected one of: {allowed_extensions}"
        )
    
    content = await file.read()
    res = file_manager.save_file(file.filename, content)
    return res

@app.get("/api/files/preview")
async def preview_file(filename: str = Query(...), sheet: Optional[str] = Query(None)):
    """Preview Excel or CSV file data grid, sheet names, and column types."""
    res = file_manager.preview_file(filename, sheet_name=sheet, max_rows=25)
    if res.get("status") == "error":
        raise HTTPException(status_code=404, detail=res.get("message"))
    return res

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """Delete a file from the vault."""
    success = file_manager.delete_file(filename)
    if not success:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    return {"status": "success", "message": f"File '{filename}' deleted."}

# ----------------- PYTHON MODEL RUNNER -----------------

@app.get("/api/models")
async def list_models():
    """List all stand-in baseline and user-uploaded .py scripts."""
    models = runner_service.list_available_scripts()
    return {"status": "success", "models": models, "total": len(models)}

@app.post("/api/models/upload")
async def upload_model_script(file: UploadFile = File(...)):
    """Upload a custom .py model or program script."""
    if not file.filename.endswith(".py"):
        raise HTTPException(status_code=400, detail="Only .py Python script files are accepted.")
    
    content = await file.read()
    code_str = content.decode("utf-8", errors="ignore")
    res = runner_service.save_script(file.filename, code_str)
    return res

@app.get("/api/models/code")
async def get_model_code(script: str = Query(...)):
    """Read Python script code for inspection and review."""
    res = runner_service.read_script(script)
    if res.get("status") == "error":
        raise HTTPException(status_code=404, detail=res.get("message"))
    return res

class RunModelRequest(BaseModel):
    input_filenames: List[str]
    subsystem: str
    custom_args: Optional[List[str]] = None

@app.post("/api/models/run")
async def run_model(req: RunModelRequest):
    """Execute a Python model script against input dataset(s)."""
    result = runner_service.execute_script(
        input_filenames=req.input_filenames,
        subsystem=req.subsystem,
        custom_args=req.custom_args
    )
    return result

@app.post("/api/predictions/package")
async def package_predictions():
    """Package all generated predictions CSVs into predictions.zip conforming to PS3 rubric."""
    res = runner_service.package_predictions_zip()
    return res

@app.get("/api/predictions/validate")
async def validate_predictions():
    """Validate all generated predictions CSVs against PS3 submission rules."""
    res = runner_service.validate_all_predictions()
    return res

@app.get("/api/predictions/preview/{subsystem}")
async def get_prediction_preview(subsystem: str):
    """View generated prediction CSV rows on screen with column stats."""
    res = runner_service.get_prediction_preview(subsystem)
    if res.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=res.get("message"))
    return res

@app.get("/api/predictions/download/{filename}")
async def download_prediction(filename: str):
    """Download prediction CSV or predictions.zip."""
    target = PREDICTIONS_DIR / os.path.basename(filename)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Prediction file not found.")
    return FileResponse(target, filename=target.name)

# ----------------- LTA DEPOT PULL-OUT TRIAGE -----------------

@app.get("/api/triage")
async def get_triage_board():
    """Get fleet condition monitoring matrix, composite THI, and pull-out priorities."""
    data = triage_engine.get_depot_fleet_triage()
    return {"status": "success", "data": data}

class WorkOrderRequest(BaseModel):
    train_id: str

@app.post("/api/triage/work-order")
async def create_work_order(req: WorkOrderRequest):
    """Generate official LTA Depot Rolling Stock Maintenance Work Order."""
    wo = triage_engine.generate_work_order(req.train_id)
    return {"status": "success", "work_order": wo}

if __name__ == "__main__":
    import uvicorn
    print(f"[*] Starting {APP_TITLE} on http://{HOST}:{PORT}")
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)
