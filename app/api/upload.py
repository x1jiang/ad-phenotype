"""
File upload API endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from typing import List, Optional, Tuple
import pandas as pd
from pathlib import Path
import shutil
import os
from app.config import settings
from app.services.data_loader import DataLoader

router = APIRouter()

# Allowed file types
ALLOWED_EXTENSIONS = {'.csv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


def validate_file(file: UploadFile) -> tuple[bool, Optional[str]]:
    """Validate uploaded file"""
    # Check extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Only CSV files are allowed."
    
    # Check file size (if available)
    if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum size is {MAX_FILE_SIZE / 1024 / 1024}MB"
    
    return True, None


def validate_csv_structure(file_path: Path, expected_columns: List[str]) -> tuple[bool, Optional[str], Optional[pd.DataFrame]]:
    """Validate CSV structure and return DataFrame if valid"""
    try:
        df = pd.read_csv(file_path, nrows=5)  # Read first 5 rows for validation
        
        # Check if required columns exist
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}", None
        
        # Check if PatientID exists
        if 'PatientID' not in df.columns:
            return False, "Missing required column: PatientID", None
        
        return True, None, df
    except Exception as e:
        return False, f"Error reading CSV: {str(e)}", None


@router.post("/ad/demographics")
async def upload_ad_demographics(file: UploadFile = File(...)):
    """Upload AD demographics file"""
    valid, error = validate_file(file)
    if not valid:
        raise HTTPException(status_code=400, detail=error)
    
    expected_cols = ['PatientID', 'Sex', 'Age', 'Race', 'Death_Status']
    return await _upload_file(file, "ad", "demographics", expected_cols)


@router.post("/ad/diagnosis")
async def upload_ad_diagnosis(file: UploadFile = File(...)):
    """Upload AD diagnosis file"""
    valid, error = validate_file(file)
    if not valid:
        raise HTTPException(status_code=400, detail=error)
    
    expected_cols = ['PatientID', 'FullDiagnosisName', 'ICD10_Code']
    return await _upload_file(file, "ad", "diagnosis", expected_cols)


@router.post("/ad/medications")
async def upload_ad_medications(file: UploadFile = File(...)):
    """Upload AD medications file"""
    valid, error = validate_file(file)
    if not valid:
        raise HTTPException(status_code=400, detail=error)
    
    expected_cols = ['PatientID', 'MedicationName']
    return await _upload_file(file, "ad", "medications", expected_cols)


@router.post("/ad/labresults")
async def upload_ad_labresults(file: UploadFile = File(...)):
    """Upload AD lab results file"""
    valid, error = validate_file(file)
    if not valid:
        raise HTTPException(status_code=400, detail=error)
    
    expected_cols = ['PatientID', 'TestName', 'Values']
    return await _upload_file(file, "ad", "labresults", expected_cols)


@router.post("/control/demographics")
async def upload_control_demographics(file: UploadFile = File(...)):
    """Upload control demographics file"""
    valid, error = validate_file(file)
    if not valid:
        raise HTTPException(status_code=400, detail=error)
    
    expected_cols = ['PatientID', 'Sex', 'Age', 'Race', 'Death_Status']
    return await _upload_file(file, "control", "demographics", expected_cols)


@router.post("/control/diagnosis")
async def upload_control_diagnosis(file: UploadFile = File(...)):
    """Upload control diagnosis file"""
    valid, error = validate_file(file)
    if not valid:
        raise HTTPException(status_code=400, detail=error)
    
    expected_cols = ['PatientID', 'FullDiagnosisName', 'ICD10_Code']
    return await _upload_file(file, "control", "diagnosis", expected_cols)


@router.post("/control/medications")
async def upload_control_medications(file: UploadFile = File(...)):
    """Upload control medications file"""
    valid, error = validate_file(file)
    if not valid:
        raise HTTPException(status_code=400, detail=error)
    
    expected_cols = ['PatientID', 'MedicationName']
    return await _upload_file(file, "control", "medications", expected_cols)


@router.post("/control/labresults")
async def upload_control_labresults(file: UploadFile = File(...)):
    """Upload control lab results file"""
    valid, error = validate_file(file)
    if not valid:
        raise HTTPException(status_code=400, detail=error)
    
    expected_cols = ['PatientID', 'TestName', 'Values']
    return await _upload_file(file, "control", "labresults", expected_cols)


async def _upload_file(
    file: UploadFile,
    cohort: str,
    file_type: str,
    expected_cols: List[str]
) -> JSONResponse:
    """Internal function to handle file upload"""
    # Create data directory if it doesn't exist
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file temporarily
    temp_path = data_dir / f"temp_{cohort}_{file_type}_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate structure
        valid, error, df = validate_csv_structure(temp_path, expected_cols)
        if not valid:
            temp_path.unlink()
            raise HTTPException(status_code=400, detail=error)
        
        # Get file stats
        file_size = temp_path.stat().st_size
        row_count = len(pd.read_csv(temp_path))
        
        # Move to final location
        final_path = data_dir / f"{cohort}_{file_type}.csv"
        shutil.move(str(temp_path), str(final_path))
        
        return JSONResponse(content={
            "status": "success",
            "message": f"File uploaded successfully",
            "filename": file.filename,
            "cohort": cohort,
            "file_type": file_type,
            "file_size": file_size,
            "row_count": row_count,
            "columns": list(df.columns),
            "sample_data": df.head(3).to_dict('records')
        })
    except HTTPException:
        raise
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/status")
async def get_upload_status(request: Request):
    """Get status of uploaded files"""
    from fastapi.templating import Jinja2Templates
    
    data_dir = Path(settings.data_dir)
    files_status = {}
    
    file_types = ['demographics', 'diagnosis', 'medications', 'labresults']
    cohorts = ['ad', 'control']
    
    for cohort in cohorts:
        files_status[cohort] = {}
        for file_type in file_types:
            file_path = data_dir / f"{cohort}_{file_type}.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, nrows=1)
                    file_size = file_path.stat().st_size
                    # Count rows more efficiently
                    with open(file_path, 'r', encoding='utf-8') as f:
                        row_count = sum(1 for line in f) - 1  # Subtract header
                    files_status[cohort][file_type] = {
                        "exists": True,
                        "filename": file_path.name,
                        "file_size": file_size,
                        "row_count": row_count,
                        "columns": list(df.columns)
                    }
                except Exception as e:
                    files_status[cohort][file_type] = {
                        "exists": True,
                        "error": str(e)
                    }
            else:
                files_status[cohort][file_type] = {
                    "exists": False
                }
    
    # Check if request wants HTML (HTMX)
    accept_header = request.headers.get("accept", "")
    
    if "text/html" in accept_header:
        from fastapi.templating import Jinja2Templates
        templates_dir = Path(__file__).parent.parent.parent / "templates"
        templates = Jinja2Templates(directory=str(templates_dir))
        return templates.TemplateResponse("status_table.html", {
            "request": request,
            "status": files_status
        })
    
    return JSONResponse(content=files_status)


@router.delete("/{cohort}/{file_type}")
async def delete_file(cohort: str, file_type: str):
    """Delete an uploaded file"""
    if cohort not in ['ad', 'control']:
        raise HTTPException(status_code=400, detail="Invalid cohort. Must be 'ad' or 'control'")
    
    if file_type not in ['demographics', 'diagnosis', 'medications', 'labresults']:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = Path(settings.data_dir) / f"{cohort}_{file_type}.csv"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        file_path.unlink()
        return JSONResponse(content={
            "status": "success",
            "message": f"File {cohort}_{file_type}.csv deleted successfully"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@router.post("/preload/{cohort}")
async def preload_sample_data(cohort: str):
    """
    Preload sample data from the Data/ folder for a specific cohort
    This allows quick loading of existing sample data without manual upload
    """
    if cohort not in ['ad', 'control']:
        raise HTTPException(status_code=400, detail="Invalid cohort. Must be 'ad' or 'control'")
    
    # Define the 7 file types
    file_types = ['demographics', 'diagnosis', 'medications', 'labresults', 'imaging', 'treatments', 'vitals']
    
    files_loaded = 0
    total_rows = 0
    errors = []
    
    # Source and destination directories
    source_dir = settings.data_directory  # Data/ folder
    dest_dir = settings.data_directory    # Same directory (data is already there)
    
    # Check if files exist and count rows
    for file_type in file_types:
        source_file = source_dir / f"{cohort}_{file_type}.csv"
        
        if not source_file.exists():
            errors.append(f"{cohort}_{file_type}.csv not found")
            continue
        
        try:
            # Read CSV to validate and count rows
            df = pd.read_csv(source_file)
            row_count = len(df)
            total_rows += row_count
            files_loaded += 1
            
        except Exception as e:
            errors.append(f"Error reading {cohort}_{file_type}.csv: {str(e)}")
    
    if files_loaded == 0:
        raise HTTPException(
            status_code=404, 
            detail=f"No sample data files found for {cohort} cohort in Data/ folder"
        )
    
    return JSONResponse(content={
        "status": "success",
        "cohort": cohort,
        "files_loaded": files_loaded,
        "total_files": len(file_types),
        "total_rows": total_rows,
        "message": f"Successfully preloaded {files_loaded}/{len(file_types)} files",
        "errors": errors if errors else None
    })

