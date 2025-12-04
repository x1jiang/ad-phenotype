"""
Data API endpoints
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from app.services.data_loader import DataLoader

router = APIRouter()


@router.get("/summary", response_class=HTMLResponse)
async def get_data_summary():
    """Get summary of loaded data as HTML"""
    try:
        loader = DataLoader()
        
        ad_counts = loader.get_patient_counts("ad")
        con_counts = loader.get_patient_counts("control")
        
        # Try to load diagnosis to get counts
        try:
            ad_diag = loader.load_diagnosis("ad")
            con_diag = loader.load_diagnosis("control")
            ad_diag_count = ad_diag['PatientID'].nunique() if 'PatientID' in ad_diag.columns else 0
            con_diag_count = con_diag['PatientID'].nunique() if 'PatientID' in con_diag.columns else 0
        except:
            ad_diag_count = 0
            con_diag_count = 0
        
        total_patients = ad_counts['total'] + con_counts['total']
        total_with_diagnosis = ad_diag_count + con_diag_count
        
        # Return HTML for the summary cards
        html = f'''
        <div class="col-md-3">
            <div class="card stat-card shadow-sm">
                <div class="card-body text-center">
                    <h2 class="display-4 text-primary">{ad_counts['total']}</h2>
                    <p class="card-title text-uppercase">AD PATIENTS</p>
                    <p class="text-muted small">{ad_counts['female']} Female, {ad_counts['male']} Male</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card stat-card shadow-sm">
                <div class="card-body text-center">
                    <h2 class="display-4 text-success">{con_counts['total']}</h2>
                    <p class="card-title text-uppercase">CONTROL PATIENTS</p>
                    <p class="text-muted small">{con_counts['female']} Female, {con_counts['male']} Male</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card stat-card shadow-sm">
                <div class="card-body text-center">
                    <h2 class="display-4 text-info">{total_patients}</h2>
                    <p class="card-title text-uppercase">TOTAL PATIENTS</p>
                    <p class="text-muted small">Combined cohort</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card stat-card shadow-sm">
                <div class="card-body text-center">
                    <h2 class="display-4 text-warning">{total_with_diagnosis}</h2>
                    <p class="card-title text-uppercase">WITH DIAGNOSIS</p>
                    <p class="text-muted small">Coded patients</p>
                </div>
            </div>
        </div>
        '''
        
        return html
    except Exception as e:
        return f'''
        <div class="col-md-12">
            <div class="card">
                <div class="card-body text-center">
                    <p class="text-danger">Error loading data: {str(e)}</p>
                    <p class="text-muted">Please upload data files or use the preload button.</p>
                </div>
            </div>
        </div>
        '''

