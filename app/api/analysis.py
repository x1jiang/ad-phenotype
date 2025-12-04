"""
Analysis API endpoints
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import json
import uuid
from pathlib import Path
import numpy as np

from app.models.schemas import (
    UMAPRequest, AssociationRequest, NetworkRequest, AnalysisResponse
)
from app.config import settings

# Lazy imports to avoid import errors if optional dependencies are missing
def get_umap_service():
    try:
        from app.services.umap_service import UMAPService
        return UMAPService
    except ImportError:
        return None

def get_association_service():
    from app.services.association_service import AssociationService
    return AssociationService

def get_network_service():
    from app.services.network_service import NetworkService
    return NetworkService

def get_control_matcher():
    from app.services.control_matching import ControlMatcher
    return ControlMatcher

router = APIRouter()

# In-memory job storage (in production, use Redis or database)
jobs: Dict[str, Dict[str, Any]] = {}


@router.post("/umap")
async def create_umap_embedding(request: Request):
    """Create UMAP embedding"""
    try:
        # Handle both JSON and form data
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            params = UMAPRequest(**body)
        else:
            form = await request.form()
            params = UMAPRequest(
                n_neighbors=int(form.get("n_neighbors", 15)),
                min_dist=float(form.get("min_dist", 0.1)),
                metric=form.get("metric", "cosine"),
                random_state=int(form.get("random_state", 42))
            )
        
        UMAPService = get_umap_service()
        if UMAPService is None:
            raise HTTPException(
                status_code=503,
                detail="UMAP service is not available. Install umap-learn: pip install umap-learn"
            )
        service = UMAPService()
        result = service.create_embedding(
            n_neighbors=params.n_neighbors,
            min_dist=params.min_dist,
            metric=params.metric,
            random_state=params.random_state
        )
        
        # Convert numpy arrays to lists for JSON serialization
        result['embedding'] = result['embedding'].tolist()
        result['labels'] = result['labels'].tolist()
        result['patient_ids'] = result['patient_ids'].tolist()
        
        # Get performance metrics
        try:
            from app.services.performance_metrics import PerformanceMetricsService
            metrics_service = PerformanceMetricsService()
            metrics = metrics_service.calculate_umap_separation_metrics(
                np.array(result['embedding']),
                np.array(result['labels'])
            )
            result['performance_metrics'] = metrics
        except Exception as e:
            print(f"Error calculating metrics: {e}")
        
        # Return JSON for HTMX to parse
        from fastapi.responses import JSONResponse
        return JSONResponse(content={
            "status": "success",
            "message": "UMAP embedding created successfully",
            "data": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/association/diagnosis")
async def analyze_diagnosis(request: Request):
    """Analyze diagnosis associations"""
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            params = AssociationRequest(**body)
        else:
            form = await request.form()
            params = AssociationRequest(
                stratify_by=form.get("stratify_by") or None,
                bonferroni_alpha=float(form.get("bonferroni_alpha", 0.05))
            )
        
        AssociationService = get_association_service()
        service = AssociationService()
        result = service.analyze_diagnosis(
            diag_key="FullDiagnosisName",
            stratify_by=params.stratify_by,
            alpha=params.bonferroni_alpha
        )
        
        from fastapi.responses import JSONResponse
        return JSONResponse(content={
            "status": "success",
            "message": "Diagnosis analysis completed",
            "data": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/association/medications")
async def analyze_medications(request: Request):
    """Analyze medication associations"""
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            params = AssociationRequest(**body)
        else:
            form = await request.form()
            params = AssociationRequest(
                bonferroni_alpha=float(form.get("bonferroni_alpha", 0.05))
            )
        
        AssociationService = get_association_service()
        service = AssociationService()
        result = service.analyze_medications(
            stratify_by=params.stratify_by,
            alpha=params.bonferroni_alpha
        )
        
        from fastapi.responses import JSONResponse
        return JSONResponse(content={
            "status": "success",
            "message": "Medication analysis completed",
            "data": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/association/labs")
async def analyze_labs(request: Request):
    """Analyze lab result associations"""
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            params = AssociationRequest(**body)
        else:
            form = await request.form()
            params = AssociationRequest(
                bonferroni_alpha=float(form.get("bonferroni_alpha", 0.05))
            )
        
        AssociationService = get_association_service()
        service = AssociationService()
        result = service.analyze_lab_results(
            stratify_by=params.stratify_by,
            alpha=params.bonferroni_alpha
        )
        
        from fastapi.responses import JSONResponse
        return JSONResponse(content={
            "status": "success",
            "message": "Lab results analysis completed",
            "data": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/network")
async def build_network(request: Request):
    """Build comorbidity network"""
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            params = NetworkRequest(**body)
        else:
            form = await request.form()
            params = NetworkRequest(
                cutoff=float(form.get("cutoff", 0.01)),
                diagnosis_level=form.get("diagnosis_level", "FullDiagnosisName")
            )
        
        NetworkService = get_network_service()
        service = NetworkService()
        
        # Build networks for both cohorts
        ad_network = service.build_network(
            cohort="ad",
            diag_key=params.diagnosis_level,
            cutoff=params.cutoff
        )
        
        control_network = service.build_network(
            cohort="control",
            diag_key=params.diagnosis_level,
            cutoff=params.cutoff
        )
        
        # Save networks
        job_id = str(uuid.uuid4())
        ad_path = settings.results_dir / f"ad_network_{job_id}.graphml"
        control_path = settings.results_dir / f"control_network_{job_id}.graphml"
        
        import networkx as nx
        nx.write_graphml(ad_network, ad_path)
        nx.write_graphml(control_network, control_path)
        
        # Return network statistics
        result = {
            'ad_network': {
                'nodes': ad_network.number_of_nodes(),
                'edges': ad_network.number_of_edges()
            },
            'control_network': {
                'nodes': control_network.number_of_nodes(),
                'edges': control_network.number_of_edges()
            },
            'ad_network_path': str(ad_path),
            'control_network_path': str(control_path)
        }
        
        from fastapi.responses import JSONResponse
        return JSONResponse(content={
            "status": "success",
            "message": "Network built successfully",
            "data": result,
            "job_id": job_id
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/network/{job_id}")
async def get_network(job_id: str):
    """Get network file"""
    import networkx as nx
    
    ad_path = settings.results_dir / f"ad_network_{job_id}.graphml"
    if not ad_path.exists():
        raise HTTPException(status_code=404, detail="Network not found")
    
    G = nx.read_graphml(ad_path)
    
    # Convert to JSON-serializable format
    nodes = [
        {
            'id': node,
            **data
        }
        for node, data in G.nodes(data=True)
    ]
    
    edges = [
        {
            'source': u,
            'target': v,
            **data
        }
        for u, v, data in G.edges(data=True)
    ]
    
    return {
        'nodes': nodes,
        'edges': edges,
        'stats': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges()
        }
    }

