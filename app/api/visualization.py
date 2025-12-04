"""
Visualization API endpoints
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import io
import base64
from typing import Dict, Any

# Lazy imports
def get_umap_service():
    try:
        from app.services.umap_service import UMAPService
        return UMAPService
    except ImportError:
        return None

def get_association_service():
    from app.services.association_service import AssociationService
    return AssociationService

router = APIRouter()


@router.get("/umap/plot")
async def plot_umap(
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine"
):
    """Generate UMAP plot"""
    try:
        UMAPService = get_umap_service()
        if UMAPService is None:
            raise HTTPException(
                status_code=503,
                detail="UMAP service is not available. Install umap-learn: pip install umap-learn"
            )
        service = UMAPService()
        result = service.create_embedding(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric
        )
        
        embedding = np.array(result['embedding'])
        labels = np.array(result['labels'])
        
        # Create plot
        plt.figure(figsize=(10, 8))
        for label in np.unique(labels):
            mask = labels == label
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                label=label,
                alpha=0.6,
                s=5
            )
        
        plt.legend()
        plt.title('UMAP Embedding: AD vs Control')
        plt.axis('off')
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return Response(content=buf.read(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/volcano/{analysis_type}")
async def plot_volcano(
    analysis_type: str,
    stratify_by: str = None,
    alpha: float = 0.05
):
    """Generate volcano plot"""
    try:
        AssociationService = get_association_service()
        service = AssociationService()
        
        if analysis_type == "diagnosis":
            result = service.analyze_diagnosis(
                stratify_by=stratify_by,
                alpha=alpha
            )
            df = pd.DataFrame(result['results'])
            x_col = 'log2_odds_ratio'
            y_col = 'neg_log10_pvalue'
            label_col = 'FullDiagnosisName' if 'FullDiagnosisName' in df.columns else 'DiagnosisName'
        elif analysis_type == "medications":
            result = service.analyze_medications(alpha=alpha)
            df = pd.DataFrame(result['results'])
            x_col = 'log2_odds_ratio'
            y_col = 'neg_log10_pvalue'
            label_col = 'MedicationGenericName'
        else:
            raise HTTPException(status_code=400, detail="Invalid analysis type")
        
        # Create volcano plot
        plt.figure(figsize=(8, 6))
        
        # Color by significance
        colors = df['enriched'].map({
            'Alzheimer Enriched': 'red',
            'Control Enriched': 'blue',
            'Not Significant': 'gray'
        })
        
        plt.scatter(
            df[x_col],
            df[y_col],
            c=colors,
            alpha=0.6,
            s=20
        )
        
        # Add significance line
        if 'summary' in result:
            corrected_alpha = result['summary'].get('corrected_alpha', alpha)
            plt.axhline(-np.log10(corrected_alpha), linestyle='--', color='black', linewidth=1)
        
        plt.axvline(1, color='gray', linestyle=':')
        plt.axvline(-1, color='gray', linestyle=':')
        plt.xlabel(r'$\log_2$(Odds Ratio)')
        plt.ylabel(r'$-\log_{10}$(p-value)')
        plt.title(f'Volcano Plot: {analysis_type.capitalize()}')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return Response(content=buf.read(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_summary():
    """Get analysis summary statistics"""
    try:
        from app.services.data_loader import DataLoader
        
        loader = DataLoader()
        ad_counts = loader.get_patient_counts("ad")
        con_counts = loader.get_patient_counts("control")
        
        return {
            'ad_patients': ad_counts,
            'control_patients': con_counts,
            'total_patients': ad_counts['total'] + con_counts['total']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

