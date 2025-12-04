"""
Performance metrics and visualization API endpoints
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional
import numpy as np

from app.services.performance_metrics import PerformanceMetricsService
from app.services.phenotype_explainer import PhenotypeExplainer
from app.services.association_service import AssociationService

# Lazy imports for optional dependencies
def get_umap_service():
    try:
        from app.services.umap_service import UMAPService
        return UMAPService
    except ImportError:
        return None

def get_enhanced_umap_service():
    try:
        from app.services.enhanced_umap_service import EnhancedUMAPService
        return EnhancedUMAPService
    except ImportError:
        return None

router = APIRouter()


@router.get("/umap/metrics")
async def get_umap_metrics(
    use_enhanced: bool = False,
    n_neighbors: int = 15,
    min_dist: float = 0.1
):
    """Get performance metrics for UMAP embedding"""
    try:
        if use_enhanced:
            EnhancedUMAPService = get_enhanced_umap_service()
            if EnhancedUMAPService is None:
                raise HTTPException(status_code=503, detail="Enhanced UMAP service not available")
            service = EnhancedUMAPService(use_llm=False)
            result = service.create_enhanced_embedding(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                use_semantic_features=True
            )
        else:
            UMAPService = get_umap_service()
            if UMAPService is None:
                raise HTTPException(status_code=503, detail="UMAP service not available. Install umap-learn")
            service = UMAPService()
            result = service.create_embedding(
                n_neighbors=n_neighbors,
                min_dist=min_dist
            )
        
        metrics_service = PerformanceMetricsService()
        metrics = metrics_service.calculate_umap_separation_metrics(
            np.array(result['embedding']),
            np.array(result['labels'])
        )
        
        return JSONResponse(content={
            'status': 'success',
            'metrics': metrics,
            'method': 'enhanced' if use_enhanced else 'original'
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/association/metrics")
async def get_association_metrics():
    """Get performance metrics for association analysis"""
    try:
        service = AssociationService()
        result = service.analyze_diagnosis(
            diag_key="FullDiagnosisName",
            stratify_by=None,
            alpha=0.05
        )
        
        metrics_service = PerformanceMetricsService()
        metrics = metrics_service.calculate_association_metrics(result['results'])
        
        return JSONResponse(content={
            'status': 'success',
            'metrics': metrics
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/phenotype/explain/{phenotype_name}")
async def explain_phenotype(
    phenotype_name: str,
    context: Optional[str] = None
):
    """Get explanation for a specific phenotype"""
    try:
        explainer = PhenotypeExplainer(use_llm=True)
        
        ctx = None
        if context:
            import json
            ctx = json.loads(context)
        
        explanation = explainer.explain_phenotype(phenotype_name, ctx)
        
        return JSONResponse(content={
            'status': 'success',
            'explanation': explanation
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/phenotype/top-explanations")
async def get_top_phenotype_explanations(top_n: int = 10):
    """Get explanations for top significant phenotypes"""
    try:
        service = AssociationService()
        result = service.analyze_diagnosis(
            diag_key="FullDiagnosisName",
            stratify_by=None,
            alpha=0.05
        )
        
        explainer = PhenotypeExplainer(use_llm=True)
        explanations = explainer.get_top_phenotypes_with_explanations(
            result['results'],
            top_n=top_n
        )
        
        return JSONResponse(content={
            'status': 'success',
            'explanations': explanations,
            'total_significant': result['summary'].get('significant_count', 0)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison/original-vs-enhanced")
async def compare_methods():
    """Compare original vs enhanced methods"""
    try:
        UMAPService = get_umap_service()
        EnhancedUMAPService = get_enhanced_umap_service()
        
        if UMAPService is None or EnhancedUMAPService is None:
            raise HTTPException(
                status_code=503,
                detail="UMAP services not available. Install umap-learn"
            )
        
        # Original UMAP
        original_service = UMAPService()
        original_result = original_service.create_embedding()
        
        # Enhanced UMAP
        enhanced_service = EnhancedUMAPService(use_llm=False)
        enhanced_result = enhanced_service.create_enhanced_embedding(use_semantic_features=True)
        
        metrics_service = PerformanceMetricsService()
        
        original_metrics = metrics_service.calculate_umap_separation_metrics(
            np.array(original_result['embedding']),
            np.array(original_result['labels'])
        )
        
        enhanced_metrics = metrics_service.calculate_umap_separation_metrics(
            np.array(enhanced_result['embedding']),
            np.array(enhanced_result['labels'])
        )
        
        # Calculate improvements
        improvements = {}
        for key in ['roc_auc', 'f1_score', 'accuracy', 'separation_ratio']:
            if key in original_metrics and key in enhanced_metrics:
                orig = original_metrics[key]
                enh = enhanced_metrics[key]
                if orig > 0:
                    improvements[key] = {
                        'original': orig,
                        'enhanced': enh,
                        'improvement': ((enh - orig) / orig) * 100,
                        'absolute_change': enh - orig
                    }
        
        return JSONResponse(content={
            'status': 'success',
            'original': original_metrics,
            'enhanced': enhanced_metrics,
            'improvements': improvements
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

