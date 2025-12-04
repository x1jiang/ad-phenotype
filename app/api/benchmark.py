"""
Benchmark API endpoints
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.services.benchmark_service import BenchmarkService

router = APIRouter()


@router.post("/run/umap")
async def benchmark_umap(n_runs: int = 3):
    """Run UMAP benchmark"""
    try:
        service = BenchmarkService()
        results = service.benchmark_umap_embeddings(n_runs=n_runs)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/association")
async def benchmark_association(n_runs: int = 3):
    """Run association analysis benchmark"""
    try:
        service = BenchmarkService()
        results = service.benchmark_association_analysis(n_runs=n_runs)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/full")
async def run_full_benchmark():
    """Run complete benchmark suite"""
    try:
        service = BenchmarkService()
        results = service.run_full_benchmark()
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results")
async def get_benchmark_results():
    """Get latest benchmark results"""
    try:
        from pathlib import Path
        import json
        from datetime import datetime
        
        results_dir = Path(__file__).parent.parent.parent.parent / "results" / "benchmarks"
        
        # Find latest results file
        result_files = list(results_dir.glob("umap_benchmark_*.json"))
        if not result_files:
            return JSONResponse(content={"message": "No benchmark results found"})
        
        latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

