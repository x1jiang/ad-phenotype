"""
FastAPI application for AD EHR Phenotype Analysis
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os

from app.api import analysis, visualization, data, model_comparison
from app.config import settings

app = FastAPI(
    title="AD EHR Phenotype Analysis",
    description="Deep phenotyping of Alzheimer's Disease using Electronic Medical Records",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
base_dir = Path(__file__).parent.parent
static_dir = base_dir / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates
templates_dir = base_dir / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Include routers
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(visualization.router, prefix="/api/visualization", tags=["visualization"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(model_comparison.router, prefix="/api/models", tags=["model_comparison"])

# Include benchmark router
from app.api import benchmark
app.include_router(benchmark.router, prefix="/api/benchmark", tags=["benchmark"])

# Include upload router
from app.api import upload
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])

# Include performance router
from app.api import performance
app.include_router(performance.router, prefix="/api/performance", tags=["performance"])


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Data upload page"""
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

