"""
Application configuration
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    # Data directory
    data_dir: Path = Path(__file__).parent.parent / "Data"
    
    # Results directory
    results_dir: Path = Path(__file__).parent.parent / "results"
    
    # UMAP parameters
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    umap_random_state: int = 42
    
    # Network parameters
    network_cutoff: float = 0.01  # 1% of patients threshold
    
    # Statistical parameters
    bonferroni_alpha: float = 0.05
    
    # LLM Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-5.1"  # Using GPT-5.1 for enhanced performance
    anthropic_api_key: Optional[str] = None
    use_llm: bool = False
    llm_provider: str = "openai"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

# Create results directory if it doesn't exist
settings.results_dir.mkdir(exist_ok=True)

