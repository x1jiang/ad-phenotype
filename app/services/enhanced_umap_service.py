"""
Enhanced UMAP service with LLM-based semantic features
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from app.services.data_loader import DataLoader
from app.services.llm_phenotype_service import LLMPhenotypeService

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class EnhancedUMAPService:
    """
    Enhanced UMAP service that uses LLM-extracted semantic features
    for better patient representation
    """
    
    def __init__(
        self,
        data_loader: Optional[DataLoader] = None,
        use_llm: bool = False
    ):
        self.data_loader = data_loader or DataLoader()
        self.llm_service = LLMPhenotypeService(
            data_loader=data_loader,
            use_llm=use_llm
        )
        self.mapper = None
    
    def create_enhanced_embedding(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int = 42,
        use_semantic_features: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Create enhanced UMAP embedding with semantic features
        
        Args:
            use_semantic_features: Whether to include LLM-extracted semantic features
        
        Returns:
            Dictionary with embedding coordinates and labels
        """
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn is not installed. Install it with: pip install umap-learn")
        
        # Load and enhance data
        ad_diag = self.llm_service.enhance_phenotype_extraction("ad")
        con_diag = self.llm_service.enhance_phenotype_extraction("control")
        
        # Create feature matrix
        if use_semantic_features:
            feature_matrix = self._create_enhanced_feature_matrix(ad_diag, con_diag)
        else:
            feature_matrix = self._create_basic_feature_matrix(ad_diag, con_diag)
        
        # Get labels
        ad_patients = set(ad_diag['PatientID'].unique())
        labels = np.array([
            'Alzheimer' if pid in ad_patients else 'Control'
            for pid in feature_matrix.index
        ])
        
        # Fit UMAP
        self.mapper = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            low_memory=True
        )
        
        embedding = self.mapper.fit_transform(feature_matrix.values.astype('float32'))
        
        # Add demographic info
        ad_demo = self.data_loader.load_demographics("ad")
        con_demo = self.data_loader.load_demographics("control")
        all_demo = pd.concat([ad_demo, con_demo], ignore_index=True)
        all_demo = all_demo.set_index('PatientID')
        
        result = {
            'embedding': embedding,
            'labels': labels,
            'patient_ids': feature_matrix.index.values,
            'demographics': all_demo.loc[feature_matrix.index].to_dict('records'),
            'feature_names': list(feature_matrix.columns),
            'n_features': len(feature_matrix.columns)
        }
        
        return result
    
    def _create_basic_feature_matrix(
        self,
        ad_diag: pd.DataFrame,
        con_diag: pd.DataFrame
    ) -> pd.DataFrame:
        """Create basic binary feature matrix (original method)"""
        diag_key = "FullDiagnosisName"
        if diag_key not in ad_diag.columns:
            diag_key = "DiagnosisName"
        
        all_diag = pd.concat([ad_diag, con_diag], ignore_index=True)
        
        pivot = pd.pivot_table(
            all_diag[[diag_key, 'PatientID']].drop_duplicates(),
            values=diag_key,
            index='PatientID',
            columns=diag_key,
            aggfunc=lambda x: 1 if len(x) > 0 else 0,
            fill_value=0
        )
        
        # Remove Alzheimer-related diagnoses
        alz_cols = [col for col in pivot.columns if 'alzheimer' in str(col).lower()]
        pivot = pivot.drop(columns=alz_cols)
        
        return pivot
    
    def _create_enhanced_feature_matrix(
        self,
        ad_diag: pd.DataFrame,
        con_diag: pd.DataFrame
    ) -> pd.DataFrame:
        """Create enhanced feature matrix with semantic features"""
        # Start with basic features
        basic_matrix = self._create_basic_feature_matrix(ad_diag, con_diag)
        
        # Add semantic category features
        all_diag = pd.concat([ad_diag, con_diag], ignore_index=True)
        
        if 'SemanticCategory' in all_diag.columns:
            semantic_pivot = pd.pivot_table(
                all_diag[['PatientID', 'SemanticCategory']].drop_duplicates(),
                values='SemanticCategory',
                index='PatientID',
                columns='SemanticCategory',
                aggfunc=lambda x: 1 if len(x) > 0 else 0,
                fill_value=0
            )
            
            # Merge with basic features
            basic_matrix = basic_matrix.merge(
                semantic_pivot,
                left_index=True,
                right_index=True,
                how='outer'
            ).fillna(0)
        
        # Add embedding dimensions if available
        embedding_cols = [col for col in all_diag.columns if col.startswith('embedding_dim_')]
        if embedding_cols:
            # Aggregate embeddings per patient (mean)
            embedding_agg = all_diag.groupby('PatientID')[embedding_cols].mean()
            basic_matrix = basic_matrix.merge(
                embedding_agg,
                left_index=True,
                right_index=True,
                how='outer'
            ).fillna(0)
        
        return basic_matrix

