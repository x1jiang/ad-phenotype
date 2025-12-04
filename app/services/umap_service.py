"""
UMAP dimensionality reduction service
"""
import pandas as pd
import numpy as np
import umap
from typing import Dict, Tuple, Optional
from app.services.data_loader import DataLoader


class UMAPService:
    """Service for UMAP dimensionality reduction"""
    
    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.data_loader = data_loader or DataLoader()
        self.mapper = None
    
    def create_embedding(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Create UMAP embedding from diagnosis data
        
        Returns:
            Dictionary with embedding coordinates and labels
        """
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn is not installed. Install it with: pip install umap-learn")
        # Load data
        ad_diag = self.data_loader.load_diagnosis("ad")
        con_diag = self.data_loader.load_diagnosis("control")
        
        # Create pivot table
        diag_key = "FullDiagnosisName"
        if diag_key not in ad_diag.columns:
            diag_key = "DiagnosisName"
        
        # Combine and create binary matrix
        all_diag = pd.concat([ad_diag, con_diag], ignore_index=True)
        
        # Create pivot: each row is a patient, each column is a diagnosis
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
        
        # Add labels
        ad_patients = set(ad_diag['PatientID'].unique())
        labels = pivot.index.map(lambda x: 'Alzheimer' if x in ad_patients else 'Control')
        
        # Fit UMAP
        self.mapper = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            low_memory=True
        )
        
        embedding = self.mapper.fit_transform(pivot.values.astype('int32'))
        
        # Add demographic info
        ad_demo = self.data_loader.load_demographics("ad")
        con_demo = self.data_loader.load_demographics("control")
        all_demo = pd.concat([ad_demo, con_demo], ignore_index=True)
        all_demo = all_demo.set_index('PatientID')
        
        result = {
            'embedding': embedding,
            'labels': labels.values,
            'patient_ids': pivot.index.values,
            'demographics': all_demo.loc[pivot.index].to_dict('records')
        }
        
        return result

