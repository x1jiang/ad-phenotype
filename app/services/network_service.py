"""
Comorbidity network service
"""
import pandas as pd
import numpy as np
import networkx as nx
import itertools
from typing import Dict, Optional
from tqdm import tqdm
from app.services.data_loader import DataLoader
from app.config import settings


class NetworkService:
    """Service for building comorbidity networks"""
    
    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.data_loader = data_loader or DataLoader()
    
    def build_network(
        self,
        cohort: str = "ad",
        diag_key: str = "FullDiagnosisName",
        cutoff: float = 0.01
    ) -> nx.Graph:
        """
        Build comorbidity network
        
        Args:
            cohort: 'ad' or 'control'
            diag_key: Diagnosis level
            cutoff: Minimum proportion of patients for edge inclusion
        
        Returns:
            NetworkX graph
        """
        # Load data
        diag = self.data_loader.load_diagnosis(cohort)
        demo = self.data_loader.load_demographics(cohort)
        counts = self.data_loader.get_patient_counts(cohort)
        
        # Get diagnosis counts
        diag_counts = self.data_loader.count_diagnosis(diag, counts['total'], diag_key)
        nodes = diag_counts[diag_key].tolist()
        
        # Create graph
        G = nx.Graph()
        G.add_nodes_from(nodes)
        
        # Add node attributes
        node_attrs = self._get_node_attributes(diag, nodes, diag_key, counts)
        nx.set_node_attributes(G, node_attrs)
        
        # Create edges
        edges = self._create_edges(diag, nodes, diag_key)
        
        # Add edge attributes
        edge_attrs = self._get_edge_attributes(edges, counts, cutoff)
        nx.set_edge_attributes(G, edge_attrs)
        
        # Remove edges below cutoff
        edges_to_remove = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get('PtCount', 0) < counts['total'] * cutoff
        ]
        G.remove_edges_from(edges_to_remove)
        
        return G
    
    def _get_node_attributes(
        self,
        diag: pd.DataFrame,
        nodes: list,
        diag_key: str,
        counts: Dict
    ) -> Dict:
        """Get node attributes"""
        # Count patients per diagnosis
        diag_patients = (
            diag[['PatientID', diag_key, 'Sex']]
            .drop_duplicates()
            .groupby([diag_key, 'Sex'])['PatientID']
            .nunique()
            .reset_index()
        )
        
        attrs = {}
        for node in nodes:
            node_data = diag_patients[diag_patients[diag_key] == node]
            
            female_count = node_data[node_data['Sex'] == 'Female']['PatientID'].sum() if len(node_data) > 0 else 0
            male_count = node_data[node_data['Sex'] == 'Male']['PatientID'].sum() if len(node_data) > 0 else 0
            
            total_count = diag[diag[diag_key] == node]['PatientID'].nunique()
            
            attrs[node] = {
                'PtCount': int(total_count),
                'Females': int(female_count),
                'Males': int(male_count),
                'pFemale': float(female_count * 100 / counts['female']) if counts['female'] > 0 else 0,
                'pMale': float(male_count * 100 / counts['male']) if counts['male'] > 0 else 0
            }
            
            # Add ICD-10 chapter if available
            if 'ValueL' in diag.columns:
                chapter = diag[diag[diag_key] == node]['ValueL'].iloc[0] if len(diag[diag[diag_key] == node]) > 0 else None
                if chapter:
                    attrs[node]['ValueL'] = chapter
        
        return attrs
    
    def _create_edges(
        self,
        diag: pd.DataFrame,
        nodes: list,
        diag_key: str
    ) -> pd.DataFrame:
        """Create edges from patient diagnoses"""
        diag_filtered = diag[diag[diag_key].isin(nodes)]
        
        edges_list = []
        for patient_id, patient_diags in tqdm(diag_filtered.groupby('PatientID')):
            diag_list = patient_diags[diag_key].unique().tolist()
            if len(diag_list) > 1:
                # Create all pairs
                for combo in itertools.combinations(sorted(diag_list), 2):
                    edges_list.append({
                        'PatientID': patient_id,
                        'Combo': combo,
                        'Sex': patient_diags['Sex'].iloc[0] if 'Sex' in patient_diags.columns else None
                    })
        
        return pd.DataFrame(edges_list)
    
    def _get_edge_attributes(
        self,
        edges: pd.DataFrame,
        counts: Dict,
        cutoff: float
    ) -> Dict:
        """Get edge attributes"""
        if len(edges) == 0:
            return {}
        
        # Count patients per edge
        edge_counts = (
            edges.groupby('Combo')
            .agg({
                'PatientID': 'nunique',
                'Sex': lambda x: dict(x.value_counts())
            })
            .reset_index()
        )
        edge_counts.columns = ['Combo', 'PtCount', 'Sex']
        
        attrs = {}
        for _, row in edge_counts.iterrows():
            combo = row['Combo']
            sex_dict = row['Sex'] if isinstance(row['Sex'], dict) else {}
            
            attrs[combo] = {
                'PtCount': int(row['PtCount']),
                'Females': int(sex_dict.get('Female', 0)),
                'Males': int(sex_dict.get('Male', 0)),
                'pFemale': float(sex_dict.get('Female', 0) * 100 / counts['female']) if counts['female'] > 0 else 0,
                'pMale': float(sex_dict.get('Male', 0) * 100 / counts['male']) if counts['male'] > 0 else 0
            }
        
        return attrs

