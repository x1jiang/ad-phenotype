"""
Enhanced Baseline Computational Phenotype Model
Advanced feature engineering with creative approaches
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx


class EnhancedPhenotypeModel:
    """
    Advanced baseline model with creative feature engineering:
    - Temporal pattern analysis
    - Comorbidity network features
    - Disease trajectory modeling
    - Multi-scale feature extraction
    - Polypharmacy patterns
    - Lab result trends
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        
    def extract_temporal_features(self, diagnosis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal patterns from diagnosis data
        """
        temporal_features = []
        
        for patient_id in diagnosis_df['PatientID'].unique():
            patient_data = diagnosis_df[diagnosis_df['PatientID'] == patient_id].copy()
            
            if 'DateOfService' in patient_data.columns:
                patient_data['DateOfService'] = pd.to_datetime(patient_data['DateOfService'], errors='coerce')
                patient_data = patient_data.sort_values('DateOfService')
                
                # Time span of diagnoses
                if len(patient_data) > 1 and patient_data['DateOfService'].notna().sum() > 1:
                    time_span = (patient_data['DateOfService'].max() - 
                                patient_data['DateOfService'].min()).days
                else:
                    time_span = 0
                
                # Diagnosis frequency (diagnoses per month)
                diagnosis_frequency = len(patient_data) / max(time_span / 30, 1)
                
                # Diagnosis diversity over time (Shannon entropy)
                diagnosis_counts = patient_data['ICD10_Code'].value_counts()
                diagnosis_entropy = entropy(diagnosis_counts)
                
                # Early vs late diagnosis diversity
                if len(patient_data) >= 4:
                    midpoint = len(patient_data) // 2
                    early_diagnoses = set(patient_data.iloc[:midpoint]['ICD10_Code'])
                    late_diagnoses = set(patient_data.iloc[midpoint:]['ICD10_Code'])
                    diagnosis_evolution = len(late_diagnoses - early_diagnoses) / max(len(early_diagnoses), 1)
                else:
                    diagnosis_evolution = 0
            else:
                time_span = 0
                diagnosis_frequency = 0
                diagnosis_entropy = 0
                diagnosis_evolution = 0
            
            temporal_features.append({
                'PatientID': patient_id,
                'time_span_days': time_span,
                'diagnosis_frequency': diagnosis_frequency,
                'diagnosis_entropy': diagnosis_entropy,
                'diagnosis_evolution': diagnosis_evolution,
                'total_diagnoses': len(patient_data),
                'unique_diagnoses': patient_data['ICD10_Code'].nunique()
            })
        
        return pd.DataFrame(temporal_features)
    
    def extract_comorbidity_network_features(self, diagnosis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract network-based features from comorbidity patterns
        """
        network_features = []
        
        # Build comorbidity network
        G = nx.Graph()
        
        for patient_id in diagnosis_df['PatientID'].unique():
            patient_diagnoses = diagnosis_df[diagnosis_df['PatientID'] == patient_id]['ICD10_Code'].tolist()
            
            # Add edges between co-occurring diagnoses
            for i in range(len(patient_diagnoses)):
                for j in range(i + 1, len(patient_diagnoses)):
                    if G.has_edge(patient_diagnoses[i], patient_diagnoses[j]):
                        G[patient_diagnoses[i]][patient_diagnoses[j]]['weight'] += 1
                    else:
                        G.add_edge(patient_diagnoses[i], patient_diagnoses[j], weight=1)
        
        # Calculate centrality measures for each patient's diagnoses
        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            clustering_coef = nx.clustering(G)
        except:
            degree_centrality = {}
            betweenness_centrality = {}
            clustering_coef = {}
        
        for patient_id in diagnosis_df['PatientID'].unique():
            patient_diagnoses = diagnosis_df[diagnosis_df['PatientID'] == patient_id]['ICD10_Code'].tolist()
            
            # Average centrality of patient's diagnoses
            avg_degree_centrality = np.mean([degree_centrality.get(d, 0) for d in patient_diagnoses])
            avg_betweenness = np.mean([betweenness_centrality.get(d, 0) for d in patient_diagnoses])
            avg_clustering = np.mean([clustering_coef.get(d, 0) for d in patient_diagnoses])
            
            # Network density for patient's diagnosis subgraph
            patient_subgraph = G.subgraph(patient_diagnoses)
            try:
                density = nx.density(patient_subgraph)
            except:
                density = 0
            
            network_features.append({
                'PatientID': patient_id,
                'avg_degree_centrality': avg_degree_centrality,
                'avg_betweenness': avg_betweenness,
                'avg_clustering': avg_clustering,
                'network_density': density
            })
        
        return pd.DataFrame(network_features)
    
    def extract_polypharmacy_features(self, medications_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract polypharmacy and medication pattern features
        """
        poly_features = []
        
        for patient_id in medications_df['PatientID'].unique():
            patient_meds = medications_df[medications_df['PatientID'] == patient_id]
            
            # Number of unique medications
            unique_meds = patient_meds['MedicationName'].nunique()
            
            # Medication diversity (entropy)
            med_counts = patient_meds['MedicationName'].value_counts()
            med_entropy = entropy(med_counts) if len(med_counts) > 0 else 0
            
            # Total medication orders
            total_orders = len(patient_meds)
            
            # Average orders per medication
            avg_orders_per_med = total_orders / max(unique_meds, 1)
            
            poly_features.append({
                'PatientID': patient_id,
                'unique_medications': unique_meds,
                'medication_entropy': med_entropy,
                'total_medication_orders': total_orders,
                'avg_orders_per_medication': avg_orders_per_med,
                'polypharmacy_score': unique_meds * med_entropy  # Combined score
            })
        
        return pd.DataFrame(poly_features)
    
    def extract_lab_trend_features(self, labs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract lab result trends and patterns
        """
        lab_features = []
        
        for patient_id in labs_df['PatientID'].unique():
            patient_labs = labs_df[labs_df['PatientID'] == patient_id]
            
            # Number of unique lab tests
            unique_tests = patient_labs['TestName'].nunique()
            
            # Total lab orders
            total_labs = len(patient_labs)
            
            # Lab diversity
            lab_counts = patient_labs['TestName'].value_counts()
            lab_entropy = entropy(lab_counts) if len(lab_counts) > 0 else 0
            
            # Abnormal result ratio (if 'AbnormalFlag' column exists)
            if 'AbnormalFlag' in patient_labs.columns:
                abnormal_ratio = (patient_labs['AbnormalFlag'] == 'Abnormal').sum() / max(len(patient_labs), 1)
            else:
                abnormal_ratio = 0
            
            lab_features.append({
                'PatientID': patient_id,
                'unique_lab_tests': unique_tests,
                'total_lab_orders': total_labs,
                'lab_entropy': lab_entropy,
                'abnormal_ratio': abnormal_ratio
            })
        
        return pd.DataFrame(lab_features)
    
    def extract_disease_trajectory_features(self, diagnosis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Model disease progression trajectories
        """
        trajectory_features = []
        
        # ICD-10 chapter groupings
        def get_icd_chapter(code: str) -> str:
            if pd.isna(code) or not isinstance(code, str):
                return 'Unknown'
            code = code.strip().upper()
            if code.startswith('A') or code.startswith('B'):
                return 'Infectious'
            elif code.startswith('C') or code.startswith('D0') or code.startswith('D1') or \
                 code.startswith('D2') or code.startswith('D3') or code.startswith('D4'):
                return 'Neoplasms'
            elif code.startswith('E'):
                return 'Endocrine'
            elif code.startswith('F'):
                return 'Mental'
            elif code.startswith('G'):
                return 'Nervous'
            elif code.startswith('I'):
                return 'Circulatory'
            elif code.startswith('J'):
                return 'Respiratory'
            elif code.startswith('K'):
                return 'Digestive'
            elif code.startswith('M'):
                return 'Musculoskeletal'
            elif code.startswith('N'):
                return 'Genitourinary'
            else:
                return 'Other'
        
        for patient_id in diagnosis_df['PatientID'].unique():
            patient_data = diagnosis_df[diagnosis_df['PatientID'] == patient_id].copy()
            
            # Add chapter information
            patient_data['Chapter'] = patient_data['ICD10_Code'].apply(get_icd_chapter)
            
            # Chapter diversity
            chapter_diversity = patient_data['Chapter'].nunique()
            
            # Most common chapter
            chapter_counts = patient_data['Chapter'].value_counts()
            dominant_chapter_ratio = chapter_counts.iloc[0] / len(patient_data) if len(chapter_counts) > 0 else 0
            
            # Transition patterns (if temporal data available)
            if 'DateOfService' in patient_data.columns:
                patient_data['DateOfService'] = pd.to_datetime(patient_data['DateOfService'], errors='coerce')
                patient_data = patient_data.sort_values('DateOfService')
                
                # Count chapter transitions
                chapters = patient_data['Chapter'].tolist()
                transitions = sum(1 for i in range(len(chapters) - 1) if chapters[i] != chapters[i + 1])
                transition_rate = transitions / max(len(chapters) - 1, 1)
            else:
                transition_rate = 0
            
            trajectory_features.append({
                'PatientID': patient_id,
                'chapter_diversity': chapter_diversity,
                'dominant_chapter_ratio': dominant_chapter_ratio,
                'transition_rate': transition_rate
            })
        
        return pd.DataFrame(trajectory_features)
    
    def create_feature_matrix(
        self,
        diagnosis_df: pd.DataFrame,
        medications_df: Optional[pd.DataFrame] = None,
        labs_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create comprehensive feature matrix combining all advanced features
        """
        # Extract all feature types
        temporal_feats = self.extract_temporal_features(diagnosis_df)
        network_feats = self.extract_comorbidity_network_features(diagnosis_df)
        trajectory_feats = self.extract_disease_trajectory_features(diagnosis_df)
        
        # Merge diagnosis-based features
        feature_matrix = temporal_feats.merge(network_feats, on='PatientID', how='outer')
        feature_matrix = feature_matrix.merge(trajectory_feats, on='PatientID', how='outer')
        
        # Add medication features if available
        if medications_df is not None and len(medications_df) > 0:
            poly_feats = self.extract_polypharmacy_features(medications_df)
            feature_matrix = feature_matrix.merge(poly_feats, on='PatientID', how='outer')
        
        # Add lab features if available
        if labs_df is not None and len(labs_df) > 0:
            lab_feats = self.extract_lab_trend_features(labs_df)
            feature_matrix = feature_matrix.merge(lab_feats, on='PatientID', how='outer')
        
        # Fill missing values
        feature_matrix = feature_matrix.fillna(0)
        
        return feature_matrix
    
    def apply_dimensionality_reduction(
        self,
        feature_matrix: pd.DataFrame,
        n_components: int = 20
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply PCA for dimensionality reduction while preserving variance
        """
        patient_ids = feature_matrix['PatientID'].copy()
        feature_cols = [col for col in feature_matrix.columns if col != 'PatientID']
        X = feature_matrix[feature_cols].values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        self.pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
        X_reduced = self.pca.fit_transform(X_scaled)
        
        # Create reduced feature dataframe
        reduced_df = pd.DataFrame(
            X_reduced,
            columns=[f'PC{i+1}' for i in range(X_reduced.shape[1])]
        )
        reduced_df.insert(0, 'PatientID', patient_ids)
        
        return reduced_df, X_scaled
    
    def cluster_phenotypes(
        self,
        features: np.ndarray,
        n_clusters: int = 5
    ) -> np.ndarray:
        """
        Identify phenotype clusters using K-means
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features)
        return clusters

