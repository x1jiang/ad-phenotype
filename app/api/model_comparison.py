"""
API endpoints for multi-model performance comparison
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import time

from app.services.data_loader import DataLoader
from app.services.enhanced_phenotype_model import EnhancedPhenotypeModel
from app.services.llm_phenotype_service import LLMPhenotypeService
from app.config import settings

router = APIRouter()


class ModelComparison:
    """Compare different phenotyping models"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.enhanced_model = EnhancedPhenotypeModel()
        self.llm_service = LLMPhenotypeService()
    
    def run_baseline_model(self) -> Dict[str, Any]:
        """Run baseline phenotyping model"""
        start_time = time.time()
        
        try:
            # Load data
            ad_diag = self.data_loader.load_diagnosis("ad")
            control_diag = self.data_loader.load_diagnosis("control")
            
            # Simple baseline: ICD-10 chapter counts
            def get_baseline_features(diag_df):
                features = []
                for patient_id in diag_df['PatientID'].unique():
                    patient_data = diag_df[diag_df['PatientID'] == patient_id]
                    
                    # Count diagnoses by chapter
                    chapter_counts = patient_data['Level2_Category'].value_counts().to_dict()
                    
                    features.append({
                        'PatientID': patient_id,
                        'total_diagnoses': len(patient_data),
                        'unique_diagnoses': patient_data['ICD10_Code'].nunique(),
                        **chapter_counts
                    })
                
                return pd.DataFrame(features).fillna(0)
            
            ad_features = get_baseline_features(ad_diag)
            control_features = get_baseline_features(control_diag)
            
            # Combine and create labels
            all_features = pd.concat([ad_features, control_features])
            labels = np.array([1] * len(ad_features) + [0] * len(control_features))
            
            # Standardize
            feature_cols = [col for col in all_features.columns if col != 'PatientID']
            X = all_features[feature_cols].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calculate metrics
            if len(np.unique(labels)) > 1 and X_scaled.shape[0] > 2:
                silhouette = silhouette_score(X_scaled, labels)
                davies_bouldin = davies_bouldin_score(X_scaled, labels)
                calinski = calinski_harabasz_score(X_scaled, labels)
            else:
                silhouette = 0
                davies_bouldin = 0
                calinski = 0
            
            execution_time = time.time() - start_time
            
            return {
                'model': 'Baseline',
                'n_features': X.shape[1],
                'n_patients': X.shape[0],
                'silhouette_score': float(silhouette),
                'davies_bouldin_score': float(davies_bouldin),
                'calinski_harabasz_score': float(calinski),
                'execution_time': execution_time,
                'feature_types': ['ICD-10 chapters', 'diagnosis counts']
            }
        except Exception as e:
            return {
                'model': 'Baseline',
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def run_enhanced_model(self) -> Dict[str, Any]:
        """Run enhanced phenotyping model with advanced features"""
        start_time = time.time()
        
        try:
            # Load data
            ad_diag = self.data_loader.load_diagnosis("ad")
            control_diag = self.data_loader.load_diagnosis("control")
            ad_meds = self.data_loader.load_medications("ad")
            control_meds = self.data_loader.load_medications("control")
            ad_labs = self.data_loader.load_labs("ad")
            control_labs = self.data_loader.load_labs("control")
            
            # Extract enhanced features
            ad_features = self.enhanced_model.create_feature_matrix(ad_diag, ad_meds, ad_labs)
            control_features = self.enhanced_model.create_feature_matrix(control_diag, control_meds, control_labs)
            
            # Combine and create labels
            all_features = pd.concat([ad_features, control_features])
            labels = np.array([1] * len(ad_features) + [0] * len(control_features))
            
            # Standardize
            feature_cols = [col for col in all_features.columns if col != 'PatientID']
            X = all_features[feature_cols].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calculate metrics
            if len(np.unique(labels)) > 1 and X_scaled.shape[0] > 2:
                silhouette = silhouette_score(X_scaled, labels)
                davies_bouldin = davies_bouldin_score(X_scaled, labels)
                calinski = calinski_harabasz_score(X_scaled, labels)
            else:
                silhouette = 0
                davies_bouldin = 0
                calinski = 0
            
            execution_time = time.time() - start_time
            
            return {
                'model': 'Enhanced',
                'n_features': X.shape[1],
                'n_patients': X.shape[0],
                'silhouette_score': float(silhouette),
                'davies_bouldin_score': float(davies_bouldin),
                'calinski_harabasz_score': float(calinski),
                'execution_time': execution_time,
                'feature_types': ['temporal', 'network', 'trajectory', 'polypharmacy', 'lab trends']
            }
        except Exception as e:
            return {
                'model': 'Enhanced',
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def run_llm_model(self) -> Dict[str, Any]:
        """Run LLM-enhanced model with GPT-5.1"""
        start_time = time.time()
        
        try:
            # Load data
            ad_diag = self.data_loader.load_diagnosis("ad")
            control_diag = self.data_loader.load_diagnosis("control")
            
            # Extract LLM semantic features (sample for speed)
            ad_sample = ad_diag.groupby('PatientID').head(20)  # Limit for API cost
            control_sample = control_diag.groupby('PatientID').head(20)
            
            ad_llm_features = self.llm_service.extract_semantic_features(ad_sample)
            control_llm_features = self.llm_service.extract_semantic_features(control_sample)
            
            # Also get enhanced features
            ad_meds = self.data_loader.load_medications("ad")
            control_meds = self.data_loader.load_medications("control")
            ad_labs = self.data_loader.load_labs("ad")
            control_labs = self.data_loader.load_labs("control")
            
            ad_enhanced = self.enhanced_model.create_feature_matrix(ad_diag, ad_meds, ad_labs)
            control_enhanced = self.enhanced_model.create_feature_matrix(control_diag, control_meds, control_labs)
            
            # Merge LLM features with enhanced features
            ad_combined = ad_enhanced.merge(ad_llm_features, on='PatientID', how='left')
            control_combined = control_enhanced.merge(control_llm_features, on='PatientID', how='left')
            
            # Combine and create labels
            all_features = pd.concat([ad_combined, control_combined])
            labels = np.array([1] * len(ad_combined) + [0] * len(control_combined))
            
            # Standardize
            feature_cols = [col for col in all_features.columns if col != 'PatientID' and col != 'llm_primary_category' and col != 'llm_themes']
            X = all_features[feature_cols].fillna(0).values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calculate metrics
            if len(np.unique(labels)) > 1 and X_scaled.shape[0] > 2:
                silhouette = silhouette_score(X_scaled, labels)
                davies_bouldin = davies_bouldin_score(X_scaled, labels)
                calinski = calinski_harabasz_score(X_scaled, labels)
            else:
                silhouette = 0
                davies_bouldin = 0
                calinski = 0
            
            execution_time = time.time() - start_time
            
            return {
                'model': 'LLM (GPT-5.1)',
                'n_features': X.shape[1],
                'n_patients': X.shape[0],
                'silhouette_score': float(silhouette),
                'davies_bouldin_score': float(davies_bouldin),
                'calinski_harabasz_score': float(calinski),
                'execution_time': execution_time,
                'feature_types': ['temporal', 'network', 'trajectory', 'polypharmacy', 'lab trends', 'LLM semantic', 'LLM severity', 'LLM complexity'],
                'llm_model': self.llm_service.model
            }
        except Exception as e:
            return {
                'model': 'LLM (GPT-5.1)',
                'error': str(e),
                'execution_time': time.time() - start_time
            }


@router.get("/compare/all")
async def compare_all_models() -> Dict[str, Any]:
    """Compare all three models"""
    try:
        comparator = ModelComparison()
        
        # Run all models
        baseline_results = comparator.run_baseline_model()
        enhanced_results = comparator.run_enhanced_model()
        llm_results = comparator.run_llm_model()
        
        # Calculate improvements
        def calc_improvement(enhanced_val, baseline_val):
            if baseline_val == 0:
                return 0
            return ((enhanced_val - baseline_val) / abs(baseline_val)) * 100
        
        # Silhouette score improvements (higher is better)
        enhanced_improvement = calc_improvement(
            enhanced_results.get('silhouette_score', 0),
            baseline_results.get('silhouette_score', 0)
        )
        
        llm_improvement = calc_improvement(
            llm_results.get('silhouette_score', 0),
            baseline_results.get('silhouette_score', 0)
        )
        
        return {
            'status': 'success',
            'models': {
                'baseline': baseline_results,
                'enhanced': enhanced_results,
                'llm': llm_results
            },
            'improvements': {
                'enhanced_vs_baseline': enhanced_improvement,
                'llm_vs_baseline': llm_improvement,
                'llm_vs_enhanced': calc_improvement(
                    llm_results.get('silhouette_score', 0),
                    enhanced_results.get('silhouette_score', 0)
                )
            },
            'summary': {
                'best_model': max(
                    [baseline_results, enhanced_results, llm_results],
                    key=lambda x: x.get('silhouette_score', 0)
                )['model'],
                'total_execution_time': sum([
                    baseline_results.get('execution_time', 0),
                    enhanced_results.get('execution_time', 0),
                    llm_results.get('execution_time', 0)
                ])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare/{model_type}")
async def compare_single_model(model_type: str) -> Dict[str, Any]:
    """Run a single model comparison"""
    try:
        comparator = ModelComparison()
        
        if model_type == "baseline":
            results = comparator.run_baseline_model()
        elif model_type == "enhanced":
            results = comparator.run_enhanced_model()
        elif model_type == "llm":
            results = comparator.run_llm_model()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
        
        return {
            'status': 'success',
            'results': results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

