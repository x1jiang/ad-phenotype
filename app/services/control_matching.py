"""
Control matching service using propensity score matching
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from app.services.data_loader import DataLoader


class ControlMatcher:
    """Service for propensity score matching of controls"""
    
    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.data_loader = data_loader or DataLoader()
    
    def match_controls(
        self,
        matching_vars: List[str],
        ratio: int = 2,
        sensitivity: bool = False
    ) -> pd.DataFrame:
        """
        Perform propensity score matching
        
        Args:
            matching_vars: Variables to match on (e.g., ['Race', 'Age', 'Sex', 'Death_Status'])
            ratio: Control to case ratio (default: 2)
            sensitivity: Include encounter count and duration (if available)
        
        Returns:
            DataFrame with matched controls
        """
        # Load AD cohort
        ad_demo = self.data_loader.load_demographics("ad")
        ad_demo['isAD'] = 1
        
        # Load all patients (background cohort)
        # In real implementation, this would come from full database
        # For now, we'll use control demographics as background
        bg_demo = self.data_loader.load_demographics("control")
        bg_demo['isAD'] = 0
        
        # Combine cohorts
        combined = pd.concat([ad_demo, bg_demo], ignore_index=True)
        
        # Prepare matching variables
        matching_data = self._prepare_matching_data(combined, matching_vars, sensitivity)
        
        # Calculate propensity scores
        propensity_scores = self._calculate_propensity_scores(matching_data)
        matching_data['propensity_score'] = propensity_scores
        
        # Perform matching
        matched = self._perform_matching(matching_data, ratio)
        
        return matched
    
    def _prepare_matching_data(
        self,
        df: pd.DataFrame,
        matching_vars: List[str],
        sensitivity: bool
    ) -> pd.DataFrame:
        """Prepare data for matching"""
        data = df[['PatientID', 'isAD'] + matching_vars].copy()
        
        # Handle missing values
        data = data.dropna(subset=matching_vars)
        
        # Encode categorical variables
        for var in matching_vars:
            if data[var].dtype == 'object':
                le = LabelEncoder()
                data[var] = le.fit_transform(data[var].astype(str))
        
        return data
    
    def _calculate_propensity_scores(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate propensity scores using logistic regression"""
        X = data.drop(['PatientID', 'isAD'], axis=1).values
        y = data['isAD'].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit logistic regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_scaled, y)
        
        # Get propensity scores
        propensity_scores = lr.predict_proba(X_scaled)[:, 1]
        
        return propensity_scores
    
    def _perform_matching(
        self,
        data: pd.DataFrame,
        ratio: int
    ) -> pd.DataFrame:
        """Perform nearest neighbor matching"""
        cases = data[data['isAD'] == 1].copy()
        controls = data[data['isAD'] == 0].copy()
        
        matched_controls = []
        used_control_indices = set()
        
        for _, case in cases.iterrows():
            # Find nearest controls
            case_ps = case['propensity_score']
            controls_available = controls[~controls.index.isin(used_control_indices)]
            
            if len(controls_available) == 0:
                break
            
            # Calculate distance
            controls_available['distance'] = abs(controls_available['propensity_score'] - case_ps)
            
            # Select nearest neighbors
            nearest = controls_available.nsmallest(ratio, 'distance')
            
            matched_controls.append(nearest)
            used_control_indices.update(nearest.index)
        
        if matched_controls:
            matched_df = pd.concat([cases] + matched_controls, ignore_index=True)
        else:
            matched_df = cases
        
        return matched_df

