"""
Data loading and preprocessing services
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from app.config import settings
from app.utils.icd10 import icd10_code_to_chapter


class DataLoader:
    """Service for loading and preprocessing EMR data"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.data_dir
    
    def load_demographics(self, cohort: str = "ad") -> pd.DataFrame:
        """Load demographics data"""
        file_path = self.data_dir / f"{cohort}_demographics.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Demographics file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        # Standardize column names
        df.columns = df.columns.str.strip()
        if 'DeathStatus' in df.columns:
            df.rename(columns={'DeathStatus': 'Death_Status'}, inplace=True)
        return df
    
    def load_diagnosis(self, cohort: str = "ad") -> pd.DataFrame:
        """Load diagnosis data"""
        file_path = self.data_dir / f"{cohort}_diagnosis.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Diagnosis file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # Add ICD-10 chapter mapping
        if 'ICD10_Code' in df.columns:
            df['ValueL'] = df['ICD10_Code'].apply(
                lambda x: icd10_code_to_chapter(str(x)[:3]) if pd.notna(x) else 'NaN'
            )
        elif 'Value' in df.columns:
            df['ValueL'] = df['Value'].apply(
                lambda x: icd10_code_to_chapter(str(x)[:3]) if pd.notna(x) else 'NaN'
            )
        
        return df
    
    def load_medications(self, cohort: str = "ad") -> pd.DataFrame:
        """Load medications data"""
        file_path = self.data_dir / f"{cohort}_medications.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Medications file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        return df
    
    def load_lab_results(self, cohort: str = "ad") -> pd.DataFrame:
        """Load lab results data"""
        file_path = self.data_dir / f"{cohort}_labresults.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Lab results file not found: {file_path}")
        
        df = pd.read_csv(file_path, low_memory=False)
        df.columns = df.columns.str.strip()
        return df
    
    def get_patient_counts(self, cohort: str = "ad") -> Dict[str, int]:
        """Get patient counts by sex"""
        demo = self.load_demographics(cohort)
        counts = {
            'total': len(demo['PatientID'].unique()),
            'female': len(demo[demo['Sex'] == 'Female']['PatientID'].unique()) if 'Sex' in demo.columns else 0,
            'male': len(demo[demo['Sex'] == 'Male']['PatientID'].unique()) if 'Sex' in demo.columns else 0
        }
        return counts
    
    def count_diagnosis(
        self,
        diagnosis_df: pd.DataFrame,
        total_patients: int,
        diag_key: str = "FullDiagnosisName"
    ) -> pd.DataFrame:
        """
        Count patients with each diagnosis
        
        Args:
            diagnosis_df: Diagnosis dataframe
            total_patients: Total number of patients
            diag_key: Column name for diagnosis (FullDiagnosisName, Level3_Category, Level2_Category)
        
        Returns:
            DataFrame with diagnosis counts
        """
        if diag_key not in diagnosis_df.columns:
            raise ValueError(f"Column {diag_key} not found in diagnosis dataframe")
        
        counts = (
            diagnosis_df[['PatientID', diag_key]]
            .drop_duplicates()
            .groupby(diag_key)['PatientID']
            .nunique()
            .reset_index()
        )
        counts.columns = [diag_key, 'Count']
        counts['Count_r'] = total_patients - counts['Count']
        
        return counts
    
    def load_imaging(self, cohort: str = "ad") -> pd.DataFrame:
        """Load imaging procedures data"""
        file_path = self.data_dir / f"{cohort}_imaging.csv"
        if not file_path.exists():
            print(f"Warning: Imaging file not found: {file_path}")
            return pd.DataFrame()
        return pd.read_csv(file_path)
    
    def load_treatments(self, cohort: str = "ad") -> pd.DataFrame:
        """Load treatment procedures data"""
        file_path = self.data_dir / f"{cohort}_treatments.csv"
        if not file_path.exists():
            print(f"Warning: Treatments file not found: {file_path}")
            return pd.DataFrame()
        return pd.read_csv(file_path)
    
    def load_vitals(self, cohort: str = "ad") -> pd.DataFrame:
        """Load vital signs data"""
        file_path = self.data_dir / f"{cohort}_vitals.csv"
        if not file_path.exists():
            print(f"Warning: Vitals file not found: {file_path}")
            return pd.DataFrame()
        return pd.read_csv(file_path)
    
    def load_all_data(self, cohort: str = "ad") -> dict:
        """Load all available data types for a cohort"""
        return {
            'demographics': self.load_demographics(cohort),
            'diagnosis': self.load_diagnosis(cohort),
            'medications': self.load_medications(cohort),
            'labs': self.load_lab_results(cohort),
            'imaging': self.load_imaging(cohort),
            'treatments': self.load_treatments(cohort),
            'vitals': self.load_vitals(cohort)
        }

