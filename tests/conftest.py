"""
Pytest configuration and fixtures
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch

from app.services.data_loader import DataLoader
from app.config import settings


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory with sample files"""
    temp_dir = tempfile.mkdtemp()
    
    # Create sample demographics
    ad_demo = pd.DataFrame({
        'PatientID': ['AD001', 'AD002', 'AD003'],
        'Sex': ['Male', 'Female', 'Male'],
        'Age': [75, 80, 72],
        'Race': ['White', 'White', 'Asian'],
        'Death_Status': ['Alive', 'Deceased', 'Alive']
    })
    ad_demo.to_csv(Path(temp_dir) / 'ad_demographics.csv', index=False)
    
    con_demo = pd.DataFrame({
        'PatientID': ['CON001', 'CON002', 'CON003', 'CON004', 'CON005', 'CON006'],
        'Sex': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'Age': [74, 79, 73, 81, 76, 78],
        'Race': ['White', 'White', 'Asian', 'White', 'White', 'White'],
        'Death_Status': ['Alive', 'Alive', 'Alive', 'Deceased', 'Alive', 'Alive']
    })
    con_demo.to_csv(Path(temp_dir) / 'control_demographics.csv', index=False)
    
    # Create sample diagnosis
    ad_diag = pd.DataFrame({
        'PatientID': ['AD001', 'AD001', 'AD002', 'AD003'],
        'FullDiagnosisName': ['Dementia', 'Hypertension', 'Dementia', 'Diabetes'],
        'ICD10_Code': ['G30.9', 'I10', 'G30.9', 'E11'],
        'Level2_Category': ['Neurological', 'Cardiovascular', 'Neurological', 'Endocrine'],
        'Level3_Category': ['Dementia', 'Hypertension', 'Dementia', 'Diabetes']
    })
    ad_diag.to_csv(Path(temp_dir) / 'ad_diagnosis.csv', index=False)
    
    con_diag = pd.DataFrame({
        'PatientID': ['CON001', 'CON002', 'CON003', 'CON004', 'CON005'],
        'FullDiagnosisName': ['Hypertension', 'Diabetes', 'Hypertension', 'Arthritis', 'Hypertension'],
        'ICD10_Code': ['I10', 'E11', 'I10', 'M25', 'I10'],
        'Level2_Category': ['Cardiovascular', 'Endocrine', 'Cardiovascular', 'Musculoskeletal', 'Cardiovascular'],
        'Level3_Category': ['Hypertension', 'Diabetes', 'Hypertension', 'Arthritis', 'Hypertension']
    })
    con_diag.to_csv(Path(temp_dir) / 'control_diagnosis.csv', index=False)
    
    # Create sample medications
    ad_meds = pd.DataFrame({
        'PatientID': ['AD001', 'AD002'],
        'MedicationName': ['Donepezil', 'Memantine'],
        'MedicationGenericName': ['donepezil', 'memantine']
    })
    ad_meds.to_csv(Path(temp_dir) / 'ad_medications.csv', index=False)
    
    con_meds = pd.DataFrame({
        'PatientID': ['CON001', 'CON002'],
        'MedicationName': ['Lisinopril', 'Metformin'],
        'MedicationGenericName': ['lisinopril', 'metformin']
    })
    con_meds.to_csv(Path(temp_dir) / 'control_medications.csv', index=False)
    
    # Create sample lab results
    ad_labs = pd.DataFrame({
        'PatientID': ['AD001', 'AD002'],
        'TestName': ['Glucose', 'Cholesterol'],
        'Values': ['100', '200']
    })
    ad_labs.to_csv(Path(temp_dir) / 'ad_labresults.csv', index=False)
    
    con_labs = pd.DataFrame({
        'PatientID': ['CON001', 'CON002'],
        'TestName': ['Glucose', 'Cholesterol'],
        'Values': ['95', '180']
    })
    con_labs.to_csv(Path(temp_dir) / 'control_labresults.csv', index=False)
    
    yield Path(temp_dir)
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def data_loader(temp_data_dir):
    """Create DataLoader with temporary data directory"""
    return DataLoader(data_dir=temp_data_dir)


@pytest.fixture
def sample_diagnosis_data():
    """Sample diagnosis data for testing"""
    return pd.DataFrame({
        'PatientID': ['P001', 'P001', 'P002'],
        'FullDiagnosisName': ['Dementia', 'Hypertension', 'Diabetes'],
        'ICD10_Code': ['G30.9', 'I10', 'E11'],
        'Level2_Category': ['Neurological', 'Cardiovascular', 'Endocrine'],
        'Level3_Category': ['Dementia', 'Hypertension', 'Diabetes']
    })

