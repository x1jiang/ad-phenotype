"""
Tests for DataLoader service
"""
import pytest
import pandas as pd
from app.services.data_loader import DataLoader


class TestDataLoader:
    """Tests for DataLoader service"""
    
    def test_load_demographics(self, data_loader):
        """Test loading demographics"""
        ad_demo = data_loader.load_demographics("ad")
        
        assert isinstance(ad_demo, pd.DataFrame)
        assert 'PatientID' in ad_demo.columns
        assert len(ad_demo) > 0
    
    def test_load_diagnosis(self, data_loader):
        """Test loading diagnosis"""
        ad_diag = data_loader.load_diagnosis("ad")
        
        assert isinstance(ad_diag, pd.DataFrame)
        assert 'PatientID' in ad_diag.columns
        assert len(ad_diag) > 0
    
    def test_load_medications(self, data_loader):
        """Test loading medications"""
        ad_meds = data_loader.load_medications("ad")
        
        assert isinstance(ad_meds, pd.DataFrame)
        assert 'PatientID' in ad_meds.columns
        assert len(ad_meds) > 0
    
    def test_load_lab_results(self, data_loader):
        """Test loading lab results"""
        ad_labs = data_loader.load_lab_results("ad")
        
        assert isinstance(ad_labs, pd.DataFrame)
        assert 'PatientID' in ad_labs.columns
        assert len(ad_labs) > 0
    
    def test_get_patient_counts(self, data_loader):
        """Test getting patient counts"""
        counts = data_loader.get_patient_counts("ad")
        
        assert 'total' in counts
        assert 'female' in counts
        assert 'male' in counts
        assert counts['total'] > 0
    
    def test_count_diagnosis(self, data_loader, sample_diagnosis_data):
        """Test counting diagnoses"""
        total_patients = 3
        counts = data_loader.count_diagnosis(
            sample_diagnosis_data,
            total_patients,
            "FullDiagnosisName"
        )
        
        assert isinstance(counts, pd.DataFrame)
        assert 'Count' in counts.columns
        assert 'Count_r' in counts.columns
        assert len(counts) > 0

