"""
Tests for AssociationService
"""
import pytest
from app.services.association_service import AssociationService
from app.services.data_loader import DataLoader


class TestAssociationService:
    """Tests for AssociationService"""
    
    @pytest.fixture
    def association_service(self, data_loader):
        """Create AssociationService with test data"""
        return AssociationService(data_loader=data_loader)
    
    def test_analyze_diagnosis_overall(self, association_service):
        """Test diagnosis analysis without stratification"""
        result = association_service.analyze_diagnosis(
            diag_key="FullDiagnosisName",
            stratify_by=None,
            alpha=0.05
        )
        
        assert 'results' in result
        assert 'summary' in result
        assert isinstance(result['results'], list)
        assert len(result['results']) > 0
    
    def test_analyze_medications(self, association_service):
        """Test medication analysis"""
        result = association_service.analyze_medications(
            stratify_by=None,
            alpha=0.05
        )
        
        assert 'results' in result
        assert 'summary' in result
        assert isinstance(result['results'], list)
    
    def test_analyze_lab_results(self, association_service):
        """Test lab results analysis"""
        result = association_service.analyze_lab_results(
            stratify_by=None,
            alpha=0.05
        )
        
        assert 'results' in result
        assert 'summary' in result
        assert isinstance(result['results'], list)

