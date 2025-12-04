"""
Unit tests for LLM Phenotype Service
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from app.services.llm_phenotype_service import LLMPhenotypeService
from app.config import settings


@pytest.fixture
def sample_diagnosis_df():
    """Create sample diagnosis data"""
    return pd.DataFrame({
        'PatientID': ['P1', 'P1', 'P2'],
        'DiagnosisCode': ['G30.9', 'I10', 'F00'],
        'FullDiagnosisName': ['Alzheimer disease', 'Hypertension', 'Dementia'],
        'Level2_Category': ['Mental, Behavioral disorders', 'Circulatory system diseases',
                          'Mental, Behavioral disorders']
    })


@pytest.fixture
def llm_service():
    """Create LLM service instance"""
    return LLMPhenotypeService()


class TestLLMServiceInitialization:
    """Test LLM service initialization"""
    
    def test_initialization(self, llm_service):
        """Test service initializes correctly"""
        assert llm_service is not None
        assert hasattr(llm_service, 'model')
        assert llm_service.model == settings.openai_model
    
    def test_client_initialization_without_key(self):
        """Test client is None without API key"""
        with patch('app.config.settings.openai_api_key', None):
            with patch('app.config.settings.use_llm', False):
                service = LLMPhenotypeService()
                assert service.client is None


class TestSemanticFeatureExtraction:
    """Test semantic feature extraction"""
    
    def test_fallback_semantic_features(self, llm_service, sample_diagnosis_df):
        """Test fallback method when LLM not available"""
        features = llm_service._fallback_semantic_features(sample_diagnosis_df)
        
        assert isinstance(features, pd.DataFrame)
        assert 'PatientID' in features.columns
        assert 'llm_primary_category' in features.columns
        assert 'llm_severity_score' in features.columns
        assert 'llm_complexity_score' in features.columns
        assert len(features) == 2  # 2 unique patients
    
    def test_semantic_features_structure(self, llm_service, sample_diagnosis_df):
        """Test semantic features have correct structure"""
        features = llm_service.extract_semantic_features(sample_diagnosis_df)
        
        assert isinstance(features, pd.DataFrame)
        required_columns = ['PatientID', 'llm_primary_category', 'llm_severity_score', 
                          'llm_complexity_score', 'llm_theme_count', 'llm_themes']
        for col in required_columns:
            assert col in features.columns
    
    def test_semantic_features_values(self, llm_service, sample_diagnosis_df):
        """Test semantic feature values are reasonable"""
        features = llm_service.extract_semantic_features(sample_diagnosis_df)
        
        # Severity and complexity should be between 0 and 10
        assert (features['llm_severity_score'] >= 0).all()
        assert (features['llm_severity_score'] <= 10).all()
        assert (features['llm_complexity_score'] >= 0).all()
        assert (features['llm_complexity_score'] <= 10).all()


class TestPhenotypeExplanation:
    """Test phenotype explanation generation"""
    
    def test_fallback_explanation(self, llm_service):
        """Test fallback explanation"""
        explanation = llm_service._fallback_explanation('Alzheimer disease')
        
        assert isinstance(explanation, dict)
        assert 'phenotype' in explanation
        assert 'explanation' in explanation
        assert 'severity' in explanation
        assert 'source' in explanation
        assert explanation['phenotype'] == 'Alzheimer disease'
        assert explanation['source'] == 'Rule-based'
    
    def test_explain_phenotype_structure(self, llm_service):
        """Test phenotype explanation structure"""
        explanation = llm_service.explain_phenotype('Hypertension')
        
        assert isinstance(explanation, dict)
        assert 'phenotype' in explanation
        assert 'explanation' in explanation
        assert 'severity' in explanation
        assert 'source' in explanation
        assert 'model' in explanation
    
    def test_batch_explain_phenotypes(self, llm_service):
        """Test batch phenotype explanation"""
        phenotypes = ['Alzheimer disease', 'Hypertension', 'Diabetes']
        explanations = llm_service.batch_explain_phenotypes(phenotypes)
        
        assert isinstance(explanations, list)
        assert len(explanations) == 3
        for exp in explanations:
            assert isinstance(exp, dict)
            assert 'phenotype' in exp
            assert 'explanation' in exp
    
    def test_batch_explain_limit(self, llm_service):
        """Test batch explanation respects limit"""
        phenotypes = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11']
        explanations = llm_service.batch_explain_phenotypes(phenotypes, limit=5)
        
        assert len(explanations) == 5


class TestCohortComparison:
    """Test cohort comparison functionality"""
    
    def test_compare_cohorts_structure(self, llm_service, sample_diagnosis_df):
        """Test cohort comparison returns correct structure"""
        ad_df = sample_diagnosis_df[sample_diagnosis_df['PatientID'] == 'P1']
        control_df = sample_diagnosis_df[sample_diagnosis_df['PatientID'] == 'P2']
        
        comparison = llm_service.compare_cohorts(ad_df, control_df)
        
        assert isinstance(comparison, dict)
        assert 'comparison' in comparison
        assert 'ad_cohort_size' in comparison or 'comparison' in comparison
    
    def test_compare_cohorts_without_llm(self, llm_service, sample_diagnosis_df):
        """Test cohort comparison without LLM"""
        # Force LLM to be unavailable
        llm_service.client = None
        
        ad_df = sample_diagnosis_df[sample_diagnosis_df['PatientID'] == 'P1']
        control_df = sample_diagnosis_df[sample_diagnosis_df['PatientID'] == 'P2']
        
        comparison = llm_service.compare_cohorts(ad_df, control_df)
        
        assert isinstance(comparison, dict)
        assert 'comparison' in comparison


class TestClinicalSummary:
    """Test clinical summary generation"""
    
    def test_clinical_summary_structure(self, llm_service):
        """Test clinical summary returns string"""
        patient_data = {
            'demographics': pd.DataFrame({'PatientID': ['P1', 'P2']}),
            'diagnoses': pd.DataFrame({'PatientID': ['P1', 'P1', 'P2']}),
            'medications': pd.DataFrame({'PatientID': ['P1', 'P2']}),
            'labs': pd.DataFrame({'PatientID': ['P1']})
        }
        
        summary = llm_service.generate_clinical_summary(patient_data)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_clinical_summary_without_llm(self, llm_service):
        """Test clinical summary without LLM"""
        llm_service.client = None
        
        patient_data = {
            'demographics': pd.DataFrame({'PatientID': ['P1']}),
            'diagnoses': pd.DataFrame({'PatientID': ['P1']}),
            'medications': pd.DataFrame({'PatientID': ['P1']}),
            'labs': pd.DataFrame({'PatientID': ['P1']})
        }
        
        summary = llm_service.generate_clinical_summary(patient_data)
        
        assert isinstance(summary, str)
        assert 'not available' in summary.lower()


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_dataframe(self, llm_service):
        """Test with empty dataframe"""
        empty_df = pd.DataFrame(columns=['PatientID', 'DiagnosisCode', 'FullDiagnosisName', 
                                        'Level2_Category'])
        
        features = llm_service.extract_semantic_features(empty_df)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 0
    
    def test_single_patient(self, llm_service):
        """Test with single patient"""
        single_patient_df = pd.DataFrame({
            'PatientID': ['P1'],
            'DiagnosisCode': ['G30.9'],
            'FullDiagnosisName': ['Alzheimer disease'],
            'Level2_Category': ['Mental, Behavioral disorders']
        })
        
        features = llm_service.extract_semantic_features(single_patient_df)
        assert len(features) == 1
        assert features['PatientID'].values[0] == 'P1'
    
    def test_explain_empty_phenotype(self, llm_service):
        """Test explaining empty phenotype name"""
        explanation = llm_service.explain_phenotype('')
        
        assert isinstance(explanation, dict)
        assert 'phenotype' in explanation

