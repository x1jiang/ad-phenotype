"""
Tests for phenotype explainer service
"""
import pytest
from app.services.phenotype_explainer import PhenotypeExplainer


class TestPhenotypeExplainer:
    """Tests for PhenotypeExplainer service"""
    
    @pytest.fixture
    def explainer(self):
        return PhenotypeExplainer(use_llm=False)
    
    def test_explain_phenotype_cognitive(self, explainer):
        """Test explanation for cognitive phenotype"""
        explanation = explainer.explain_phenotype("Alzheimer's Disease")
        
        assert 'description' in explanation
        assert 'clinical_significance' in explanation
        assert 'ad_association' in explanation
        assert 'implications' in explanation
        assert 'severity' in explanation
        assert explanation['severity'] == 'Severe'
    
    def test_explain_phenotype_cardiovascular(self, explainer):
        """Test explanation for cardiovascular phenotype"""
        explanation = explainer.explain_phenotype("Hypertension")
        
        assert 'description' in explanation
        assert 'cardiovascular' in explanation['description'].lower() or 'cardiac' in explanation['description'].lower()
        assert explanation['severity'] in ['Moderate', 'Moderate to Severe']
    
    def test_explain_phenotype_metabolic(self, explainer):
        """Test explanation for metabolic phenotype"""
        explanation = explainer.explain_phenotype("Type 2 Diabetes")
        
        assert 'description' in explanation
        assert 'metabolic' in explanation['description'].lower()
        assert explanation['severity'] == 'Moderate'
    
    def test_explain_phenotype_with_context(self, explainer):
        """Test explanation with context"""
        context = {
            'ad_prevalence': 15.5,
            'control_prevalence': 5.2,
            'odds_ratio': 3.2,
            'pvalue': 0.001
        }
        
        explanation = explainer.explain_phenotype("Dementia", context)
        
        assert 'description' in explanation
        assert 'clinical_significance' in explanation
    
    def test_explain_phenotype_group(self, explainer):
        """Test explaining a group of phenotypes"""
        phenotypes = ["Alzheimer's Disease", "Hypertension", "Diabetes"]
        
        group_explanation = explainer.explain_phenotype_group(phenotypes, "Test Group")
        
        assert 'group_name' in group_explanation
        assert 'phenotypes' in group_explanation
        assert len(group_explanation['phenotypes']) == 3
        assert 'primary_categories' in group_explanation

