"""
Tests for performance metrics service
"""
import pytest
import numpy as np
import pandas as pd
from app.services.performance_metrics import PerformanceMetricsService


class TestPerformanceMetrics:
    """Tests for PerformanceMetricsService"""
    
    @pytest.fixture
    def metrics_service(self):
        return PerformanceMetricsService()
    
    def test_calculate_umap_separation_metrics(self, metrics_service):
        """Test UMAP separation metrics calculation"""
        # Create sample embedding with good separation
        n_ad = 100
        n_control = 100
        
        ad_embedding = np.random.randn(n_ad, 2) + np.array([2, 2])
        control_embedding = np.random.randn(n_control, 2) + np.array([-2, -2])
        
        embedding = np.vstack([ad_embedding, control_embedding])
        labels = np.array(['Alzheimer'] * n_ad + ['Control'] * n_control)
        
        metrics = metrics_service.calculate_umap_separation_metrics(embedding, labels)
        
        assert 'roc_auc' in metrics
        assert 'f1_score' in metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'separation_ratio' in metrics
        assert 'confusion_matrix' in metrics
        
        # With good separation, metrics should be reasonable
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_calculate_association_metrics(self, metrics_service):
        """Test association metrics calculation"""
        association_results = [
            {
                'FullDiagnosisName': 'Dementia',
                'pvalue': 0.001,
                'significant': True,
                'enriched': 'Alzheimer Enriched',
                'log2_odds_ratio': 2.5,
                'odds_ratio': 5.66
            },
            {
                'FullDiagnosisName': 'Hypertension',
                'pvalue': 0.05,
                'significant': True,
                'enriched': 'Control Enriched',
                'log2_odds_ratio': -1.2,
                'odds_ratio': 0.44
            },
            {
                'FullDiagnosisName': 'Diabetes',
                'pvalue': 0.1,
                'significant': False,
                'enriched': 'Not Significant',
                'log2_odds_ratio': 0.5,
                'odds_ratio': 1.41
            }
        ]
        
        metrics = metrics_service.calculate_association_metrics(association_results)
        
        assert 'total_tests' in metrics
        assert 'significant_count' in metrics
        assert 'significance_rate' in metrics
        assert metrics['total_tests'] == 3
        assert metrics['significant_count'] == 2
        assert metrics['significance_rate'] == 2/3

