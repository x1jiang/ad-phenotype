"""
Unit tests for Model Comparison API
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestModelComparisonEndpoints:
    """Test model comparison API endpoints"""
    
    def test_compare_all_models_endpoint(self):
        """Test /api/models/compare/all endpoint"""
        response = client.get("/api/models/compare/all")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'status' in data
        assert data['status'] == 'success'
        assert 'models' in data
        assert 'baseline' in data['models']
        assert 'enhanced' in data['models']
        assert 'llm' in data['models']
    
    def test_compare_baseline_model(self):
        """Test /api/models/compare/baseline endpoint"""
        response = client.get("/api/models/compare/baseline")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'status' in data
        assert 'results' in data
        assert data['results']['model'] == 'Baseline'
    
    def test_compare_enhanced_model(self):
        """Test /api/models/compare/enhanced endpoint"""
        response = client.get("/api/models/compare/enhanced")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'status' in data
        assert 'results' in data
        assert data['results']['model'] == 'Enhanced'
    
    def test_compare_llm_model(self):
        """Test /api/models/compare/llm endpoint"""
        response = client.get("/api/models/compare/llm")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'status' in data
        assert 'results' in data
        assert 'LLM' in data['results']['model']
    
    def test_compare_invalid_model(self):
        """Test with invalid model type"""
        response = client.get("/api/models/compare/invalid")
        
        assert response.status_code == 400


class TestModelComparisonResults:
    """Test model comparison result structure"""
    
    def test_baseline_results_structure(self):
        """Test baseline model results have correct structure"""
        response = client.get("/api/models/compare/baseline")
        data = response.json()
        results = data['results']
        
        assert 'model' in results
        assert 'n_features' in results
        assert 'n_patients' in results
        assert 'execution_time' in results
    
    def test_all_models_comparison_structure(self):
        """Test all models comparison has correct structure"""
        response = client.get("/api/models/compare/all")
        data = response.json()
        
        assert 'improvements' in data
        assert 'summary' in data
        assert 'best_model' in data['summary']
        assert 'total_execution_time' in data['summary']
    
    def test_improvements_calculation(self):
        """Test improvements are calculated"""
        response = client.get("/api/models/compare/all")
        data = response.json()
        improvements = data['improvements']
        
        assert 'enhanced_vs_baseline' in improvements
        assert 'llm_vs_baseline' in improvements
        assert 'llm_vs_enhanced' in improvements
        
        # Improvements should be numeric
        assert isinstance(improvements['enhanced_vs_baseline'], (int, float))


class TestModelMetrics:
    """Test model performance metrics"""
    
    def test_metrics_are_numeric(self):
        """Test all metrics are numeric"""
        response = client.get("/api/models/compare/all")
        data = response.json()
        
        for model_name, model_data in data['models'].items():
            if 'silhouette_score' in model_data:
                assert isinstance(model_data['silhouette_score'], (int, float))
            if 'execution_time' in model_data:
                assert isinstance(model_data['execution_time'], (int, float))
                assert model_data['execution_time'] >= 0
    
    def test_feature_counts(self):
        """Test feature counts are reasonable"""
        response = client.get("/api/models/compare/all")
        data = response.json()
        
        baseline_features = data['models']['baseline']['n_features']
        enhanced_features = data['models']['enhanced']['n_features']
        
        # Enhanced should have more features than baseline
        assert enhanced_features >= baseline_features
    
    def test_patient_counts_consistent(self):
        """Test patient counts are consistent across models"""
        response = client.get("/api/models/compare/all")
        data = response.json()
        
        baseline_patients = data['models']['baseline']['n_patients']
        enhanced_patients = data['models']['enhanced']['n_patients']
        llm_patients = data['models']['llm']['n_patients']
        
        # All models should have same number of patients
        assert baseline_patients == enhanced_patients
        assert enhanced_patients == llm_patients


class TestErrorHandling:
    """Test error handling in model comparison"""
    
    def test_invalid_endpoint(self):
        """Test invalid endpoint returns 404"""
        response = client.get("/api/models/invalid")
        assert response.status_code == 404
    
    def test_wrong_http_method(self):
        """Test wrong HTTP method"""
        response = client.post("/api/models/compare/all")
        assert response.status_code == 405  # Method not allowed

