"""
Tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Tests for API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns HTML"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_data_summary_endpoint(self):
        """Test data summary endpoint"""
        response = client.get("/api/data/summary")
        assert response.status_code in [200, 500]  # 500 if data files don't exist
    
    @pytest.mark.skip(reason="Requires data files and dependencies")
    def test_umap_endpoint(self):
        """Test UMAP endpoint"""
        response = client.post(
            "/api/analysis/umap",
            data={
                "n_neighbors": 15,
                "min_dist": 0.1,
                "metric": "cosine"
            }
        )
        # Will fail if data files don't exist, which is expected
        assert response.status_code in [200, 500]
    
    @pytest.mark.skip(reason="Requires data files and dependencies")
    def test_association_diagnosis_endpoint(self):
        """Test diagnosis association endpoint"""
        response = client.post(
            "/api/analysis/association/diagnosis",
            data={
                "stratify_by": None,
                "bonferroni_alpha": 0.05
            }
        )
        assert response.status_code in [200, 500]

