"""
Tests for file upload functionality
"""
import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory"""
    temp_dir = tempfile.mkdtemp()
    original_dir = settings.data_dir
    settings.data_dir = Path(temp_dir)
    yield temp_dir
    shutil.rmtree(temp_dir)
    settings.data_dir = original_dir


@pytest.fixture
def sample_csv_file():
    """Create a sample CSV file for testing"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df = pd.DataFrame({
        'PatientID': ['P001', 'P002', 'P003'],
        'Sex': ['Male', 'Female', 'Male'],
        'Age': [75, 80, 72],
        'Race': ['White', 'White', 'Asian'],
        'Death_Status': ['Alive', 'Deceased', 'Alive']
    })
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    yield temp_file.name
    Path(temp_file.name).unlink()


@pytest.fixture
def invalid_csv_file():
    """Create an invalid CSV file (missing columns)"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df = pd.DataFrame({
        'PatientID': ['P001', 'P002'],
        'Name': ['John', 'Jane']  # Missing required columns
    })
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    yield temp_file.name
    Path(temp_file.name).unlink()


class TestFileUpload:
    """Tests for file upload endpoints"""
    
    def test_upload_ad_demographics_success(self, temp_data_dir, sample_csv_file):
        """Test successful AD demographics upload"""
        with open(sample_csv_file, 'rb') as f:
            response = client.post(
                "/api/upload/ad/demographics",
                files={"file": ("test.csv", f, "text/csv")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["cohort"] == "ad"
        assert data["file_type"] == "demographics"
        assert "row_count" in data
        assert data["row_count"] == 3
    
    def test_upload_control_demographics_success(self, temp_data_dir, sample_csv_file):
        """Test successful control demographics upload"""
        with open(sample_csv_file, 'rb') as f:
            response = client.post(
                "/api/upload/control/demographics",
                files={"file": ("test.csv", f, "text/csv")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["cohort"] == "control"
    
    def test_upload_invalid_file_type(self, temp_data_dir):
        """Test upload with invalid file type"""
        # Create a non-CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write("test content")
        temp_file.close()
        
        with open(temp_file.name, 'rb') as f:
            response = client.post(
                "/api/upload/ad/demographics",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
        
        Path(temp_file.name).unlink()
    
    def test_upload_missing_columns(self, temp_data_dir, invalid_csv_file):
        """Test upload with missing required columns"""
        with open(invalid_csv_file, 'rb') as f:
            response = client.post(
                "/api/upload/ad/demographics",
                files={"file": ("test.csv", f, "text/csv")}
            )
        
        assert response.status_code == 400
        assert "Missing required columns" in response.json()["detail"]
    
    def test_upload_status_empty(self, temp_data_dir):
        """Test upload status with no files"""
        response = client.get("/api/upload/status")
        assert response.status_code == 200
        data = response.json()
        
        assert "ad" in data
        assert "control" in data
        assert data["ad"]["demographics"]["exists"] == False
    
    def test_upload_status_with_files(self, temp_data_dir, sample_csv_file):
        """Test upload status with uploaded files"""
        # Upload a file first
        with open(sample_csv_file, 'rb') as f:
            client.post(
                "/api/upload/ad/demographics",
                files={"file": ("test.csv", f, "text/csv")}
            )
        
        # Check status
        response = client.get("/api/upload/status")
        assert response.status_code == 200
        data = response.json()
        
        assert data["ad"]["demographics"]["exists"] == True
        assert data["ad"]["demographics"]["row_count"] == 3
    
    def test_delete_file(self, temp_data_dir, sample_csv_file):
        """Test file deletion"""
        # Upload a file first
        with open(sample_csv_file, 'rb') as f:
            client.post(
                "/api/upload/ad/demographics",
                files={"file": ("test.csv", f, "text/csv")}
            )
        
        # Delete it
        response = client.delete("/api/upload/ad/demographics")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        
        # Verify it's deleted
        status_response = client.get("/api/upload/status")
        status_data = status_response.json()
        assert status_data["ad"]["demographics"]["exists"] == False
    
    def test_delete_nonexistent_file(self, temp_data_dir):
        """Test deleting a file that doesn't exist"""
        response = client.delete("/api/upload/ad/demographics")
        assert response.status_code == 404
    
    def test_upload_diagnosis_file(self, temp_data_dir):
        """Test uploading diagnosis file"""
        # Create diagnosis CSV
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df = pd.DataFrame({
            'PatientID': ['P001', 'P002'],
            'FullDiagnosisName': ['Dementia', 'Hypertension'],
            'ICD10_Code': ['G30.9', 'I10']
        })
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        with open(temp_file.name, 'rb') as f:
            response = client.post(
                "/api/upload/ad/diagnosis",
                files={"file": ("diagnosis.csv", f, "text/csv")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["file_type"] == "diagnosis"
        
        Path(temp_file.name).unlink()
    
    def test_upload_medications_file(self, temp_data_dir):
        """Test uploading medications file"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df = pd.DataFrame({
            'PatientID': ['P001'],
            'MedicationName': ['Donepezil']
        })
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        with open(temp_file.name, 'rb') as f:
            response = client.post(
                "/api/upload/ad/medications",
                files={"file": ("medications.csv", f, "text/csv")}
            )
        
        assert response.status_code == 200
        Path(temp_file.name).unlink()
    
    def test_upload_labresults_file(self, temp_data_dir):
        """Test uploading lab results file"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df = pd.DataFrame({
            'PatientID': ['P001'],
            'TestName': ['Glucose'],
            'Values': ['100']
        })
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        with open(temp_file.name, 'rb') as f:
            response = client.post(
                "/api/upload/ad/labresults",
                files={"file": ("labs.csv", f, "text/csv")}
            )
        
        assert response.status_code == 200
        Path(temp_file.name).unlink()


class TestFileValidation:
    """Tests for file validation functions"""
    
    def test_validate_file_csv(self):
        """Test CSV file validation"""
        from app.api.upload import validate_file
        from io import BytesIO
        
        class MockFile:
            filename = "test.csv"
            size = 1000
        
        valid, error = validate_file(MockFile())
        assert valid == True
        assert error is None
    
    def test_validate_file_invalid_type(self):
        """Test invalid file type validation"""
        from app.api.upload import validate_file
        
        class MockFile:
            filename = "test.txt"
            size = 1000
        
        valid, error = validate_file(MockFile())
        assert valid == False
        assert "Invalid file type" in error
    
    def test_validate_csv_structure(self, temp_data_dir):
        """Test CSV structure validation"""
        from app.api.upload import validate_csv_structure
        
        # Create valid CSV
        temp_file = Path(temp_data_dir) / "test.csv"
        df = pd.DataFrame({
            'PatientID': ['P001'],
            'Sex': ['Male'],
            'Age': [75]
        })
        df.to_csv(temp_file, index=False)
        
        valid, error, df_result = validate_csv_structure(
            temp_file,
            ['PatientID', 'Sex', 'Age']
        )
        
        assert valid == True
        assert error is None
        assert df_result is not None
    
    def test_validate_csv_missing_columns(self, temp_data_dir):
        """Test CSV validation with missing columns"""
        from app.api.upload import validate_csv_structure
        
        temp_file = Path(temp_data_dir) / "test.csv"
        df = pd.DataFrame({
            'PatientID': ['P001'],
            'Name': ['John']
        })
        df.to_csv(temp_file, index=False)
        
        valid, error, df_result = validate_csv_structure(
            temp_file,
            ['PatientID', 'Sex', 'Age']
        )
        
        assert valid == False
        assert "Missing required columns" in error

