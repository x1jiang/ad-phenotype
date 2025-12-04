"""
Unit tests for Enhanced Phenotype Model
"""
import pytest
import pandas as pd
import numpy as np
from app.services.enhanced_phenotype_model import EnhancedPhenotypeModel


@pytest.fixture
def sample_diagnosis_df():
    """Create sample diagnosis data"""
    return pd.DataFrame({
        'PatientID': ['P1', 'P1', 'P1', 'P2', 'P2', 'P3'],
        'DiagnosisCode': ['G30.9', 'I10', 'E11.9', 'F00', 'I10', 'G30.1'],
        'FullDiagnosisName': ['Alzheimer disease', 'Hypertension', 'Diabetes', 
                             'Dementia', 'Hypertension', 'Alzheimer disease'],
        'Level2_Category': ['Mental, Behavioral disorders', 'Circulatory system diseases',
                          'Endocrine, nutritional disorders', 'Mental, Behavioral disorders',
                          'Circulatory system diseases', 'Mental, Behavioral disorders'],
        'DateOfService': ['2023-01-01', '2023-02-01', '2023-03-01', 
                         '2023-01-15', '2023-02-15', '2023-01-20']
    })


@pytest.fixture
def sample_medications_df():
    """Create sample medications data"""
    return pd.DataFrame({
        'PatientID': ['P1', 'P1', 'P2', 'P2', 'P3'],
        'MedicationName': ['Donepezil', 'Metformin', 'Memantine', 'Lisinopril', 'Donepezil'],
        'MedicationGenericName': ['donepezil', 'metformin', 'memantine', 'lisinopril', 'donepezil']
    })


@pytest.fixture
def sample_labs_df():
    """Create sample lab results data"""
    return pd.DataFrame({
        'PatientID': ['P1', 'P1', 'P2', 'P3'],
        'LabTest': ['Hemoglobin A1c', 'TSH', 'Vitamin D', 'Hemoglobin A1c'],
        'TestResult': ['6.5%', '2.5 mIU/L', '30 ng/mL', '5.8%']
    })


@pytest.fixture
def model():
    """Create model instance"""
    return EnhancedPhenotypeModel()


class TestTemporalFeatures:
    """Test temporal feature extraction"""
    
    def test_extract_temporal_features(self, model, sample_diagnosis_df):
        """Test temporal feature extraction"""
        features = model.extract_temporal_features(sample_diagnosis_df)
        
        assert isinstance(features, pd.DataFrame)
        assert 'PatientID' in features.columns
        assert 'time_span_days' in features.columns
        assert 'diagnosis_frequency' in features.columns
        assert 'diagnosis_entropy' in features.columns
        assert len(features) == 3  # 3 unique patients
    
    def test_temporal_features_values(self, model, sample_diagnosis_df):
        """Test temporal feature values are reasonable"""
        features = model.extract_temporal_features(sample_diagnosis_df)
        
        # Check all values are non-negative
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        assert (features[numeric_cols] >= 0).all().all()
        
        # Check patient IDs match
        assert set(features['PatientID']) == set(sample_diagnosis_df['PatientID'].unique())


class TestComorbidityNetworkFeatures:
    """Test comorbidity network feature extraction"""
    
    def test_extract_network_features(self, model, sample_diagnosis_df):
        """Test network feature extraction"""
        features = model.extract_comorbidity_network_features(sample_diagnosis_df)
        
        assert isinstance(features, pd.DataFrame)
        assert 'PatientID' in features.columns
        assert 'avg_degree_centrality' in features.columns
        assert 'avg_betweenness' in features.columns
        assert 'network_density' in features.columns
        assert len(features) == 3
    
    def test_network_features_range(self, model, sample_diagnosis_df):
        """Test network features are in valid range"""
        features = model.extract_comorbidity_network_features(sample_diagnosis_df)
        
        # Centrality measures should be between 0 and 1
        assert (features['avg_degree_centrality'] >= 0).all()
        assert (features['avg_degree_centrality'] <= 1).all()
        assert (features['network_density'] >= 0).all()
        assert (features['network_density'] <= 1).all()


class TestPolypharmacyFeatures:
    """Test polypharmacy feature extraction"""
    
    def test_extract_polypharmacy_features(self, model, sample_medications_df):
        """Test polypharmacy feature extraction"""
        features = model.extract_polypharmacy_features(sample_medications_df)
        
        assert isinstance(features, pd.DataFrame)
        assert 'PatientID' in features.columns
        assert 'unique_medications' in features.columns
        assert 'medication_entropy' in features.columns
        assert 'polypharmacy_score' in features.columns
    
    def test_polypharmacy_counts(self, model, sample_medications_df):
        """Test medication counts are correct"""
        features = model.extract_polypharmacy_features(sample_medications_df)
        
        # P1 has 2 unique medications
        p1_features = features[features['PatientID'] == 'P1']
        assert p1_features['unique_medications'].values[0] == 2


class TestLabTrendFeatures:
    """Test lab trend feature extraction"""
    
    def test_extract_lab_features(self, model, sample_labs_df):
        """Test lab feature extraction"""
        features = model.extract_lab_trend_features(sample_labs_df)
        
        assert isinstance(features, pd.DataFrame)
        assert 'PatientID' in features.columns
        assert 'unique_lab_tests' in features.columns
        assert 'lab_entropy' in features.columns
    
    def test_lab_counts(self, model, sample_labs_df):
        """Test lab test counts"""
        features = model.extract_lab_trend_features(sample_labs_df)
        
        # P1 has 2 lab tests
        p1_features = features[features['PatientID'] == 'P1']
        assert p1_features['unique_lab_tests'].values[0] == 2


class TestDiseaseTrajectoryFeatures:
    """Test disease trajectory feature extraction"""
    
    def test_extract_trajectory_features(self, model, sample_diagnosis_df):
        """Test trajectory feature extraction"""
        features = model.extract_disease_trajectory_features(sample_diagnosis_df)
        
        assert isinstance(features, pd.DataFrame)
        assert 'PatientID' in features.columns
        assert 'chapter_diversity' in features.columns
        assert 'dominant_chapter_ratio' in features.columns
    
    def test_trajectory_values(self, model, sample_diagnosis_df):
        """Test trajectory feature values"""
        features = model.extract_disease_trajectory_features(sample_diagnosis_df)
        
        # Chapter diversity should be positive
        assert (features['chapter_diversity'] > 0).all()
        
        # Dominant chapter ratio should be between 0 and 1
        assert (features['dominant_chapter_ratio'] >= 0).all()
        assert (features['dominant_chapter_ratio'] <= 1).all()


class TestFeatureMatrix:
    """Test complete feature matrix creation"""
    
    def test_create_feature_matrix(self, model, sample_diagnosis_df, 
                                   sample_medications_df, sample_labs_df):
        """Test complete feature matrix creation"""
        features = model.create_feature_matrix(
            sample_diagnosis_df,
            sample_medications_df,
            sample_labs_df
        )
        
        assert isinstance(features, pd.DataFrame)
        assert 'PatientID' in features.columns
        assert len(features) == 3  # 3 unique patients
        
        # Check that features from all sources are present
        assert 'time_span_days' in features.columns  # Temporal
        assert 'avg_degree_centrality' in features.columns  # Network
        assert 'unique_medications' in features.columns  # Polypharmacy
        assert 'unique_lab_tests' in features.columns  # Labs
        assert 'chapter_diversity' in features.columns  # Trajectory
    
    def test_feature_matrix_no_nulls(self, model, sample_diagnosis_df):
        """Test feature matrix has no null values"""
        features = model.create_feature_matrix(sample_diagnosis_df)
        
        # Should have no null values after fillna
        assert features.isnull().sum().sum() == 0


class TestDimensionalityReduction:
    """Test dimensionality reduction"""
    
    def test_apply_dimensionality_reduction(self, model, sample_diagnosis_df):
        """Test PCA dimensionality reduction"""
        features = model.create_feature_matrix(sample_diagnosis_df)
        reduced_df, scaled_features = model.apply_dimensionality_reduction(features, n_components=2)
        
        assert isinstance(reduced_df, pd.DataFrame)
        assert 'PatientID' in reduced_df.columns
        assert 'PC1' in reduced_df.columns
        assert 'PC2' in reduced_df.columns
        assert len(reduced_df) == len(features)
        
        # Check scaled features
        assert isinstance(scaled_features, np.ndarray)
        assert scaled_features.shape[0] == len(features)


class TestClustering:
    """Test phenotype clustering"""
    
    def test_cluster_phenotypes(self, model):
        """Test K-means clustering"""
        # Create sample features
        X = np.random.rand(10, 5)
        
        clusters = model.cluster_phenotypes(X, n_clusters=3)
        
        assert isinstance(clusters, np.ndarray)
        assert len(clusters) == 10
        assert len(np.unique(clusters)) <= 3  # Should have at most 3 clusters


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_dataframe(self, model):
        """Test with empty dataframe"""
        empty_df = pd.DataFrame(columns=['PatientID', 'DiagnosisCode', 'FullDiagnosisName', 
                                        'Level2_Category', 'DateOfService'])
        
        features = model.extract_temporal_features(empty_df)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 0
    
    def test_single_patient(self, model):
        """Test with single patient"""
        single_patient_df = pd.DataFrame({
            'PatientID': ['P1'],
            'DiagnosisCode': ['G30.9'],
            'FullDiagnosisName': ['Alzheimer disease'],
            'Level2_Category': ['Mental, Behavioral disorders']
        })
        
        features = model.create_feature_matrix(single_patient_df)
        assert len(features) == 1
        assert features['PatientID'].values[0] == 'P1'

