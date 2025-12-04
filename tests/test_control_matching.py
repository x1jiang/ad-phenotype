"""
Tests for ControlMatching service
"""
import pytest
from app.services.control_matching import ControlMatcher
from app.services.data_loader import DataLoader


class TestControlMatcher:
    """Tests for ControlMatcher service"""
    
    @pytest.fixture
    def control_matcher(self, data_loader):
        """Create ControlMatcher with test data"""
        return ControlMatcher(data_loader=data_loader)
    
    @pytest.mark.skip(reason="Requires more complex test data")
    def test_match_controls(self, control_matcher):
        """Test control matching"""
        matching_vars = ['Race', 'Age', 'Sex', 'Death_Status']
        matched = control_matcher.match_controls(
            matching_vars=matching_vars,
            ratio=2
        )
        
        assert matched is not None
        # Add more assertions based on expected output

