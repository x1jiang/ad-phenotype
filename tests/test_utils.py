"""
Tests for utility functions
"""
import pytest
from app.utils.icd10 import icd10_code_to_chapter, icd_chapter_to_name
from app.utils.statistics import (
    perform_statistical_test,
    test_continuous_variable,
    bonferroni_correction
)
import pandas as pd
import numpy as np


class TestICD10Utils:
    """Tests for ICD-10 utility functions"""
    
    def test_icd10_code_to_chapter(self):
        """Test ICD-10 code to chapter conversion"""
        assert icd10_code_to_chapter('G30.9') == 'G00–G99'
        assert icd10_code_to_chapter('I10') == 'I00–I99'
        assert icd10_code_to_chapter('E11') == 'E00–E90'
        assert icd10_code_to_chapter('A00') == 'A00–B99'
        assert icd10_code_to_chapter('nan') == 'NaN'
        assert icd10_code_to_chapter('') == 'NaN'
    
    def test_icd_chapter_to_name(self):
        """Test ICD-10 chapter to name conversion"""
        assert 'nervous system' in icd_chapter_to_name('G00–G99').lower()
        assert 'circulatory' in icd_chapter_to_name('I00–I99').lower()
        assert 'endocrine' in icd_chapter_to_name('E00–E90').lower()
        assert icd_chapter_to_name('invalid') == ' '


class TestStatistics:
    """Tests for statistical functions"""
    
    def test_perform_statistical_test(self):
        """Test statistical test function"""
        result = perform_statistical_test(
            case_pos=100,
            case_neg=50,
            control_pos=50,
            control_neg=100
        )
        
        assert 'odds_ratio' in result
        assert 'pvalue' in result
        assert 'log2_odds_ratio' in result
        assert 'neg_log10_pvalue' in result
        assert 'test_type' in result
        assert result['odds_ratio'] > 0
        assert result['pvalue'] >= 0 and result['pvalue'] <= 1
    
    def test_perform_statistical_test_small_sample(self):
        """Test with small sample (should use Fisher's exact)"""
        result = perform_statistical_test(
            case_pos=2,
            case_neg=1,
            control_pos=1,
            control_neg=2
        )
        
        assert result['test_type'] == 'fisher'
    
    def test_test_continuous_variable(self):
        """Test continuous variable testing"""
        case_vals = pd.Series([1, 2, 3, 4, 5])
        control_vals = pd.Series([6, 7, 8, 9, 10])
        
        result = test_continuous_variable(case_vals, control_vals)
        
        assert 'statistic' in result
        assert 'pvalue' in result
        assert 'neg_log10_pvalue' in result
        assert result['pvalue'] >= 0 and result['pvalue'] <= 1
    
    def test_bonferroni_correction(self):
        """Test Bonferroni correction"""
        alpha = 0.05
        n_tests = 100
        
        corrected = bonferroni_correction(alpha, n_tests)
        
        assert corrected == 0.05 / 100
        assert corrected < alpha

