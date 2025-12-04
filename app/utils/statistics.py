"""
Statistical analysis utilities
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from math import log10, log2
from typing import Dict, Tuple, Optional


def perform_statistical_test(
    case_pos: int,
    case_neg: int,
    control_pos: int,
    control_neg: int
) -> Dict[str, float]:
    """
    Perform Chi-square or Fisher's exact test on 2x2 contingency table
    
    Args:
        case_pos: Cases with condition
        case_neg: Cases without condition
        control_pos: Controls with condition
        control_neg: Controls without condition
    
    Returns:
        Dictionary with test results (odds_ratio, pvalue, test_type)
    """
    contingency = np.array([[case_pos, case_neg], [control_pos, control_neg]])
    
    # Use Fisher's exact if any cell has < 5
    if contingency.min() < 5:
        odds_ratio, pvalue = fisher_exact(contingency)
        test_type = 'fisher'
    else:
        chi2, pvalue, dof, expected = chi2_contingency(contingency)
        # Calculate odds ratio manually
        if control_pos == 0 or case_neg == 0:
            odds_ratio = np.inf if case_pos > 0 and control_neg > 0 else 0
        else:
            odds_ratio = (case_pos * control_neg) / (control_pos * case_neg)
        test_type = 'chi2'
    
    # Handle edge cases
    if odds_ratio == 0:
        odds_ratio = 0.001
    elif odds_ratio == np.inf:
        odds_ratio = 1000
    
    log2_or = log2(odds_ratio) if odds_ratio > 0 else -10
    neg_log10_p = -log10(pvalue) if pvalue > 0 else 10
    
    return {
        'odds_ratio': float(odds_ratio),
        'log2_odds_ratio': float(log2_or),
        'pvalue': float(pvalue),
        'neg_log10_pvalue': float(neg_log10_p),
        'test_type': test_type
    }


def test_continuous_variable(
    case_values: pd.Series,
    control_values: pd.Series
) -> Dict[str, float]:
    """
    Perform Mann-Whitney U test on continuous variables
    
    Args:
        case_values: Series of case values
        control_values: Series of control values
    
    Returns:
        Dictionary with test results
    """
    try:
        statistic, pvalue = mannwhitneyu(case_values, control_values, alternative='two-sided')
        return {
            'statistic': float(statistic),
            'pvalue': float(pvalue),
            'neg_log10_pvalue': float(-log10(pvalue)) if pvalue > 0 else 10
        }
    except Exception:
        return {
            'statistic': 0.0,
            'pvalue': 1.0,
            'neg_log10_pvalue': 0.0
        }


def bonferroni_correction(alpha: float, n_tests: int) -> float:
    """
    Calculate Bonferroni corrected significance threshold
    
    Args:
        alpha: Original significance level (e.g., 0.05)
        n_tests: Number of tests performed
    
    Returns:
        Corrected significance threshold
    """
    return alpha / n_tests

