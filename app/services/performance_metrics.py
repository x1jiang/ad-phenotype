"""
Performance metrics calculation service
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from app.services.data_loader import DataLoader


class PerformanceMetricsService:
    """Service for calculating performance metrics"""
    
    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.data_loader = data_loader or DataLoader()
    
    def calculate_umap_separation_metrics(
        self,
        embedding: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """
        Calculate separation metrics for UMAP embedding
        
        Returns:
            Dictionary with AUC, F1, precision, recall, and other metrics
        """
        # Convert labels to binary (AD=1, Control=0)
        y_true = (labels == 'Alzheimer').astype(int)
        
        # Use first UMAP dimension as predictor
        y_scores = embedding[:, 0]
        
        # Normalize scores to [0, 1] for ROC
        y_scores_norm = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10)
        
        # Binary predictions using median threshold
        threshold = np.median(y_scores_norm)
        y_pred = (y_scores_norm >= threshold).astype(int)
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
        
        # ROC AUC
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_scores_norm))
        except:
            metrics['roc_auc'] = 0.5
        
        # Precision-Recall AUC
        try:
            metrics['pr_auc'] = float(average_precision_score(y_true, y_scores_norm))
        except:
            metrics['pr_auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = {
            'true_negative': int(cm[0, 0]),
            'false_positive': int(cm[0, 1]),
            'false_negative': int(cm[1, 0]),
            'true_positive': int(cm[1, 1])
        }
        
        # ROC curve data
        try:
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores_norm)
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            }
        except:
            metrics['roc_curve'] = None
        
        # Precision-Recall curve data
        try:
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores_norm)
            metrics['pr_curve'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        except:
            metrics['pr_curve'] = None
        
        # Separation statistics
        ad_embedding = embedding[labels == 'Alzheimer']
        control_embedding = embedding[labels == 'Control']
        
        if len(ad_embedding) > 0 and len(control_embedding) > 0:
            ad_center = np.mean(ad_embedding, axis=0)
            control_center = np.mean(control_embedding, axis=0)
            
            metrics['centroid_distance'] = float(np.linalg.norm(ad_center - control_center))
            metrics['ad_variance'] = float(np.mean(np.var(ad_embedding, axis=0)))
            metrics['control_variance'] = float(np.mean(np.var(control_embedding, axis=0)))
            metrics['separation_ratio'] = float(
                metrics['centroid_distance'] / 
                (np.sqrt(metrics['ad_variance']) + np.sqrt(metrics['control_variance']) + 1e-10)
            )
        
        return metrics
    
    def calculate_association_metrics(
        self,
        association_results: List[Dict]
    ) -> Dict:
        """
        Calculate metrics for association analysis
        
        Returns:
            Dictionary with performance metrics
        """
        if not association_results:
            return {}
        
        results_df = pd.DataFrame(association_results)
        
        # Significant findings
        significant = results_df.get('significant', pd.Series([False] * len(results_df)))
        n_significant = int(significant.sum())
        n_total = len(results_df)
        
        # Enrichment metrics
        enriched = results_df.get('enriched', pd.Series(['Not Significant'] * len(results_df)))
        ad_enriched = int((enriched == 'Alzheimer Enriched').sum())
        control_enriched = int((enriched == 'Control Enriched').sum())
        
        # P-value distribution
        pvalues = results_df.get('pvalue', pd.Series([1.0] * len(results_df)))
        
        metrics = {
            'total_tests': n_total,
            'significant_count': n_significant,
            'significance_rate': float(n_significant / n_total) if n_total > 0 else 0.0,
            'alzheimer_enriched': ad_enriched,
            'control_enriched': control_enriched,
            'pvalue_distribution': {
                'mean': float(pvalues.mean()),
                'median': float(pvalues.median()),
                'min': float(pvalues.min()),
                'max': float(pvalues.max()),
                'q25': float(pvalues.quantile(0.25)),
                'q75': float(pvalues.quantile(0.75))
            }
        }
        
        # Odds ratio distribution for significant findings
        if n_significant > 0:
            sig_results = results_df[significant]
            odds_ratios = sig_results.get('log2_odds_ratio', pd.Series([0] * len(sig_results)))
            metrics['odds_ratio_distribution'] = {
                'mean': float(odds_ratios.mean()),
                'median': float(odds_ratios.median()),
                'min': float(odds_ratios.min()),
                'max': float(odds_ratios.max())
            }
        
        return metrics

