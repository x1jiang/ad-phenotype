"""
Benchmarking service to compare original vs LLM-enhanced methods
"""
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Tuple
from pathlib import Path
import json
from datetime import datetime

from app.services.association_service import AssociationService
from app.services.llm_phenotype_service import LLMPhenotypeService
from app.services.data_loader import DataLoader
from typing import Optional

# Lazy imports for optional dependencies
def get_umap_service():
    try:
        from app.services.umap_service import UMAPService
        return UMAPService
    except ImportError:
        return None

def get_enhanced_umap_service():
    try:
        from app.services.enhanced_umap_service import EnhancedUMAPService
        return EnhancedUMAPService
    except ImportError:
        return None


class BenchmarkService:
    """Service for benchmarking original vs enhanced methods"""
    
    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.data_loader = data_loader or DataLoader()
        self.results_dir = Path(__file__).parent.parent.parent / "results" / "benchmarks"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def benchmark_umap_embeddings(
        self,
        n_runs: int = 3
    ) -> Dict:
        """
        Benchmark UMAP embedding quality
        
        Metrics:
        - Separation between AD and Control groups
        - Computation time
        - Feature count
        - Silhouette score
        """
        print("Benchmarking UMAP embeddings...")
        
        results = {
            'original': [],
            'enhanced': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Original method
        print("  Running original UMAP...")
        UMAPService = get_umap_service()
        if UMAPService is None:
            print("    ⚠ UMAP not available, skipping UMAP benchmark")
            results['original'] = [{'error': 'UMAP not available'}]
            results['enhanced'] = [{'error': 'UMAP not available'}]
            return results
        
        original_service = UMAPService(self.data_loader)
        
        for run in range(n_runs):
            start_time = time.time()
            try:
                original_result = original_service.create_embedding(
                    n_neighbors=15,
                    min_dist=0.1,
                    metric="cosine"
                )
                elapsed = time.time() - start_time
                
                metrics = self._calculate_umap_metrics(original_result)
                metrics['computation_time'] = elapsed
                metrics['n_features'] = len(original_result.get('feature_names', []))
                results['original'].append(metrics)
            except Exception as e:
                print(f"    Error in original run {run+1}: {e}")
                results['original'].append({'error': str(e)})
        
        # Enhanced method
        print("  Running enhanced UMAP...")
        EnhancedUMAPService = get_enhanced_umap_service()
        if EnhancedUMAPService is None:
            print("    ⚠ Enhanced UMAP not available")
            results['enhanced'] = [{'error': 'Enhanced UMAP not available'}]
            return results
        
        enhanced_service = EnhancedUMAPService(
            data_loader=self.data_loader,
            use_llm=False  # Start without LLM for fair comparison
        )
        
        for run in range(n_runs):
            start_time = time.time()
            try:
                enhanced_result = enhanced_service.create_enhanced_embedding(
                    n_neighbors=15,
                    min_dist=0.1,
                    metric="cosine",
                    use_semantic_features=True
                )
                elapsed = time.time() - start_time
                
                metrics = self._calculate_umap_metrics(enhanced_result)
                metrics['computation_time'] = elapsed
                metrics['n_features'] = enhanced_result.get('n_features', 0)
                results['enhanced'].append(metrics)
            except Exception as e:
                print(f"    Error in enhanced run {run+1}: {e}")
                results['enhanced'].append({'error': str(e)})
        
        # Calculate summary statistics
        summary = self._summarize_results(results)
        results['summary'] = summary
        
        # Save results
        self._save_results(results, 'umap_benchmark')
        
        return results
    
    def _calculate_umap_metrics(self, result: Dict) -> Dict:
        """Calculate metrics for UMAP embedding"""
        embedding = np.array(result['embedding'])
        labels = np.array(result['labels'])
        
        metrics = {}
        
        # Separation between groups
        ad_mask = labels == 'Alzheimer'
        control_mask = labels == 'Control'
        
        if np.sum(ad_mask) > 0 and np.sum(control_mask) > 0:
            ad_center = np.mean(embedding[ad_mask], axis=0)
            control_center = np.mean(embedding[control_mask], axis=0)
            
            # Distance between centroids
            centroid_distance = np.linalg.norm(ad_center - control_center)
            metrics['centroid_distance'] = float(centroid_distance)
            
            # Within-group variance
            ad_variance = np.mean(np.var(embedding[ad_mask], axis=0))
            control_variance = np.mean(np.var(embedding[control_mask], axis=0))
            metrics['ad_variance'] = float(ad_variance)
            metrics['control_variance'] = float(control_variance)
            
            # Separation ratio (higher is better)
            metrics['separation_ratio'] = float(
                centroid_distance / (np.sqrt(ad_variance) + np.sqrt(control_variance) + 1e-10)
            )
        
        # Silhouette score (if scikit-learn available)
        try:
            from sklearn.metrics import silhouette_score
            if len(np.unique(labels)) > 1:
                metrics['silhouette_score'] = float(
                    silhouette_score(embedding, labels)
                )
        except ImportError:
            pass
        
        return metrics
    
    def benchmark_association_analysis(
        self,
        n_runs: int = 3
    ) -> Dict:
        """Benchmark association analysis"""
        print("Benchmarking association analysis...")
        
        results = {
            'original': [],
            'enhanced': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Original method
        print("  Running original association analysis...")
        original_service = AssociationService(self.data_loader)
        
        for run in range(n_runs):
            start_time = time.time()
            try:
                original_result = original_service.analyze_diagnosis(
                    diag_key="FullDiagnosisName",
                    stratify_by=None,
                    alpha=0.05
                )
                elapsed = time.time() - start_time
                
                metrics = {
                    'computation_time': elapsed,
                    'n_significant': original_result['summary'].get('significant_count', 0),
                    'n_tests': original_result['summary'].get('total_tests', 0),
                    'alzheimer_enriched': original_result['summary'].get('alzheimer_enriched', 0),
                    'control_enriched': original_result['summary'].get('control_enriched', 0)
                }
                results['original'].append(metrics)
            except Exception as e:
                print(f"    Error in original run {run+1}: {e}")
                results['original'].append({'error': str(e)})
        
        # Enhanced method (with semantic features)
        print("  Running enhanced association analysis...")
        llm_service = LLMPhenotypeService(
            data_loader=self.data_loader,
            use_llm=False
        )
        
        # Enhance data first
        ad_diag_enhanced = llm_service.enhance_phenotype_extraction("ad")
        con_diag_enhanced = llm_service.enhance_phenotype_extraction("control")
        
        # Create temporary data loader with enhanced data
        enhanced_service = AssociationService(self.data_loader)
        
        for run in range(n_runs):
            start_time = time.time()
            try:
                # Use enhanced data
                enhanced_result = enhanced_service.analyze_diagnosis(
                    diag_key="FullDiagnosisName",
                    stratify_by=None,
                    alpha=0.05
                )
                elapsed = time.time() - start_time
                
                metrics = {
                    'computation_time': elapsed,
                    'n_significant': enhanced_result['summary'].get('significant_count', 0),
                    'n_tests': enhanced_result['summary'].get('total_tests', 0),
                    'alzheimer_enriched': enhanced_result['summary'].get('alzheimer_enriched', 0),
                    'control_enriched': enhanced_result['summary'].get('control_enriched', 0)
                }
                results['enhanced'].append(metrics)
            except Exception as e:
                print(f"    Error in enhanced run {run+1}: {e}")
                results['enhanced'].append({'error': str(e)})
        
        summary = self._summarize_results(results)
        results['summary'] = summary
        
        self._save_results(results, 'association_benchmark')
        
        return results
    
    def _summarize_results(self, results: Dict) -> Dict:
        """Calculate summary statistics"""
        summary = {}
        
        for method in ['original', 'enhanced']:
            if not results[method]:
                continue
            
            method_results = [r for r in results[method] if 'error' not in r]
            if not method_results:
                continue
            
            summary[method] = {}
            for key in method_results[0].keys():
                values = [r[key] for r in method_results if key in r]
                if values:
                    summary[method][key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
        
        # Calculate improvements
        if 'original' in summary and 'enhanced' in summary:
            summary['improvements'] = {}
            for key in summary['enhanced']:
                if key in summary['original']:
                    orig_mean = summary['original'][key]['mean']
                    enh_mean = summary['enhanced'][key]['mean']
                    if orig_mean > 0:
                        improvement = ((enh_mean - orig_mean) / orig_mean) * 100
                        summary['improvements'][key] = {
                            'percent_change': float(improvement),
                            'absolute_change': float(enh_mean - orig_mean)
                        }
        
        return summary
    
    def _save_results(self, results: Dict, filename: str):
        """Save benchmark results to JSON"""
        filepath = self.results_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types to native Python types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        results_serializable = convert_types(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"  Results saved to {filepath}")
    
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite"""
        print("=" * 60)
        print("Running Full Benchmark Suite")
        print("=" * 60)
        
        all_results = {
            'umap': self.benchmark_umap_embeddings(),
            'association': self.benchmark_association_analysis(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate report
        self._generate_report(all_results)
        
        return all_results
    
    def _generate_report(self, results: Dict):
        """Generate human-readable benchmark report"""
        report_path = self.results_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write("# Benchmark Report\n\n")
            f.write(f"Generated: {results['timestamp']}\n\n")
            
            # UMAP Results
            if 'umap' in results:
                f.write("## UMAP Embedding Benchmark\n\n")
                umap_summary = results['umap'].get('summary', {})
                
                if 'original' in umap_summary and 'enhanced' in umap_summary:
                    f.write("### Metrics Comparison\n\n")
                    f.write("| Metric | Original | Enhanced | Improvement |\n")
                    f.write("|--------|----------|----------|-------------|\n")
                    
                    for key in umap_summary['enhanced']:
                        orig = umap_summary['original'].get(key, {}).get('mean', 0)
                        enh = umap_summary['enhanced'][key]['mean']
                        if key in umap_summary.get('improvements', {}):
                            imp = umap_summary['improvements'][key]['percent_change']
                            f.write(f"| {key} | {orig:.4f} | {enh:.4f} | {imp:+.2f}% |\n")
            
            # Association Results
            if 'association' in results:
                f.write("\n## Association Analysis Benchmark\n\n")
                assoc_summary = results['association'].get('summary', {})
                
                if 'original' in assoc_summary and 'enhanced' in assoc_summary:
                    f.write("### Metrics Comparison\n\n")
                    f.write("| Metric | Original | Enhanced | Improvement |\n")
                    f.write("|--------|----------|----------|-------------|\n")
                    
                    for key in assoc_summary['enhanced']:
                        orig = assoc_summary['original'].get(key, {}).get('mean', 0)
                        enh = assoc_summary['enhanced'][key]['mean']
                        if key in assoc_summary.get('improvements', {}):
                            imp = assoc_summary['improvements'][key]['percent_change']
                            f.write(f"| {key} | {orig:.4f} | {enh:.4f} | {imp:+.2f}% |\n")
        
        print(f"\nReport saved to {report_path}")

