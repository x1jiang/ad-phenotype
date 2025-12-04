/**
 * Dashboard JavaScript for AD Phenotyping Platform
 * Handles model comparison, visualizations, and interactions
 */

// Model comparison functionality
async function loadModelComparison(modelType) {
    const metricsDiv = document.getElementById('model-comparison-metrics');
    
    if (!metricsDiv) return;
    
    // Show loading state
    metricsDiv.innerHTML = `
        <div class="col-12 text-center">
            <div class="spinner-border text-primary" role="status"></div>
            <p class="mt-2">Running ${modelType} model comparison...</p>
        </div>
    `;
    
    try {
        let endpoint = '/api/models/compare/all';
        if (modelType !== 'all') {
            endpoint = `/api/models/compare/${modelType}`;
        }
        
        const response = await fetch(endpoint);
        const data = await response.json();
        
        if (data.status === 'success') {
            displayModelComparison(data, modelType);
        } else {
            throw new Error('Failed to load model comparison');
        }
    } catch (error) {
        console.error('Error loading model comparison:', error);
        metricsDiv.innerHTML = `
            <div class="col-12">
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle"></i> Error loading model comparison: ${error.message}
                </div>
            </div>
        `;
    }
}

function displayModelComparison(data, modelType) {
    const metricsDiv = document.getElementById('model-comparison-metrics');
    
    if (modelType === 'all' && data.models) {
        // Display all three models
        const models = [data.models.baseline, data.models.enhanced, data.models.llm];
        const improvements = data.improvements;
        
        let html = '<div class="row">';
        
        models.forEach((model, index) => {
            const modelName = model.model;
            const improvementKey = index === 1 ? 'enhanced_vs_baseline' : 
                                  index === 2 ? 'llm_vs_baseline' : null;
            const improvement = improvementKey ? improvements[improvementKey] : 0;
            
            const cardClass = getMetricCardClass(model.silhouette_score);
            const improvementClass = improvement > 0 ? 'positive' : 'negative';
            
            html += `
                <div class="col-md-4 mb-4 fade-in" style="animation-delay: ${index * 0.1}s">
                    <div class="metric-card ${cardClass}">
                        <div class="metric-label">${modelName}</div>
                        <div class="metric-value">${model.silhouette_score ? model.silhouette_score.toFixed(4) : 'N/A'}</div>
                        <small class="text-muted">Silhouette Score</small>
                        ${improvement !== 0 ? `
                            <div class="metric-improvement ${improvementClass}">
                                <i class="bi bi-${improvement > 0 ? 'arrow-up' : 'arrow-down'}"></i>
                                ${Math.abs(improvement).toFixed(1)}% vs Baseline
                            </div>
                        ` : ''}
                        <hr>
                        <div class="small text-start mt-3">
                            <div><strong>Features:</strong> ${model.n_features}</div>
                            <div><strong>Patients:</strong> ${model.n_patients}</div>
                            <div><strong>Time:</strong> ${model.execution_time ? model.execution_time.toFixed(2) : 'N/A'}s</div>
                            ${model.davies_bouldin_score ? `<div><strong>DB Score:</strong> ${model.davies_bouldin_score.toFixed(3)}</div>` : ''}
                            ${model.calinski_harabasz_score ? `<div><strong>CH Score:</strong> ${model.calinski_harabasz_score.toFixed(1)}</div>` : ''}
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        
        // Add summary section
        if (data.summary) {
            html += `
                <div class="row mt-4">
                    <div class="col-12">
                        <div class="alert alert-success">
                            <h5><i class="bi bi-trophy-fill"></i> Best Model: ${data.summary.best_model}</h5>
                            <p class="mb-0">Total execution time: ${data.summary.total_execution_time.toFixed(2)}s</p>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Add comparison chart
        html += '<div class="row mt-4"><div class="col-12"><div id="comparison-chart"></div></div></div>';
        
        metricsDiv.innerHTML = html;
        
        // Create comparison chart
        createComparisonChart(models);
        
    } else if (data.results) {
        // Display single model
        const model = data.results;
        const cardClass = getMetricCardClass(model.silhouette_score);
        
        metricsDiv.innerHTML = `
            <div class="row">
                <div class="col-md-6 offset-md-3">
                    <div class="metric-card ${cardClass}">
                        <div class="metric-label">${model.model}</div>
                        <div class="metric-value">${model.silhouette_score ? model.silhouette_score.toFixed(4) : 'N/A'}</div>
                        <small class="text-muted">Silhouette Score</small>
                        <hr>
                        <div class="small text-start mt-3">
                            <div><strong>Features:</strong> ${model.n_features}</div>
                            <div><strong>Patients:</strong> ${model.n_patients}</div>
                            <div><strong>Time:</strong> ${model.execution_time ? model.execution_time.toFixed(2) : 'N/A'}s</div>
                            ${model.feature_types ? `<div><strong>Feature Types:</strong> ${model.feature_types.join(', ')}</div>` : ''}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
}

function getMetricCardClass(score) {
    if (score >= 0.5) return 'excellent';
    if (score >= 0.3) return 'good';
    if (score >= 0.1) return 'fair';
    return 'poor';
}

function createComparisonChart(models) {
    const chartDiv = document.getElementById('comparison-chart');
    if (!chartDiv) return;
    
    const modelNames = models.map(m => m.model);
    const silhouetteScores = models.map(m => m.silhouette_score || 0);
    const dbScores = models.map(m => m.davies_bouldin_score || 0);
    const chScores = models.map(m => m.calinski_harabasz_score || 0);
    
    const trace1 = {
        x: modelNames,
        y: silhouetteScores,
        name: 'Silhouette Score',
        type: 'bar',
        marker: {
            color: 'rgba(102, 126, 234, 0.8)',
            line: {
                color: 'rgba(102, 126, 234, 1)',
                width: 2
            }
        }
    };
    
    const trace2 = {
        x: modelNames,
        y: dbScores,
        name: 'Davies-Bouldin Score',
        type: 'bar',
        marker: {
            color: 'rgba(245, 87, 108, 0.8)',
            line: {
                color: 'rgba(245, 87, 108, 1)',
                width: 2
            }
        }
    };
    
    const trace3 = {
        x: modelNames,
        y: chScores.map(s => s / 100), // Normalize for visualization
        name: 'Calinski-Harabasz Score (÷100)',
        type: 'bar',
        marker: {
            color: 'rgba(67, 233, 123, 0.8)',
            line: {
                color: 'rgba(67, 233, 123, 1)',
                width: 2
            }
        }
    };
    
    const layout = {
        title: 'Model Performance Comparison',
        barmode: 'group',
        xaxis: { title: 'Model' },
        yaxis: { title: 'Score' },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'inherit' }
    };
    
    Plotly.newPlot(chartDiv, [trace1, trace2, trace3], layout, {responsive: true});
}

// Phenotype explanations
async function loadPhenotypeExplanations() {
    const resultsDiv = document.getElementById('phenotype-explanations');
    if (!resultsDiv) return;
    
    resultsDiv.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status"></div>
            <p class="mt-2">Generating AI insights with GPT-5.1...</p>
        </div>
    `;
    
    try {
        // Get top phenotypes from data
        const response = await fetch('/api/analysis/top-phenotypes?limit=5');
        const data = await response.json();
        
        if (data.phenotypes) {
            displayPhenotypeExplanations(data.phenotypes);
        }
    } catch (error) {
        console.error('Error loading phenotype explanations:', error);
        resultsDiv.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i> Error loading phenotype explanations
            </div>
        `;
    }
}

function displayPhenotypeExplanations(phenotypes) {
    const resultsDiv = document.getElementById('phenotype-explanations');
    
    let html = '<div class="row">';
    
    phenotypes.forEach((phenotype, index) => {
        const severityClass = `severity-${phenotype.severity ? phenotype.severity.toLowerCase() : 'moderate'}`;
        
        html += `
            <div class="col-12 mb-3 fade-in" style="animation-delay: ${index * 0.1}s">
                <div class="phenotype-card">
                    <div class="phenotype-title">
                        <i class="bi bi-clipboard-pulse"></i> ${phenotype.name}
                    </div>
                    <div class="phenotype-explanation">
                        ${phenotype.explanation || 'Loading explanation...'}
                    </div>
                    <div>
                        <span class="severity-badge ${severityClass}">
                            ${phenotype.severity || 'Moderate'}
                        </span>
                        ${phenotype.count ? `<span class="badge bg-secondary">${phenotype.count} patients</span>` : ''}
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    
    resultsDiv.innerHTML = html;
}

// Method comparison
async function comparePhenotypeMethods() {
    const resultsDiv = document.getElementById('phenotype-explanations');
    if (!resultsDiv) return;
    
    resultsDiv.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status"></div>
            <p class="mt-2">Comparing phenotyping methods...</p>
        </div>
    `;
    
    try {
        const response = await fetch('/api/models/compare/all');
        const data = await response.json();
        
        if (data.status === 'success') {
            displayMethodComparison(data);
        }
    } catch (error) {
        console.error('Error comparing methods:', error);
        resultsDiv.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i> Error comparing methods
            </div>
        `;
    }
}

function displayMethodComparison(data) {
    const resultsDiv = document.getElementById('phenotype-explanations');
    
    const models = [data.models.baseline, data.models.enhanced, data.models.llm];
    
    let html = `
        <div class="comparison-table">
            <div class="comparison-row comparison-header">
                <div class="comparison-cell">Metric</div>
                <div class="comparison-cell">Baseline</div>
                <div class="comparison-cell">Enhanced</div>
                <div class="comparison-cell">LLM (GPT-5.1)</div>
            </div>
    `;
    
    const metrics = [
        { key: 'silhouette_score', label: 'Silhouette Score', format: v => v.toFixed(4) },
        { key: 'davies_bouldin_score', label: 'Davies-Bouldin', format: v => v.toFixed(3) },
        { key: 'calinski_harabasz_score', label: 'Calinski-Harabasz', format: v => v.toFixed(1) },
        { key: 'n_features', label: 'Features', format: v => v },
        { key: 'execution_time', label: 'Time (s)', format: v => v.toFixed(2) }
    ];
    
    metrics.forEach(metric => {
        html += `
            <div class="comparison-row">
                <div class="comparison-cell"><strong>${metric.label}</strong></div>
        `;
        
        models.forEach(model => {
            const value = model[metric.key];
            html += `<div class="comparison-cell">${value ? metric.format(value) : 'N/A'}</div>`;
        });
        
        html += '</div>';
    });
    
    html += '</div>';
    
    resultsDiv.innerHTML = html;
}

// Full benchmark
async function runFullBenchmark() {
    const resultsDiv = document.getElementById('benchmark-results');
    if (!resultsDiv) return;
    
    resultsDiv.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status"></div>
            <p class="mt-2">Running comprehensive benchmark (this may take a few minutes)...</p>
        </div>
    `;
    
    try {
        const response = await fetch('/api/models/compare/all');
        const data = await response.json();
        
        if (data.status === 'success') {
            displayFullBenchmark(data);
        }
    } catch (error) {
        console.error('Error running benchmark:', error);
        resultsDiv.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i> Error running benchmark: ${error.message}
            </div>
        `;
    }
}

function displayFullBenchmark(data) {
    const resultsDiv = document.getElementById('benchmark-results');
    
    let html = `
        <div class="row">
            <div class="col-12 mb-4">
                <div class="alert alert-info">
                    <h5><i class="bi bi-info-circle-fill"></i> Benchmark Complete</h5>
                    <p class="mb-0">Best performing model: <strong>${data.summary.best_model}</strong></p>
                    <p class="mb-0">Total execution time: ${data.summary.total_execution_time.toFixed(2)}s</p>
                </div>
            </div>
        </div>
    `;
    
    // Display detailed results
    displayModelComparison(data, 'all');
    
    resultsDiv.innerHTML = html + document.getElementById('model-comparison-metrics').innerHTML;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialized');
    
    // Auto-load model comparison on dashboard load
    const metricsDiv = document.getElementById('model-comparison-metrics');
    if (metricsDiv) {
        // Don't auto-load to save API calls - user can click button
        console.log('Model comparison ready');
    }
});

// Export functions for global access
window.loadModelComparison = loadModelComparison;
window.loadPhenotypeExplanations = loadPhenotypeExplanations;
window.comparePhenotypeMethods = comparePhenotypeMethods;
window.runFullBenchmark = runFullBenchmark;
