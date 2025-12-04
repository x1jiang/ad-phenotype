// HTMX event handlers and custom JavaScript

document.body.addEventListener('htmx:beforeRequest', function(evt) {
    // Show spinner
    const target = evt.detail.target;
    const spinner = target.querySelector('.spinner-border');
    if (spinner) {
        spinner.classList.remove('d-none');
    }
});

document.body.addEventListener('htmx:afterRequest', function(evt) {
    // Hide spinner
    const spinner = document.querySelector('.spinner-border');
    if (spinner) {
        spinner.classList.add('d-none');
    }
});

// Handle UMAP results - HTMX will swap JSON as text, so we need to parse it
htmx.on('#umap-results', 'htmx:afterSwap', function(evt) {
    try {
        // HTMX swaps the response as-is, so we need to parse JSON from the element
        const text = evt.detail.target.textContent || evt.detail.target.innerText;
        const response = JSON.parse(text);
        if (response.data && response.data.embedding) {
            evt.detail.target.innerHTML = '<div id="umap-plot"></div>';
            renderUMAP(response.data);
        }
    } catch (e) {
        // Try parsing from xhr directly
        try {
            const response = JSON.parse(evt.detail.xhr.response);
            if (response.data && response.data.embedding) {
                evt.detail.target.innerHTML = '<div id="umap-plot"></div>';
                renderUMAP(response.data);
            }
        } catch (e2) {
            console.error('Error parsing UMAP response:', e2);
        }
    }
});

function renderUMAP(data) {
    const embedding = data.embedding;
    const labels = data.labels;
    const patientIds = data.patient_ids || [];
    
    const traces = [];
    const uniqueLabels = [...new Set(labels)];
    const colors = {
        'Alzheimer': '#dc3545',
        'Control': '#0d6efd'
    };
    
    uniqueLabels.forEach(label => {
        const mask = labels.map((l, i) => l === label);
        const x = embedding.filter((_, i) => mask[i]).map(e => e[0]);
        const y = embedding.filter((_, i) => mask[i]).map(e => e[1]);
        const ids = patientIds.filter((_, i) => mask[i]);
        
        traces.push({
            x: x,
            y: y,
            mode: 'markers',
            type: 'scatter',
            name: label,
            marker: {
                size: 6,
                opacity: 0.7,
                color: colors[label] || '#6c757d',
                line: {
                    width: 0.5,
                    color: 'white'
                }
            },
            text: ids,
            hovertemplate: `<b>${label}</b><br>Patient ID: %{text}<br>UMAP 1: %{x:.2f}<br>UMAP 2: %{y:.2f}<extra></extra>`
        });
    });
    
    const layout = {
        title: {
            text: 'UMAP Embedding: Patient Clustering',
            font: { size: 18 }
        },
        xaxis: { 
            title: 'UMAP Dimension 1',
            showgrid: true,
            gridcolor: '#e9ecef',
            zeroline: false
        },
        yaxis: { 
            title: 'UMAP Dimension 2',
            showgrid: true,
            gridcolor: '#e9ecef',
            zeroline: false
        },
        hovermode: 'closest',
        height: 700,
        plot_bgcolor: '#f8f9fa',
        paper_bgcolor: 'white',
        legend: {
            x: 0.02,
            y: 0.98,
            bgcolor: 'rgba(255,255,255,0.8)'
        }
    };
    
    Plotly.newPlot('umap-plot', traces, layout, {
        responsive: true,
        displayModeBar: true
    });
    
    // Load performance metrics
    setTimeout(() => {
        fetch('/api/performance/umap/metrics?use_enhanced=false')
            .then(r => r.json())
            .then(data => {
                if (data.metrics) {
                    const metricsContainer = document.getElementById('performance-metrics');
                    if (metricsContainer) {
                        renderPerformanceMetrics(metricsContainer, data.metrics);
                    }
                }
            })
            .catch(e => console.error('Error loading metrics:', e));
    }, 1000);
}

// Handle association results
htmx.on('#diagnosis-results, #medications-results, #labs-results', 'htmx:afterSwap', function(evt) {
    try {
        const text = evt.detail.target.textContent || evt.detail.target.innerText;
        const response = JSON.parse(text);
        if (response.data && response.data.results) {
            renderAssociationResults(evt.detail.target, response.data);
        }
    } catch (e) {
        try {
            const response = JSON.parse(evt.detail.xhr.response);
            if (response.data && response.data.results) {
                renderAssociationResults(evt.detail.target, response.data);
            }
        } catch (e2) {
            console.error('Error parsing association response:', e2);
        }
    }
});

function renderAssociationResults(container, data) {
    const results = data.results;
    const summary = data.summary;
    
    // Create summary cards with enhanced styling
    let html = '<div class="row mb-4">';
    if (summary) {
        html += `
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-value">${summary.total_tests || 0}</div>
                    <div class="metric-label">Total Tests</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-value text-success">${summary.significant_count || 0}</div>
                    <div class="metric-label">Significant</div>
                    <div class="metric-change">${((summary.significant_count / summary.total_tests) * 100).toFixed(1)}% rate</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-value text-danger">${summary.alzheimer_enriched || 0}</div>
                    <div class="metric-label">AD Enriched</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-value text-primary">${summary.control_enriched || 0}</div>
                    <div class="metric-label">Control Enriched</div>
                </div>
            </div>
        `;
    }
    html += '</div>';
    
    // Create volcano plot
    html += '<div class="plot-container"><div id="volcano-plot"></div></div>';
    
    // Create interactive results table with sorting
    html += `
        <div class="card mt-4">
            <div class="card-header">
                <h6 class="mb-0"><i class="bi bi-table"></i> Detailed Results (Top 100)</h6>
            </div>
            <div class="card-body">
                <div class="table-responsive" style="max-height: 500px; overflow-y: auto;">
                    <table class="table table-striped table-hover table-sm">
                        <thead class="table-light sticky-top">
                            <tr>
                                <th>Phenotype</th>
                                <th>Odds Ratio</th>
                                <th>Log2 OR</th>
                                <th>P-value</th>
                                <th>-Log10 P</th>
                                <th>Status</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody>
    `;
    
    // Sort by p-value and take top 100
    const sortedResults = [...results].sort((a, b) => 
        (a.pvalue || 1) - (b.pvalue || 1)
    ).slice(0, 100);
    
    sortedResults.forEach(row => {
        const phenotype = row.FullDiagnosisName || row.DiagnosisName || row.MedicationGenericName || row.TestName || 'Unknown';
        const enriched = row.enriched || 'Not Significant';
        const enrichedClass = enriched === 'Alzheimer Enriched' ? 'danger' : 
                             enriched === 'Control Enriched' ? 'primary' : 'secondary';
        
        html += `
            <tr>
                <td><strong>${phenotype}</strong></td>
                <td>${(row.odds_ratio || 0).toFixed(3)}</td>
                <td>${(row.log2_odds_ratio || 0).toFixed(3)}</td>
                <td>${(row.pvalue || 1).toExponential(2)}</td>
                <td>${(row.neg_log10_pvalue || 0).toFixed(2)}</td>
                <td><span class="badge bg-${enrichedClass}">${enriched}</span></td>
                <td>
                    <button class="btn btn-sm btn-outline-info" onclick="explainPhenotype('${phenotype}')">
                        <i class="bi bi-info-circle"></i> Explain
                    </button>
                </td>
            </tr>
        `;
    });
    
    html += `
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
    
    // Render volcano plot
    if (results.length > 0) {
        renderVolcanoPlot(results, summary);
    }
}

// Explain phenotype function
async function explainPhenotype(phenotypeName) {
    try {
        const response = await fetch(`/api/performance/phenotype/explain/${encodeURIComponent(phenotypeName)}`);
        const data = await response.json();
        
        if (data.explanation) {
            showPhenotypeModal(data.explanation);
        }
    } catch (e) {
        alert(`Error loading explanation: ${e.message}`);
    }
}

function showPhenotypeModal(explanation) {
    const modal = document.createElement('div');
    modal.className = 'modal fade show';
    modal.style.display = 'block';
    modal.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header bg-primary text-white">
                    <h5 class="modal-title">
                        <i class="bi bi-clipboard-data"></i> ${explanation.phenotype}
                    </h5>
                    <button type="button" class="btn-close btn-close-white" onclick="this.closest('.modal').remove()"></button>
                </div>
                <div class="modal-body">
                    <div class="phenotype-card">
                        <p><strong>Description:</strong> ${explanation.description}</p>
                        <p><strong>Clinical Significance:</strong> ${explanation.clinical_significance}</p>
                        <p><strong>AD Association:</strong> ${explanation.ad_association}</p>
                        <p><strong>Implications:</strong> ${explanation.implications}</p>
                        <p><strong>Severity:</strong> <span class="badge bg-${explanation.severity === 'Severe' ? 'danger' : explanation.severity === 'Moderate' ? 'warning' : 'success'}">${explanation.severity}</span></p>
                        ${explanation.common_comorbidities && explanation.common_comorbidities.length > 0 ? `
                            <div class="mt-3">
                                <strong>Common Comorbidities:</strong><br>
                                ${explanation.common_comorbidities.map(c => `<span class="badge bg-secondary me-1">${c}</span>`).join('')}
                            </div>
                        ` : ''}
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" onclick="this.closest('.modal').remove()">Close</button>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    document.body.style.overflow = 'hidden';
    
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.remove();
            document.body.style.overflow = '';
        }
    });
}

function renderVolcanoPlot(results, summary) {
    const x = results.map(r => r.log2_odds_ratio || r.log2_odds_ratio_F || 0);
    const y = results.map(r => r.neg_log10_pvalue || r.neg_log10_pvalue_F || 0);
    const texts = results.map(r => r.FullDiagnosisName || r.MedicationGenericName || r.TestName || 'Unknown');
    const enriched = results.map(r => r.enriched || 'Not Significant');
    
    // Create traces for each category
    const adEnriched = {
        x: x.filter((_, i) => enriched[i] === 'Alzheimer Enriched'),
        y: y.filter((_, i) => enriched[i] === 'Alzheimer Enriched'),
        text: texts.filter((_, i) => enriched[i] === 'Alzheimer Enriched'),
        mode: 'markers',
        type: 'scatter',
        name: 'AD Enriched',
        marker: { color: '#dc3545', size: 10, opacity: 0.7 },
        hovertemplate: '<b>%{text}</b><br>Log2 OR: %{x:.3f}<br>-Log10 P: %{y:.2f}<extra></extra>'
    };
    
    const controlEnriched = {
        x: x.filter((_, i) => enriched[i] === 'Control Enriched'),
        y: y.filter((_, i) => enriched[i] === 'Control Enriched'),
        text: texts.filter((_, i) => enriched[i] === 'Control Enriched'),
        mode: 'markers',
        type: 'scatter',
        name: 'Control Enriched',
        marker: { color: '#0d6efd', size: 10, opacity: 0.7 },
        hovertemplate: '<b>%{text}</b><br>Log2 OR: %{x:.3f}<br>-Log10 P: %{y:.2f}<extra></extra>'
    };
    
    const notSignificant = {
        x: x.filter((_, i) => enriched[i] === 'Not Significant'),
        y: y.filter((_, i) => enriched[i] === 'Not Significant'),
        text: texts.filter((_, i) => enriched[i] === 'Not Significant'),
        mode: 'markers',
        type: 'scatter',
        name: 'Not Significant',
        marker: { color: '#6c757d', size: 6, opacity: 0.4 },
        hovertemplate: '<b>%{text}</b><br>Log2 OR: %{x:.3f}<br>-Log10 P: %{y:.2f}<extra></extra>'
    };
    
    const threshold = -Math.log10(summary?.corrected_alpha || 0.05);
    
    const layout = {
        title: {
            text: 'Volcano Plot: Association Analysis',
            font: { size: 18 }
        },
        xaxis: { 
            title: 'Log₂(Odds Ratio)',
            zeroline: true,
            zerolinecolor: '#ccc'
        },
        yaxis: { 
            title: '-Log₁₀(p-value)',
            zeroline: false
        },
        shapes: [
            {
                type: 'line',
                x0: -10, x1: 10,
                y0: threshold,
                y1: threshold,
                line: { dash: 'dash', color: '#333', width: 2 },
                annotation: {
                    text: `Significance Threshold (α=${(summary?.corrected_alpha || 0.05).toExponential(2)})`,
                    showarrow: false,
                    x: 0,
                    y: threshold,
                    yshift: 10
                }
            },
            {
                type: 'line',
                x0: 1, x1: 1,
                y0: 0, y1: Math.max(...y) + 1,
                line: { dash: 'dot', color: '#999', width: 1 }
            },
            {
                type: 'line',
                x0: -1, x1: -1,
                y0: 0, y1: Math.max(...y) + 1,
                line: { dash: 'dot', color: '#999', width: 1 }
            }
        ],
        height: 600,
        hovermode: 'closest',
        legend: {
            x: 0.02,
            y: 0.98,
            bgcolor: 'rgba(255,255,255,0.8)'
        }
    };
    
    Plotly.newPlot('volcano-plot', [notSignificant, controlEnriched, adEnriched], layout, {
        responsive: true,
        displayModeBar: true
    });
}

// Handle network results
htmx.on('#network-results', 'htmx:afterSwap', function(evt) {
    try {
        const text = evt.detail.target.textContent || evt.detail.target.innerText;
        const response = JSON.parse(text);
        if (response.data && response.job_id) {
            renderNetworkResults(evt.detail.target, response.data, response.job_id);
        }
    } catch (e) {
        try {
            const response = JSON.parse(evt.detail.xhr.response);
            if (response.data && response.job_id) {
                renderNetworkResults(evt.detail.target, response.data, response.job_id);
            }
        } catch (e2) {
            console.error('Error parsing network response:', e2);
        }
    }
});

function renderNetworkResults(container, data, jobId) {
    let html = `
        <div class="alert alert-success">
            <h5>Network Built Successfully!</h5>
            <p><strong>AD Network:</strong> ${data.ad_network.nodes} nodes, ${data.ad_network.edges} edges</p>
            <p><strong>Control Network:</strong> ${data.control_network.nodes} nodes, ${data.control_network.edges} edges</p>
            <p><small>Network files saved. Use Cytoscape to visualize the GraphML files.</small></p>
        </div>
    `;
    container.innerHTML = html;
}

// Handle summary cards
htmx.on('#summary-cards', 'htmx:afterSwap', function(evt) {
    try {
        const data = JSON.parse(evt.detail.xhr.response);
        renderSummaryCards(evt.detail.target, data);
    } catch (e) {
        console.error('Error parsing summary response:', e);
    }
});

function renderSummaryCards(container, data) {
    const html = `
        <div class="row">
            <div class="col-md-3">
                <div class="card summary-card">
                    <div class="card-body text-center">
                        <div class="stat-number">${data.ad?.patients || 0}</div>
                        <div class="stat-label">AD Patients</div>
                        <small class="text-muted">${data.ad?.female || 0} Female, ${data.ad?.male || 0} Male</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card summary-card">
                    <div class="card-body text-center">
                        <div class="stat-number">${data.control?.patients || 0}</div>
                        <div class="stat-label">Control Patients</div>
                        <small class="text-muted">${data.control?.female || 0} Female, ${data.control?.male || 0} Male</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card summary-card">
                    <div class="card-body text-center">
                        <div class="stat-number">${data.total_patients || 0}</div>
                        <div class="stat-label">Total Patients</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card summary-card">
                    <div class="card-body text-center">
                        <div class="stat-number">${(data.ad?.with_diagnosis || 0) + (data.control?.with_diagnosis || 0)}</div>
                        <div class="stat-label">With Diagnosis</div>
                    </div>
                </div>
            </div>
        </div>
    `;
    container.innerHTML = html;
}

