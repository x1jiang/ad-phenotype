// Upload page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    initializeUploadZones();
    initializeStatusCheck();
});

function initializeUploadZones() {
    const uploadZones = document.querySelectorAll('.upload-zone');
    
    uploadZones.forEach(zone => {
        const fileInput = zone.querySelector('.file-input');
        const fileInfo = zone.querySelector('.file-info');
        
        // Click to upload
        zone.addEventListener('click', () => {
            fileInput.click();
        });
        
        // Drag and drop
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('drag-over');
        });
        
        zone.addEventListener('dragleave', () => {
            zone.classList.remove('drag-over');
        });
        
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('drag-over');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect(fileInput);
            }
        });
        
        // File input change
        fileInput.addEventListener('change', function() {
            handleFileSelect(this);
        });
    });
}

function handleFileSelect(input) {
    const file = input.files[0];
    if (!file) return;
    
    const zone = input.closest('.upload-zone');
    const fileInfo = zone.querySelector('.file-info');
    const fileName = fileInfo.querySelector('.file-name');
    const fileSize = fileInfo.querySelector('.file-size');
    
    // Validate file
    if (!file.name.endsWith('.csv')) {
        showStatus('error', 'Please upload a CSV file');
        return;
    }
    
    // Show file info
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.style.display = 'block';
    
    // Upload file
    uploadFile(file, input.dataset.cohort, input.dataset.type, zone);
}

function uploadFile(file, cohort, fileType, zone) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Show uploading state
    zone.classList.add('uploading');
    zone.classList.remove('uploaded');
    
    // Create progress indicator
    let progressContainer = zone.querySelector('.progress-container');
    if (!progressContainer) {
        progressContainer = document.createElement('div');
        progressContainer.className = 'progress-container';
        progressContainer.innerHTML = `
            <div class="progress" style="height: 6px;">
                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                     role="progressbar" style="width: 0%"></div>
            </div>
        `;
        zone.appendChild(progressContainer);
    }
    progressContainer.classList.add('active');
    const progressBar = progressContainer.querySelector('.progress-bar');
    
    // Upload
    fetch(`/api/upload/${cohort}/${fileType}`, {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.detail || 'Upload failed');
            });
        }
        return response.json();
    })
    .then(data => {
        // Success
        zone.classList.remove('uploading');
        zone.classList.add('uploaded');
        progressContainer.classList.remove('active');
        
        showStatus('success', `File uploaded successfully: ${data.filename} (${data.row_count} rows)`);
        
        // Update file info
        const fileInfo = zone.querySelector('.file-info');
        fileInfo.innerHTML = `
            <i class="bi bi-check-circle text-success"></i>
            <span class="file-name">${data.filename}</span>
            <span class="file-size">${formatFileSize(data.file_size)} • ${data.row_count} rows</span>
            <button class="btn btn-sm btn-outline-danger btn-delete ms-2" 
                    onclick="deleteFile('${cohort}', '${fileType}', this)">
                <i class="bi bi-trash"></i>
            </button>
        `;
        
        // Refresh status table
        setTimeout(() => {
            htmx.trigger('#status-table', 'refresh');
        }, 500);
    })
    .catch(error => {
        // Error
        zone.classList.remove('uploading');
        progressContainer.classList.remove('active');
        showStatus('error', `Upload failed: ${error.message}`);
    });
}

function deleteFile(cohort, fileType, button) {
    if (!confirm(`Are you sure you want to delete ${cohort}_${fileType}.csv?`)) {
        return;
    }
    
    fetch(`/api/upload/${cohort}/${fileType}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            const zone = button.closest('.upload-zone');
            zone.classList.remove('uploaded');
            const fileInfo = zone.querySelector('.file-info');
            fileInfo.style.display = 'none';
            const fileInput = zone.querySelector('.file-input');
            fileInput.value = '';
            
            showStatus('success', data.message);
            
            // Refresh status table
            setTimeout(() => {
                htmx.trigger('#status-table', 'refresh');
            }, 500);
        }
    })
    .catch(error => {
        showStatus('error', `Delete failed: ${error.message}`);
    });
}

function showStatus(type, message) {
    const banner = document.getElementById('status-banner');
    const messageEl = document.getElementById('status-message');
    
    banner.className = `alert alert-${type} show`;
    messageEl.textContent = message;
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        banner.classList.remove('show');
    }, 5000);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function initializeStatusCheck() {
    const checkBtn = document.getElementById('check-status-btn');
    if (checkBtn) {
        checkBtn.addEventListener('click', () => {
            htmx.trigger('#status-table', 'refresh');
            showStatus('info', 'Status refreshed');
        });
    }
    
    // Auto-refresh status every 10 seconds
    setInterval(() => {
        htmx.trigger('#status-table', 'refresh');
    }, 10000);
}

// Preload sample data from Data/ folder
function preloadData(cohort) {
    const btn = document.getElementById(`preload-${cohort}-btn`);
    const originalText = btn.innerHTML;
    
    // Disable button and show loading
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Loading...';
    
    showStatus('info', `Preloading ${cohort} cohort data from Data/ folder...`);
    
    // Make API call to preload data
    fetch(`/api/upload/preload/${cohort}`, {
        method: 'POST'
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.detail || 'Preload failed');
            });
        }
        return response.json();
    })
    .then(data => {
        // Success
        btn.disabled = false;
        btn.innerHTML = originalText;
        
        showStatus('success', `✅ Preloaded ${data.files_loaded} files for ${cohort} cohort (${data.total_rows} total rows)`);
        
        // Update all upload zones for this cohort to show as uploaded
        const panel = document.getElementById(`${cohort}-panel`);
        panel.querySelectorAll('.upload-zone').forEach(zone => {
            zone.classList.add('uploaded');
        });
        
        // Refresh status table
        setTimeout(() => {
            htmx.trigger('#status-table', 'refresh');
        }, 500);
    })
    .catch(error => {
        // Error
        btn.disabled = false;
        btn.innerHTML = originalText;
        showStatus('error', `Preload failed: ${error.message}`);
    });
}

// HTMX swap handler for status table
htmx.on('#status-table', 'htmx:afterSwap', function(evt) {
    // Add delete button handlers
    document.querySelectorAll('.btn-delete').forEach(btn => {
        btn.addEventListener('click', function() {
            const cohort = this.dataset.cohort;
            const fileType = this.dataset.fileType;
            deleteFile(cohort, fileType, this);
        });
    });
});

