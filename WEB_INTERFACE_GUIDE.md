# 🌐 Web Interface Guide - ADRD Phenotyping Platform

## 🚀 Quick Start

```bash
# 1. Install dependencies (one-time setup)
pip install -r requirements.txt

# 2. Start the server
python3 run.py

# 3. Open browser
open http://localhost:8000
```

---

## 📱 Main Dashboard (`/`)

### Interface Layout

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                       AD Deep Phenotyping Platform                        ║
║                    [Upload Data] [Powered by GPT-5.1]                     ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  🎯 MULTI-MODEL PERFORMANCE COMPARISON                                    ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  [Compare All Models] [Baseline] [Enhanced] [LLM]                │   ║
║  │                                                                    │   ║
║  │  📊 Interactive Performance Charts:                               │   ║
║  │     • ROC Curves (3 models side-by-side)                          │   ║
║  │     • Bar charts comparing AUC, F1, Accuracy                      │   ║
║  │     • Processing time comparison                                  │   ║
║  │     • Interactive Plotly visualizations                           │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                                                                           ║
║  📈 DATA SUMMARY CARDS                                                    ║
║  ┌─────────────────┬─────────────────┬─────────────────┐                ║
║  │   🏥 1,000      │   👥 1,000      │   🧬 114        │                ║
║  │   AD Patients   │   Controls      │   Concepts      │                ║
║  │                 │                 │                 │                ║
║  │   📊 7 Types    │   🕸️ 92,869     │   ⚡ 1.42s      │                ║
║  │   Data Sources  │   Relations     │   Processing    │                ║
║  └─────────────────┴─────────────────┴─────────────────┘                ║
║                                                                           ║
║  📑 ANALYSIS TABS                                                         ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │ 🔬 UMAP | 📊 Association | 🧬 Phenotypes | 🕸️ Network | ⚡ Metrics │   ║
║  ├──────────────────────────────────────────────────────────────────┤   ║
║  │                                                                    │   ║
║  │  [Content dynamically loads based on selected tab]                │   ║
║  │                                                                    │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## 📑 Tab Contents

### 🔬 Tab 1: UMAP Analysis

**What you see:**
- Interactive 2D scatter plot (Plotly)
- Color-coded points: 🔴 Red = AD patients, 🔵 Blue = Controls
- **Hover** over points to see patient details
- **Zoom** with mouse wheel
- **Pan** by clicking and dragging
- Clustering metrics displayed below:
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Score

**Buttons:**
- `[View 2D]` - Standard UMAP view
- `[View 3D]` - Interactive 3D visualization
- `[Export PNG]` - Save visualization

---

### 📊 Tab 2: Association Analysis

**What you see:**
- **Statistical comparison** between AD and Control groups
- **Tables showing:**
  - Feature name
  - AD value (mean ± std)
  - Control value (mean ± std)
  - P-value
  - Effect size
  - Test method (Chi-square, Mann-Whitney U)

**Key Features:**
- Automatically detects categorical vs continuous variables
- Applies appropriate statistical test
- Color-codes significant results (p < 0.05)
- Sortable columns

---

### 🧬 Tab 3: Phenotype Analysis

**What you see:**
- **Top discriminative features** between AD and Control
- **LLM-generated explanations** for each phenotype (if GPT-5.1 enabled)
- **Clinical interpretation** of findings

**Example Output:**
```
Top 5 Discriminative Features:
1. Essential hypertension (59% vs 48%, OR=1.6, p<0.001)
   💡 Explanation: Hypertension is a known ADRD risk factor...
   
2. Type 2 diabetes mellitus (43% vs 28%, OR=1.9, p<0.001)
   💡 Explanation: Metabolic dysfunction contributes to...
   
3. MRI Brain imaging (72% vs 45%, OR=3.1, p<0.001)
   💡 Explanation: Increased neuroimaging in AD patients...
```

---

### 🕸️ Tab 4: Comorbidity Network

**What you see:**
- **Interactive force-directed graph**
- **Nodes** = Medical conditions
- **Edges** = Co-occurrence in same patients
- **Node size** = Prevalence
- **Edge thickness** = Co-occurrence frequency

**Interactions:**
- **Hover** over nodes to see condition details
- **Click** nodes to highlight connections
- **Drag** nodes to rearrange layout
- **Zoom** with mouse wheel

**Legend:**
- 🔴 Red nodes: High prevalence in AD
- 🔵 Blue nodes: High prevalence in Controls
- 🟢 Green nodes: Equal prevalence

---

### ⚡ Tab 5: Performance Metrics

**What you see:**
- **Comprehensive performance table**
- **Three models compared:**
  1. Knowledge Graph Baseline (GAT)
  2. Enhanced Feature Engineering (22 features)
  3. LLM-Enhanced (GPT-5.1)

**Metrics displayed:**
- AUC-ROC
- Accuracy
- Sensitivity
- Specificity
- F1-Score
- Processing Time

**Visual indicators:**
- ✅ Green = Best performance
- ⚠️ Yellow = Good performance
- 🔴 Red = Needs improvement

---

## 📤 Upload Page (`/upload`)

### Two-Panel Interface

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                         Data Upload Interface                             ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  📁 AD COHORT DATA                                                        ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │                                                                    │   ║
║  │              🔽 Drag & Drop CSV Files Here                        │   ║
║  │                    or click to browse                             │   ║
║  │                                                                    │   ║
║  │  Accepted files:                                                  │   ║
║  │    • ad_demographics.csv                                          │   ║
║  │    • ad_diagnosis.csv                                             │   ║
║  │    • ad_medications.csv                                           │   ║
║  │    • ad_labresults.csv                                            │   ║
║  │    • ad_imaging.csv                                               │   ║
║  │    • ad_treatments.csv                                            │   ║
║  │    • ad_vitals.csv                                                │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                                                                           ║
║  📁 CONTROL COHORT DATA                                                   ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │                                                                    │   ║
║  │              🔽 Drag & Drop CSV Files Here                        │   ║
║  │                    or click to browse                             │   ║
║  │                                                                    │   ║
║  │  Accepted files:                                                  │   ║
║  │    • control_demographics.csv                                     │   ║
║  │    • control_diagnosis.csv                                        │   ║
║  │    • control_medications.csv                                      │   ║
║  │    • control_labresults.csv                                       │   ║
║  │    • control_imaging.csv                                          │   ║
║  │    • control_treatments.csv                                       │   ║
║  │    • control_vitals.csv                                           │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                                                                           ║
║  [Upload Files] [Reset] [Back to Dashboard]                              ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

**Features:**
- ✅ Drag & drop interface
- ✅ Progress bars for each file
- ✅ Validation feedback (✓ or ✗)
- ✅ Error messages for invalid formats
- ✅ Automatic detection of file types
- ✅ Real-time upload status

---

## 📊 API Documentation (`/docs`)

**FastAPI Auto-Generated Documentation**

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                         API Documentation (Swagger UI)                    ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  Endpoints:                                                               ║
║                                                                           ║
║  GET  /                                    [Try it out] ▼                 ║
║  GET  /upload                              [Try it out] ▼                 ║
║  GET  /api/data/summary                    [Try it out] ▼                 ║
║  GET  /api/umap                            [Try it out] ▼                 ║
║  GET  /api/umap/3d                         [Try it out] ▼                 ║
║  GET  /api/umap/metrics                    [Try it out] ▼                 ║
║  GET  /api/association                     [Try it out] ▼                 ║
║  GET  /api/phenotypes                      [Try it out] ▼                 ║
║  GET  /api/network                         [Try it out] ▼                 ║
║  GET  /api/model_comparison/compare        [Try it out] ▼                 ║
║  GET  /api/benchmark                       [Try it out] ▼                 ║
║  POST /api/upload/ad                       [Try it out] ▼                 ║
║  POST /api/upload/control                  [Try it out] ▼                 ║
║                                                                           ║
║  Each endpoint shows:                                                     ║
║    • Parameters                                                           ║
║    • Request body schema                                                  ║
║    • Response schema                                                      ║
║    • Example responses                                                    ║
║    • Interactive testing ("Try it out" button)                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## 🎨 Design Features

### Professional Styling

**Color Scheme:**
- Primary: Deep Blue (#4e73df)
- Success: Green (#1cc88a)
- Warning: Orange (#f6c23e)
- Danger: Red (#e74a3b)
- Background: Gradient from #667eea to #764ba2

**Typography:**
- Headers: System font stack (San Francisco, Roboto, etc.)
- Body: -apple-system, BlinkMacSystemFont
- Code: Monaco, Consolas, monospace

**Components:**
- Bootstrap 5 cards with shadows
- Smooth fade-in animations
- Hover effects on buttons
- Loading spinners for async operations
- Toast notifications for success/error

---

## 🔧 Technical Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI (Python 3.9+) |
| **Frontend** | HTMX + Bootstrap 5 |
| **Visualization** | Plotly.js |
| **Icons** | Bootstrap Icons |
| **Charts** | Plotly (interactive) |
| **API Docs** | FastAPI Swagger UI |
| **Server** | Uvicorn (ASGI) |

---

## 📱 Responsive Design

**Desktop (>1200px):**
- Full 3-column layout
- Expanded navigation
- Large charts and graphs

**Tablet (768px - 1200px):**
- 2-column layout
- Collapsible navigation
- Medium-sized visualizations

**Mobile (<768px):**
- Single-column layout
- Hamburger menu
- Touch-optimized charts

---

## ⚡ Performance Features

- **Lazy Loading**: Components load on demand
- **HTMX**: Dynamic updates without full page reload
- **Caching**: API responses cached in browser
- **Async Processing**: Non-blocking API calls
- **Progressive Enhancement**: Works without JavaScript (basic features)

---

## 🎯 User Workflow

### Typical User Journey

1. **Start Server**
   ```bash
   python3 run.py
   ```

2. **Open Dashboard** (`http://localhost:8000`)
   - View data summary cards
   - See dataset statistics

3. **Compare Models**
   - Click "Compare All Models"
   - View ROC curves, metrics
   - Analyze performance differences

4. **Explore Analyses**
   - Navigate between tabs
   - Interact with visualizations
   - Export results

5. **Upload New Data** (Optional)
   - Go to `/upload`
   - Drag & drop CSV files
   - Validate and process

6. **API Testing** (Advanced)
   - Visit `/docs`
   - Test endpoints interactively
   - Integrate with other tools

---

## 💡 Tips & Tricks

### For Best Experience:

1. **Use Chrome or Firefox** for best Plotly support
2. **Allow pop-ups** for export features
3. **Enable JavaScript** for interactive features
4. **Use wider screens** for better visualization
5. **Check console** for debugging (F12)

### Keyboard Shortcuts:

- `Ctrl+R` - Refresh page
- `Ctrl+Shift+I` - Open developer tools
- `Ctrl+K` - Focus search (in API docs)
- `Esc` - Close modals

### Hidden Features:

- **Double-click** charts to reset zoom
- **Shift+Drag** on charts for box select
- **Right-click** charts for export menu
- **Hover** on metrics for detailed tooltips

---

## 🐛 Troubleshooting

### Common Issues:

**Server won't start:**
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill process if needed
kill -9 <PID>

# Try different port
uvicorn app.main:app --port 8001
```

**Charts not loading:**
- Check browser console (F12)
- Verify Plotly.js loaded
- Clear browser cache
- Try incognito mode

**Upload fails:**
- Verify CSV format matches expected schema
- Check file sizes (< 50MB recommended)
- Ensure correct column names
- View server logs for details

**LLM features not working:**
- Check `.env` file has `OPENAI_API_KEY`
- Verify API key is valid
- Check quota limits
- LLM features are optional - system works without them

---

## 📚 Additional Resources

- **PLAYBOOK.md** - Detailed user guide
- **README.md** - Project overview
- **research_paper_v1.md** - Research documentation
- **API Docs** - http://localhost:8000/docs

---

## 🎉 Summary

**Your Web Interface Includes:**

✅ Professional dashboard with modern design  
✅ Interactive visualizations (UMAP, ROC curves, networks)  
✅ Multi-model performance comparison  
✅ Drag-and-drop data upload  
✅ Real-time API documentation  
✅ Tab-based analysis navigation  
✅ Mobile-responsive layout  
✅ LLM-powered insights (optional)  
✅ Export capabilities  
✅ Comprehensive statistics  

**Ready to use right now - just run `python3 run.py`!** 🚀

