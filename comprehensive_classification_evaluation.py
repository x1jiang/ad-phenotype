"""
Comprehensive Classification Evaluation
Calculates AUC, Precision, Recall, F1, Accuracy, Specificity, Sensitivity
for all three models with proper train/test splits
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score, 
    f1_score, accuracy_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import time
import warnings
warnings.filterwarnings('ignore')

from app.services.data_loader import DataLoader
from app.services.knowledge_graph_baseline import KnowledgeGraphBaselineModel
from app.services.enhanced_phenotype_model import EnhancedPhenotypeModel
from app.services.llm_phenotype_service import LLMPhenotypeService

print("="*100)
print(" "*25 + "COMPREHENSIVE CLASSIFICATION EVALUATION")
print(" "*20 + "AUC, Precision, Recall, F1, Accuracy, Specificity")
print("="*100)
print()

# Load data
print("📂 Loading comprehensive data...")
loader = DataLoader()
ad_data = loader.load_all_data('ad')
control_data = loader.load_all_data('control')

all_demographics = pd.concat([ad_data['demographics'], control_data['demographics']])
all_diagnoses = pd.concat([ad_data['diagnosis'], control_data['diagnosis']])
all_medications = pd.concat([ad_data['medications'], control_data['medications']])
all_labs = pd.concat([ad_data['labs'], control_data['labs']])
all_imaging = pd.concat([ad_data['imaging'], control_data['imaging']])
all_treatments = pd.concat([ad_data['treatments'], control_data['treatments']])

ad_patient_ids = set(ad_data['demographics']['PatientID'].values)
control_patient_ids = set(control_data['demographics']['PatientID'].values)
all_patient_ids = list(ad_patient_ids) + list(control_patient_ids)
labels = np.array([1] * len(ad_patient_ids) + [0] * len(control_patient_ids))

print(f"✅ Loaded: {len(all_patient_ids)} patients, {len(all_diagnoses):,} diagnoses")
print()

def calculate_metrics(y_true, y_pred, y_pred_proba, model_name):
    """Calculate comprehensive classification metrics"""
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y_true, y_pred_proba),
        'AP': average_precision_score(y_true, y_pred_proba),
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }
    
    return metrics

def evaluate_classifier(X, y, classifier_name='Logistic Regression'):
    """Train and evaluate a classifier with proper train/test split"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Initialize classifier
    if classifier_name == 'Logistic Regression':
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif classifier_name == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier_name == 'Gradient Boosting':
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=42)
    
    # Train
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, classifier_name)
    
    # Cross-validation AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc').mean()
    metrics['CV-AUC'] = cv_auc
    
    return metrics

# ============================================================================
# MODEL 1: KNOWLEDGE GRAPH BASELINE
# ============================================================================

print("="*100)
print("MODEL 1: KNOWLEDGE GRAPH BASELINE (Deep Learning + Ontology)")
print("="*100)

start_time = time.time()

# Build knowledge graph
kg_model = KnowledgeGraphBaselineModel()
kg_stats = kg_model.build_knowledge_graph(
    demographics_df=all_demographics,
    diagnoses_df=all_diagnoses,
    medications_df=all_medications,
    labs_df=all_labs,
    imaging_df=all_imaging,
    treatments_df=all_treatments
)

# Compute risk scores
risk_scores = kg_model.compute_risk_scores()

# Extract features
graph_features = []
for pid in all_patient_ids:
    degree = kg_model.kg.graph.degree(pid) if pid in kg_model.kg.graph else 0
    neighbors = list(kg_model.kg.graph.neighbors(pid)) if pid in kg_model.kg.graph else []
    
    n_diagnoses = sum(1 for n in neighbors if kg_model.kg.graph.nodes[n].get('node_type') == 'diagnosis')
    n_meds = sum(1 for n in neighbors if kg_model.kg.graph.nodes[n].get('node_type') == 'medication')
    n_labs = sum(1 for n in neighbors if kg_model.kg.graph.nodes[n].get('node_type') == 'lab_test')
    n_imaging = sum(1 for n in neighbors if kg_model.kg.graph.nodes[n].get('node_type') == 'imaging')
    
    risk_score = risk_scores.get(pid, 0.0)
    
    graph_features.append([
        degree, n_diagnoses, n_meds, n_labs, n_imaging, risk_score
    ])

X_kg = np.array(graph_features)
scaler_kg = StandardScaler()
X_kg_scaled = scaler_kg.fit_transform(X_kg)

kg_time = time.time() - start_time

print(f"\n📊 Features extracted: {X_kg.shape[1]} features from knowledge graph")
print(f"⏱️  Total time: {kg_time:.2f}s")

# Evaluate with multiple classifiers
print("\n🎯 Classification Results:\n")

kg_results = []
for clf_name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting']:
    metrics = evaluate_classifier(X_kg_scaled, labels, clf_name)
    kg_results.append(metrics)
    print(f"{clf_name:25s} | AUC: {metrics['AUC-ROC']:.4f} | F1: {metrics['F1-Score']:.4f} | "
          f"Acc: {metrics['Accuracy']:.4f} | Prec: {metrics['Precision']:.4f} | Rec: {metrics['Recall']:.4f}")

# Select best classifier
best_kg = max(kg_results, key=lambda x: x['AUC-ROC'])
print(f"\n🏆 Best KG Classifier: {best_kg['Model']} (AUC: {best_kg['AUC-ROC']:.4f})")

# ============================================================================
# MODEL 2: ENHANCED MODEL
# ============================================================================

print("\n" + "="*100)
print("MODEL 2: ENHANCED MODEL (Advanced Feature Engineering)")
print("="*100)

start_time = time.time()

enhanced_model = EnhancedPhenotypeModel()

ad_features_enhanced = enhanced_model.create_feature_matrix(
    ad_data['diagnosis'], 
    ad_data['medications'], 
    ad_data['labs']
)
control_features_enhanced = enhanced_model.create_feature_matrix(
    control_data['diagnosis'],
    control_data['medications'],
    control_data['labs']
)

all_features_enhanced = pd.concat([ad_features_enhanced, control_features_enhanced])
feature_cols = [col for col in all_features_enhanced.columns if col != 'PatientID']
X_enhanced = all_features_enhanced[feature_cols].fillna(0).values

scaler_enhanced = StandardScaler()
X_enhanced_scaled = scaler_enhanced.fit_transform(X_enhanced)

enhanced_time = time.time() - start_time

print(f"\n📊 Features extracted: {X_enhanced.shape[1]} advanced features")
print(f"⏱️  Total time: {enhanced_time:.2f}s")

# Evaluate with multiple classifiers
print("\n🎯 Classification Results:\n")

enhanced_results = []
for clf_name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting']:
    metrics = evaluate_classifier(X_enhanced_scaled, labels, clf_name)
    enhanced_results.append(metrics)
    print(f"{clf_name:25s} | AUC: {metrics['AUC-ROC']:.4f} | F1: {metrics['F1-Score']:.4f} | "
          f"Acc: {metrics['Accuracy']:.4f} | Prec: {metrics['Precision']:.4f} | Rec: {metrics['Recall']:.4f}")

# Select best classifier
best_enhanced = max(enhanced_results, key=lambda x: x['AUC-ROC'])
print(f"\n🏆 Best Enhanced Classifier: {best_enhanced['Model']} (AUC: {best_enhanced['AUC-ROC']:.4f})")

# ============================================================================
# MODEL 3: LLM MODEL (Sample)
# ============================================================================

print("\n" + "="*100)
print("MODEL 3: LLM MODEL (GPT-5.1 Powered) - Using sample for demonstration")
print("="*100)

start_time = time.time()

llm_service = LLMPhenotypeService()

# Use sample for LLM
sample_size = 50
ad_sample_ids = list(ad_patient_ids)[:sample_size]
control_sample_ids = list(control_patient_ids)[:sample_size]

ad_diag_sample = ad_data['diagnosis'][ad_data['diagnosis']['PatientID'].isin(ad_sample_ids)]
control_diag_sample = control_data['diagnosis'][control_data['diagnosis']['PatientID'].isin(control_sample_ids)]
ad_meds_sample = ad_data['medications'][ad_data['medications']['PatientID'].isin(ad_sample_ids)]
control_meds_sample = control_data['medications'][control_data['medications']['PatientID'].isin(control_sample_ids)]
ad_labs_sample = ad_data['labs'][ad_data['labs']['PatientID'].isin(ad_sample_ids)]
control_labs_sample = control_data['labs'][control_data['labs']['PatientID'].isin(control_sample_ids)]

# Get enhanced features for sample
ad_enhanced_sample = enhanced_model.create_feature_matrix(ad_diag_sample, ad_meds_sample, ad_labs_sample)
control_enhanced_sample = enhanced_model.create_feature_matrix(control_diag_sample, control_meds_sample, control_labs_sample)

# Extract LLM features
print("  Extracting semantic features with GPT-5.1...")
ad_llm_features = llm_service.extract_semantic_features(ad_diag_sample)
control_llm_features = llm_service.extract_semantic_features(control_diag_sample)

# Merge
ad_combined = ad_enhanced_sample.merge(ad_llm_features, on='PatientID', how='left')
control_combined = control_enhanced_sample.merge(control_llm_features, on='PatientID', how='left')

all_features_llm = pd.concat([ad_combined, control_combined])
labels_sample = np.array([1] * len(ad_combined) + [0] * len(control_combined))

feature_cols_llm = [col for col in all_features_llm.columns 
                    if col != 'PatientID' and col != 'llm_primary_category' and col != 'llm_themes']
X_llm = all_features_llm[feature_cols_llm].fillna(0).values

scaler_llm = StandardScaler()
X_llm_scaled = scaler_llm.fit_transform(X_llm)

llm_time = time.time() - start_time

print(f"\n📊 Features extracted: {X_llm.shape[1]} features (enhanced + LLM)")
print(f"⏱️  Total time: {llm_time:.2f}s")

# Evaluate with multiple classifiers
print("\n🎯 Classification Results:\n")

llm_results = []
for clf_name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting']:
    metrics = evaluate_classifier(X_llm_scaled, labels_sample, clf_name)
    llm_results.append(metrics)
    print(f"{clf_name:25s} | AUC: {metrics['AUC-ROC']:.4f} | F1: {metrics['F1-Score']:.4f} | "
          f"Acc: {metrics['Accuracy']:.4f} | Prec: {metrics['Precision']:.4f} | Rec: {metrics['Recall']:.4f}")

# Select best classifier
best_llm = max(llm_results, key=lambda x: x['AUC-ROC'])
print(f"\n🏆 Best LLM Classifier: {best_llm['Model']} (AUC: {best_llm['AUC-ROC']:.4f})")

# ============================================================================
# FINAL COMPARISON
# ============================================================================

print("\n" + "="*100)
print("📊 COMPREHENSIVE PERFORMANCE COMPARISON")
print("="*100)
print()

# Create comparison table
print(f"{'Model':<30} {'Classifier':<20} {'AUC':<8} {'F1':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'Spec':<8}")
print("-" * 100)

print(f"{'Knowledge Graph':<30} {best_kg['Model']:<20} {best_kg['AUC-ROC']:<8.4f} {best_kg['F1-Score']:<8.4f} "
      f"{best_kg['Accuracy']:<8.4f} {best_kg['Precision']:<8.4f} {best_kg['Recall']:<8.4f} {best_kg['Specificity']:<8.4f}")

print(f"{'Enhanced':<30} {best_enhanced['Model']:<20} {best_enhanced['AUC-ROC']:<8.4f} {best_enhanced['F1-Score']:<8.4f} "
      f"{best_enhanced['Accuracy']:<8.4f} {best_enhanced['Precision']:<8.4f} {best_enhanced['Recall']:<8.4f} {best_enhanced['Specificity']:<8.4f}")

print(f"{'LLM (GPT-5.1) [sample]':<30} {best_llm['Model']:<20} {best_llm['AUC-ROC']:<8.4f} {best_llm['F1-Score']:<8.4f} "
      f"{best_llm['Accuracy']:<8.4f} {best_llm['Precision']:<8.4f} {best_llm['Recall']:<8.4f} {best_llm['Specificity']:<8.4f}")

print()

# Determine overall winner
all_best = [
    ('Knowledge Graph', best_kg),
    ('Enhanced', best_enhanced),
    ('LLM (GPT-5.1)', best_llm)
]

winner = max(all_best, key=lambda x: x[1]['AUC-ROC'])

print("🏆 OVERALL WINNER:")
print(f"  Model: {winner[0]}")
print(f"  Classifier: {winner[1]['Model']}")
print(f"  AUC-ROC: {winner[1]['AUC-ROC']:.4f} ⭐")
print(f"  F1-Score: {winner[1]['F1-Score']:.4f}")
print(f"  Accuracy: {winner[1]['Accuracy']:.4f}")
print(f"  Precision: {winner[1]['Precision']:.4f}")
print(f"  Recall/Sensitivity: {winner[1]['Recall']:.4f}")
print(f"  Specificity: {winner[1]['Specificity']:.4f}")
print()

# Detailed confusion matrix for winner
print(f"📊 Confusion Matrix ({winner[0]} - {winner[1]['Model']}):")
print(f"  True Positives (TP):  {winner[1]['TP']}")
print(f"  True Negatives (TN):  {winner[1]['TN']}")
print(f"  False Positives (FP): {winner[1]['FP']}")
print(f"  False Negatives (FN): {winner[1]['FN']}")
print()

# Performance summary
print("="*100)
print("📈 KEY INSIGHTS")
print("="*100)
print()

print(f"1. Knowledge Graph Baseline:")
print(f"   • AUC-ROC: {best_kg['AUC-ROC']:.4f}")
print(f"   • Best for: Ontology-driven analysis")
print(f"   • Features: {X_kg.shape[1]} (graph topology + risk scores)")
print()

print(f"2. Enhanced Model:")
print(f"   • AUC-ROC: {best_enhanced['AUC-ROC']:.4f}")
print(f"   • Best for: Traditional ML with engineered features")
print(f"   • Features: {X_enhanced.shape[1]} (temporal, network, trajectory)")
print()

print(f"3. LLM Model (GPT-5.1):")
print(f"   • AUC-ROC: {best_llm['AUC-ROC']:.4f}")
print(f"   • Best for: Semantic understanding and AI insights")
print(f"   • Features: {X_llm.shape[1]} (enhanced + LLM semantic)")
print()

# Recommendations
print("="*100)
print("🎯 RECOMMENDATIONS")
print("="*100)
print()

if winner[0] == 'Knowledge Graph':
    print("✅ RECOMMENDED: Knowledge Graph Baseline")
    print("   • Highest AUC-ROC performance")
    print("   • Leverages biomedical ontology structure")
    print("   • Captures comorbidity networks")
    print("   • Interpretable through graph relationships")
elif winner[0] == 'Enhanced':
    print("✅ RECOMMENDED: Enhanced Model")
    print("   • Highest AUC-ROC performance")
    print("   • Sophisticated feature engineering")
    print("   • Fast and efficient")
    print("   • No API costs")
else:
    print("✅ RECOMMENDED: LLM Model (GPT-5.1)")
    print("   • Highest AUC-ROC performance")
    print("   • AI-powered semantic understanding")
    print("   • Clinical context awareness")
    print("   • Automated phenotype explanations")

print()
print("="*100)
print("✅ COMPREHENSIVE CLASSIFICATION EVALUATION COMPLETE!")
print("="*100)

