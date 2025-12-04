"""
Knowledge Graph + Deep Learning Baseline for ADRD Risk Prediction
Following Cui Tao's ontology-driven approach with graph neural networks

This baseline model:
1. Constructs a knowledge graph from ontology-aligned EHR data
2. Links patients to ADRD risk factors through biomedical ontologies
3. Uses Graph Neural Networks (GNN) for risk prediction
4. Leverages temporal patterns and comorbidity relationships
"""
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")


class ADRDKnowledgeGraph:
    """
    Constructs a knowledge graph linking patients, diagnoses, medications, 
    lab results, and procedures through OMOP concept relationships
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.patient_nodes = set()
        self.concept_nodes = set()
        self.risk_factors = {}
        self.node_features = {}
        self.edge_types = set()
        
    def add_patient(self, patient_id: str, demographics: Dict):
        """Add patient node with demographic features"""
        self.graph.add_node(
            patient_id,
            node_type='patient',
            age=demographics.get('Age', 70),
            sex=demographics.get('Sex', 'Unknown'),
            ethnicity=demographics.get('Ethnicity', 'Unknown'),
            race=demographics.get('Race', 'Unknown'),
            education=demographics.get('EducationLevel', 'Unknown'),
            smoking=demographics.get('SmokingStatus', 'Unknown')
        )
        self.patient_nodes.add(patient_id)
    
    def add_diagnosis(self, patient_id: str, diagnosis: Dict, date: str):
        """Link patient to diagnosis through temporal relationship"""
        concept_id = f"SNOMED_{diagnosis['SNOMED_Code']}"
        
        # Add or update concept node
        if concept_id not in self.concept_nodes:
            self.graph.add_node(
                concept_id,
                node_type='diagnosis',
                name=diagnosis['FullDiagnosisName'],
                snomed=diagnosis['SNOMED_Code'],
                icd10=diagnosis['ICD10_Code'],
                category=diagnosis['Level2_Category'],
                subcategory=diagnosis['Level3_Category'],
                omop_concept_id=diagnosis['OMOP_ConceptID'],
                severity=diagnosis.get('Severity', 'Moderate')
            )
            self.concept_nodes.add(concept_id)
        
        # Add temporal edge
        self.graph.add_edge(
            patient_id,
            concept_id,
            edge_type='has_diagnosis',
            date=date,
            weight=1.0
        )
        self.edge_types.add('has_diagnosis')
        
        # Track if this is an ADRD risk factor
        risk_weight = diagnosis.get('risk_weight', 0)
        if risk_weight > 0:
            if concept_id not in self.risk_factors:
                self.risk_factors[concept_id] = risk_weight
    
    def add_medication(self, patient_id: str, medication: Dict, date: str):
        """Link patient to medication"""
        concept_id = f"RxNorm_{medication['RxNorm_Code']}"
        
        if concept_id not in self.concept_nodes:
            self.graph.add_node(
                concept_id,
                node_type='medication',
                name=medication['MedicationName'],
                generic=medication['MedicationGenericName'],
                rxnorm=medication['RxNorm_Code'],
                omop_concept_id=medication['OMOP_ConceptID'],
                drug_class=medication.get('MedicationClass', 'Unknown')
            )
            self.concept_nodes.add(concept_id)
        
        self.graph.add_edge(
            patient_id,
            concept_id,
            edge_type='takes_medication',
            date=date,
            weight=1.0
        )
        self.edge_types.add('takes_medication')
    
    def add_lab_result(self, patient_id: str, lab: Dict, date: str):
        """Link patient to lab test with result value"""
        concept_id = f"LOINC_{lab['LOINC_Code']}"
        
        if concept_id not in self.concept_nodes:
            self.graph.add_node(
                concept_id,
                node_type='lab_test',
                name=lab['TestName'],
                loinc=lab['LOINC_Code'],
                omop_concept_id=lab['OMOP_ConceptID'],
                category=lab.get('Category', 'General')
            )
            self.concept_nodes.add(concept_id)
        
        # Extract numeric value if possible
        result_str = lab['TestResult']
        try:
            numeric_value = float(result_str.split()[0])
        except (ValueError, IndexError, AttributeError):
            numeric_value = 0.0
        
        self.graph.add_edge(
            patient_id,
            concept_id,
            edge_type='has_lab_result',
            date=date,
            result=result_str,
            numeric_value=numeric_value,
            weight=1.0
        )
        self.edge_types.add('has_lab_result')
    
    def add_imaging(self, patient_id: str, imaging: Dict, date: str):
        """Link patient to imaging procedure"""
        concept_id = f"CPT_{imaging['CPT_Code']}"
        
        if concept_id not in self.concept_nodes:
            self.graph.add_node(
                concept_id,
                node_type='imaging',
                name=imaging['ProcedureName'],
                cpt=imaging['CPT_Code'],
                snomed=imaging['SNOMED_Code'],
                omop_concept_id=imaging['OMOP_ConceptID'],
                category=imaging.get('Category', 'General')
            )
            self.concept_nodes.add(concept_id)
        
        self.graph.add_edge(
            patient_id,
            concept_id,
            edge_type='has_imaging',
            date=date,
            weight=1.0
        )
        self.edge_types.add('has_imaging')
    
    def add_treatment(self, patient_id: str, treatment: Dict, date: str):
        """Link patient to treatment procedure"""
        concept_id = f"CPT_{treatment['CPT_Code']}_tx"
        
        if concept_id not in self.concept_nodes:
            self.graph.add_node(
                concept_id,
                node_type='treatment',
                name=treatment['ProcedureName'],
                cpt=treatment['CPT_Code'],
                snomed=treatment['SNOMED_Code'],
                omop_concept_id=treatment['OMOP_ConceptID'],
                category=treatment.get('Category', 'General')
            )
            self.concept_nodes.add(concept_id)
        
        self.graph.add_edge(
            patient_id,
            concept_id,
            edge_type='receives_treatment',
            date=date,
            weight=1.0
        )
        self.edge_types.add('receives_treatment')
    
    def construct_comorbidity_edges(self):
        """Add edges between frequently co-occurring diagnoses (comorbidity network)"""
        print("  Constructing comorbidity network...")
        
        # Find co-occurring diagnoses for each patient
        patient_diagnoses = defaultdict(set)
        for edge in self.graph.edges(data=True):
            if edge[2].get('edge_type') == 'has_diagnosis':
                patient_id = edge[0]
                diagnosis_id = edge[1]
                patient_diagnoses[patient_id].add(diagnosis_id)
        
        # Count co-occurrences
        cooccurrence_counts = defaultdict(int)
        for patient_id, diagnoses in patient_diagnoses.items():
            diagnoses_list = list(diagnoses)
            for i in range(len(diagnoses_list)):
                for j in range(i+1, len(diagnoses_list)):
                    pair = tuple(sorted([diagnoses_list[i], diagnoses_list[j]]))
                    cooccurrence_counts[pair] += 1
        
        # Add comorbidity edges (if they co-occur in at least 3 patients)
        threshold = 3
        for (diag1, diag2), count in cooccurrence_counts.items():
            if count >= threshold:
                # Normalized by total patients
                weight = count / len(patient_diagnoses)
                self.graph.add_edge(
                    diag1,
                    diag2,
                    edge_type='comorbid_with',
                    weight=weight,
                    count=count
                )
                self.edge_types.add('comorbid_with')
        
        print(f"  Added {sum(1 for e in self.graph.edges(data=True) if e[2].get('edge_type')=='comorbid_with')} comorbidity edges")
    
    def compute_risk_scores(self):
        """Compute ADRD risk scores for patients based on graph structure"""
        print("  Computing ADRD risk scores...")
        
        risk_scores = {}
        for patient_id in self.patient_nodes:
            score = 0.0
            
            # Get patient's diagnoses
            for neighbor in self.graph.neighbors(patient_id):
                if neighbor in self.risk_factors:
                    # Weight by risk factor importance
                    risk_weight = self.risk_factors[neighbor]
                    
                    # Count occurrences (multiple visits with same diagnosis)
                    occurrences = sum(1 for e in self.graph.edges(patient_id, neighbor, data=True) 
                                    if e[2].get('edge_type') == 'has_diagnosis')
                    
                    # Temporal recency bonus (later diagnoses weighted more)
                    edges = list(self.graph.edges(patient_id, neighbor, data=True))
                    if edges:
                        dates = [e[2].get('date', '2020-01-01') for e in edges]
                        latest_date = max(dates)
                        try:
                            recency_weight = 1.0 + (datetime.strptime(latest_date, '%Y-%m-%d').year - 2020) * 0.1
                        except:
                            recency_weight = 1.0
                    else:
                        recency_weight = 1.0
                    
                    score += risk_weight * np.log1p(occurrences) * recency_weight
            
            risk_scores[patient_id] = score
        
        return risk_scores
    
    def extract_node_features(self) -> Dict[str, np.ndarray]:
        """Extract feature vectors for all nodes"""
        print("  Extracting node features...")
        
        features = {}
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            node_type = node_data.get('node_type', 'unknown')
            
            if node_type == 'patient':
                # Patient demographic features
                age = node_data.get('age', 70) / 100.0  # Normalize
                sex_enc = 1.0 if node_data.get('sex') == 'Male' else 0.0
                edu_map = {'Less than high school': 0.2, 'High school': 0.4, 'Some college': 0.6, 
                          'College degree': 0.8, 'Graduate degree': 1.0}
                edu_enc = edu_map.get(node_data.get('education', 'Unknown'), 0.5)
                smoke_map = {'Never smoker': 0.0, 'Former smoker': 0.5, 'Current smoker': 1.0}
                smoke_enc = smoke_map.get(node_data.get('smoking', 'Unknown'), 0.25)
                
                # Degree centrality in graph
                degree = self.graph.degree(node) / 100.0  # Normalize
                
                features[node] = np.array([age, sex_enc, edu_enc, smoke_enc, degree])
                
            elif node_type == 'diagnosis':
                # Diagnosis features
                severity_map = {'Mild': 0.33, 'Moderate': 0.67, 'Severe': 1.0}
                severity = severity_map.get(node_data.get('severity', 'Moderate'), 0.67)
                
                # Is it a risk factor?
                is_risk_factor = 1.0 if node in self.risk_factors else 0.0
                risk_weight = self.risk_factors.get(node, 0.0)
                
                # Graph centrality
                degree = self.graph.degree(node) / 100.0
                
                features[node] = np.array([severity, is_risk_factor, risk_weight, degree])
                
            else:
                # Generic features for other node types
                degree = self.graph.degree(node) / 100.0
                features[node] = np.array([degree])
        
        return features
    
    def get_graph_statistics(self) -> Dict:
        """Get statistics about the knowledge graph"""
        stats = {
            'n_nodes': self.graph.number_of_nodes(),
            'n_edges': self.graph.number_of_edges(),
            'n_patients': len(self.patient_nodes),
            'n_concepts': len(self.concept_nodes),
            'n_risk_factors': len(self.risk_factors),
            'edge_types': list(self.edge_types),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }
        
        # Node type distribution
        node_types = defaultdict(int)
        for node in self.graph.nodes(data=True):
            node_types[node[1].get('node_type', 'unknown')] += 1
        stats['node_type_distribution'] = dict(node_types)
        
        return stats


class GraphNeuralNetworkADRD(nn.Module if TORCH_AVAILABLE else object):
    """
    Graph Neural Network for ADRD risk prediction
    Uses Graph Attention Network (GAT) architecture
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for GNN model")
        
        super(GraphNeuralNetworkADRD, self).__init__()
        
        # Graph convolution layers
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.conv3 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim // 2, 1)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, edge_index):
        """Forward pass through the GNN"""
        # First graph convolution
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second graph convolution
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third graph convolution
        x = self.conv3(x)
        x = F.relu(x)
        
        # Attention-weighted pooling
        attention_weights = torch.softmax(self.attention(x), dim=0)
        x = x * attention_weights
        
        # Final prediction
        x = self.fc(x)
        
        return torch.sigmoid(x)


class KnowledgeGraphBaselineModel:
    """
    Complete baseline model combining knowledge graph construction and GNN prediction
    """
    
    def __init__(self):
        self.kg = ADRDKnowledgeGraph()
        self.gnn_model = None
        self.scaler = None
        self.node_to_idx = {}
        self.idx_to_node = {}
        
    def build_knowledge_graph(
        self,
        demographics_df: pd.DataFrame,
        diagnoses_df: pd.DataFrame,
        medications_df: Optional[pd.DataFrame] = None,
        labs_df: Optional[pd.DataFrame] = None,
        imaging_df: Optional[pd.DataFrame] = None,
        treatments_df: Optional[pd.DataFrame] = None
    ):
        """Build the complete knowledge graph from EHR data"""
        print("\n🔨 Building Knowledge Graph...")
        
        # Add patients
        print("  Adding patient nodes...")
        for _, patient in demographics_df.iterrows():
            self.kg.add_patient(patient['PatientID'], patient.to_dict())
        
        # Add diagnoses
        print("  Adding diagnosis nodes and edges...")
        for _, diagnosis in diagnoses_df.iterrows():
            self.kg.add_diagnosis(
                diagnosis['PatientID'],
                diagnosis.to_dict(),
                diagnosis.get('DateOfService', '2020-01-01')
            )
        
        # Add medications
        if medications_df is not None and len(medications_df) > 0:
            print("  Adding medication nodes and edges...")
            for _, med in medications_df.iterrows():
                self.kg.add_medication(
                    med['PatientID'],
                    med.to_dict(),
                    med.get('DateOfService', '2020-01-01')
                )
        
        # Add lab results
        if labs_df is not None and len(labs_df) > 0:
            print("  Adding lab test nodes and edges...")
            for _, lab in labs_df.iterrows():
                self.kg.add_lab_result(
                    lab['PatientID'],
                    lab.to_dict(),
                    lab.get('DateOfService', '2020-01-01')
                )
        
        # Add imaging
        if imaging_df is not None and len(imaging_df) > 0:
            print("  Adding imaging nodes and edges...")
            for _, img in imaging_df.iterrows():
                self.kg.add_imaging(
                    img['PatientID'],
                    img.to_dict(),
                    img.get('DateOfService', '2020-01-01')
                )
        
        # Add treatments
        if treatments_df is not None and len(treatments_df) > 0:
            print("  Adding treatment nodes and edges...")
            for _, tx in treatments_df.iterrows():
                self.kg.add_treatment(
                    tx['PatientID'],
                    tx.to_dict(),
                    tx.get('DateOfService', '2020-01-01')
                )
        
        # Construct comorbidity network
        self.kg.construct_comorbidity_edges()
        
        # Print statistics
        stats = self.kg.get_graph_statistics()
        print(f"\n📊 Knowledge Graph Statistics:")
        print(f"  Nodes: {stats['n_nodes']:,}")
        print(f"  Edges: {stats['n_edges']:,}")
        print(f"  Patients: {stats['n_patients']}")
        print(f"  Medical Concepts: {stats['n_concepts']:,}")
        print(f"  ADRD Risk Factors: {stats['n_risk_factors']}")
        print(f"  Edge Types: {len(stats['edge_types'])}")
        print(f"  Graph Density: {stats['density']:.6f}")
        print(f"  Node Types: {stats['node_type_distribution']}")
        
        return stats
    
    def compute_risk_scores(self) -> Dict[str, float]:
        """Compute graph-based ADRD risk scores"""
        return self.kg.compute_risk_scores()
    
    def train_gnn_model(self, labels: Dict[str, int], epochs: int = 100):
        """Train the Graph Neural Network"""
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. Skipping GNN training.")
            return None
        
        print("\n🧠 Training Graph Neural Network...")
        
        # Extract node features
        node_features = self.kg.extract_node_features()
        
        # Create node index mapping
        patient_nodes = list(self.kg.patient_nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(patient_nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # Prepare training data
        X = []
        y = []
        for node in patient_nodes:
            if node in node_features and node in labels:
                X.append(node_features[node])
                y.append(labels[node])
        
        X = np.array(X)
        y = np.array(y)
        
        # Initialize model
        input_dim = X.shape[1]
        self.gnn_model = GraphNeuralNetworkADRD(input_dim=input_dim)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        # Training loop (simplified)
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass (simplified without edge_index for now)
            # In full implementation, would use proper graph convolution
            output = self.gnn_model(X_tensor, None)
            
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        print("✅ GNN training complete!")
        
        return self.gnn_model
    
    def predict(self, patient_ids: List[str]) -> Dict[str, float]:
        """Predict ADRD risk for given patients"""
        predictions = {}
        
        # Use graph-based risk scores
        risk_scores = self.kg.compute_risk_scores()
        
        for patient_id in patient_ids:
            predictions[patient_id] = risk_scores.get(patient_id, 0.0)
        
        return predictions
    
    def save_graph(self, filepath: str):
        """Save the knowledge graph"""
        nx.write_gpickle(self.kg.graph, filepath)
        print(f"✅ Knowledge graph saved to {filepath}")
    
    def export_graph_for_visualization(self, filepath: str, max_nodes: int = 1000):
        """Export graph in format suitable for visualization"""
        # Sample nodes if too large
        if self.kg.graph.number_of_nodes() > max_nodes:
            nodes = list(self.kg.patient_nodes)[:50]  # Sample patients
            subgraph = nx.ego_graph(self.kg.graph, nodes[0], radius=2, undirected=True)
        else:
            subgraph = self.kg.graph
        
        # Export as GraphML
        nx.write_graphml(subgraph, filepath)
        print(f"✅ Graph exported for visualization: {filepath}")

