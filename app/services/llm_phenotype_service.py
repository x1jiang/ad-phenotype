"""
LLM-Enhanced Phenotype Service using GPT-5.1
Advanced AI-powered phenotype extraction and analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from openai import OpenAI
from app.config import settings
import json


class LLMPhenotypeService:
    """
    GPT-5.1 powered phenotype analysis service
    """
    
    def __init__(self):
        self.use_llm = settings.use_llm or (settings.openai_api_key is not None)
        self.model = settings.openai_model  # GPT-5.1
        self.client = None
        
        if self.use_llm and settings.openai_api_key:
            self.client = OpenAI(api_key=settings.openai_api_key)
    
    def _call_llm(self, prompt: str, response_format: str = "text") -> str:
        """Call GPT-5.1 API"""
        if not self.client:
            return "LLM not available. Please configure OpenAI API key."
        
        try:
            messages = [
                {"role": "system", "content": "You are an expert medical AI assistant specializing in Alzheimer's Disease phenotyping and EHR analysis."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent medical analysis
                max_completion_tokens=1000  # GPT-5.1 uses max_completion_tokens
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return f"Error: {str(e)}"
    
    def extract_semantic_features(self, diagnosis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract semantic features from diagnoses using GPT-5.1
        """
        if not self.client:
            print("LLM not available, using fallback method")
            return self._fallback_semantic_features(diagnosis_df)
        
        semantic_features = []
        
        # Group by patient
        for patient_id in diagnosis_df['PatientID'].unique():
            patient_diagnoses = diagnosis_df[diagnosis_df['PatientID'] == patient_id]
            diagnosis_list = patient_diagnoses['FullDiagnosisName'].unique().tolist()
            
            # Create prompt for GPT-5.1
            prompt = f"""Analyze the following patient diagnoses and extract key phenotypic features:

Diagnoses: {', '.join(diagnosis_list[:20])}  # Limit to first 20 to avoid token limits

Provide a structured analysis:
1. Primary disease category (e.g., Neurodegenerative, Cardiovascular, Metabolic)
2. Severity score (1-10)
3. Comorbidity complexity score (1-10)
4. Key clinical themes (list 3-5 keywords)

Format your response as JSON with keys: primary_category, severity, complexity, themes"""
            
            try:
                response = self._call_llm(prompt)
                
                # Parse JSON response
                try:
                    parsed = json.loads(response)
                    primary_category = parsed.get('primary_category', 'Unknown')
                    severity = float(parsed.get('severity', 5))
                    complexity = float(parsed.get('complexity', 5))
                    themes = parsed.get('themes', [])
                except:
                    # Fallback parsing
                    primary_category = 'Mixed'
                    severity = 5.0
                    complexity = 5.0
                    themes = []
                
                semantic_features.append({
                    'PatientID': patient_id,
                    'llm_primary_category': primary_category,
                    'llm_severity_score': severity,
                    'llm_complexity_score': complexity,
                    'llm_theme_count': len(themes),
                    'llm_themes': ','.join(themes) if themes else ''
                })
            except Exception as e:
                print(f"Error processing patient {patient_id}: {e}")
                semantic_features.append({
                    'PatientID': patient_id,
                    'llm_primary_category': 'Error',
                    'llm_severity_score': 0,
                    'llm_complexity_score': 0,
                    'llm_theme_count': 0,
                    'llm_themes': ''
                })
        
        return pd.DataFrame(semantic_features)
    
    def _fallback_semantic_features(self, diagnosis_df: pd.DataFrame) -> pd.DataFrame:
        """Fallback method when LLM is not available"""
        semantic_features = []
        
        for patient_id in diagnosis_df['PatientID'].unique():
            patient_diagnoses = diagnosis_df[diagnosis_df['PatientID'] == patient_id]
            
            # Simple rule-based categorization
            categories = patient_diagnoses['Level2_Category'].value_counts()
            primary_category = categories.index[0] if len(categories) > 0 else 'Unknown'
            
            # Simple severity based on diagnosis count
            severity = min(10, len(patient_diagnoses) / 5)
            complexity = min(10, patient_diagnoses['FullDiagnosisName'].nunique() / 3)
            
            semantic_features.append({
                'PatientID': patient_id,
                'llm_primary_category': primary_category,
                'llm_severity_score': severity,
                'llm_complexity_score': complexity,
                'llm_theme_count': 0,
                'llm_themes': ''
            })
        
        return pd.DataFrame(semantic_features)
    
    def explain_phenotype(self, phenotype_name: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate detailed phenotype explanation using GPT-5.1
        """
        if not self.client:
            return self._fallback_explanation(phenotype_name)
        
        context_str = ""
        if context:
            context_str = f"\n\nAdditional context:\n{json.dumps(context, indent=2)}"
        
        prompt = f"""Provide a comprehensive analysis of the phenotype: "{phenotype_name}"

Focus on:
1. Clinical significance in Alzheimer's Disease context
2. Typical presentation and symptoms
3. Association with AD progression
4. Severity classification (Mild/Moderate/Severe)
5. Common comorbidities
6. Treatment implications
7. Research insights

{context_str}

Provide a detailed but concise explanation suitable for medical researchers."""
        
        explanation = self._call_llm(prompt)
        
        # Determine severity from explanation
        severity = 'Moderate'
        if any(word in explanation.lower() for word in ['severe', 'critical', 'advanced']):
            severity = 'Severe'
        elif any(word in explanation.lower() for word in ['mild', 'early', 'minimal']):
            severity = 'Mild'
        
        return {
            'phenotype': phenotype_name,
            'explanation': explanation,
            'severity': severity,
            'source': 'GPT-5.1',
            'model': self.model
        }
    
    def _fallback_explanation(self, phenotype_name: str) -> Dict[str, Any]:
        """Fallback explanation when LLM is not available"""
        return {
            'phenotype': phenotype_name,
            'explanation': f"Rule-based analysis: {phenotype_name} is a clinical condition that may be associated with Alzheimer's Disease. LLM analysis not available.",
            'severity': 'Moderate',
            'source': 'Rule-based',
            'model': 'Fallback'
        }
    
    def batch_explain_phenotypes(self, phenotypes: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Batch explain multiple phenotypes
        """
        explanations = []
        
        for phenotype in phenotypes[:limit]:
            explanation = self.explain_phenotype(phenotype)
            explanations.append(explanation)
        
        return explanations
    
    def compare_cohorts(self, ad_diagnoses: pd.DataFrame, control_diagnoses: pd.DataFrame) -> Dict[str, Any]:
        """
        Use GPT-5.1 to compare AD vs Control cohorts
        """
        if not self.client:
            return {'comparison': 'LLM not available', 'insights': []}
        
        # Get top diagnoses for each cohort
        ad_top = ad_diagnoses['FullDiagnosisName'].value_counts().head(10).to_dict()
        control_top = control_diagnoses['FullDiagnosisName'].value_counts().head(10).to_dict()
        
        prompt = f"""Compare these two patient cohorts:

AD Cohort Top Diagnoses:
{json.dumps(ad_top, indent=2)}

Control Cohort Top Diagnoses:
{json.dumps(control_top, indent=2)}

Provide:
1. Key differences between cohorts
2. AD-specific patterns
3. Shared comorbidities
4. Clinical insights
5. Research implications

Be concise but comprehensive."""
        
        comparison = self._call_llm(prompt)
        
        return {
            'comparison': comparison,
            'model': self.model,
            'ad_cohort_size': len(ad_diagnoses['PatientID'].unique()),
            'control_cohort_size': len(control_diagnoses['PatientID'].unique())
        }
    
    def generate_clinical_summary(self, patient_data: Dict[str, pd.DataFrame]) -> str:
        """
        Generate comprehensive clinical summary using GPT-5.1
        """
        if not self.client:
            return "LLM not available for clinical summary generation."
        
        # Extract key information
        summary_data = {
            'total_patients': len(patient_data.get('demographics', pd.DataFrame())),
            'total_diagnoses': len(patient_data.get('diagnoses', pd.DataFrame())),
            'total_medications': len(patient_data.get('medications', pd.DataFrame())),
            'total_labs': len(patient_data.get('labs', pd.DataFrame()))
        }
        
        prompt = f"""Generate a clinical research summary for an Alzheimer's Disease phenotyping study:

Dataset Statistics:
{json.dumps(summary_data, indent=2)}

Provide:
1. Study overview
2. Data characteristics
3. Potential research questions
4. Analytical approaches
5. Expected insights

Keep it professional and research-focused."""
        
        return self._call_llm(prompt)
