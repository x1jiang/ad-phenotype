"""
Phenotype explanation service using LLM
"""
import pandas as pd
from typing import Dict, List, Optional
from app.services.data_loader import DataLoader
from app.config import settings
import json
import os


class PhenotypeExplainer:
    """Service for explaining phenotypes using LLM"""
    
    def __init__(
        self,
        data_loader: Optional[DataLoader] = None,
        use_llm: Optional[bool] = None
    ):
        self.data_loader = data_loader or DataLoader()
        self.use_llm = use_llm if use_llm is not None else settings.use_llm
        self._llm_client = None
        self._phenotype_cache = {}
    
    def _get_llm_client(self):
        """Get LLM client"""
        if self._llm_client is not None:
            return self._llm_client
        
        if not self.use_llm:
            return None
        
        try:
            if settings.llm_provider == "openai":
                import openai
                api_key = settings.openai_api_key or os.getenv('OPENAI_API_KEY')
                if api_key:
                    self._llm_client = openai.OpenAI(api_key=api_key)
        except ImportError:
            pass
        
        return self._llm_client
    
    def explain_phenotype(
        self,
        phenotype_name: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Explain a phenotype with clinical significance
        
        Args:
            phenotype_name: Name of the phenotype/diagnosis
            context: Additional context (prevalence, associations, etc.)
        
        Returns:
            Dictionary with explanation, clinical significance, and implications
        """
        # Check cache first
        if phenotype_name in self._phenotype_cache:
            return self._phenotype_cache[phenotype_name]
        
        if self.use_llm and self._get_llm_client():
            explanation = self._llm_explain_phenotype(phenotype_name, context)
        else:
            explanation = self._rule_based_explanation(phenotype_name, context)
        
        # Cache result
        self._phenotype_cache[phenotype_name] = explanation
        
        return explanation
    
    def _llm_explain_phenotype(
        self,
        phenotype_name: str,
        context: Optional[Dict]
    ) -> Dict:
        """Use LLM to explain phenotype"""
        client = self._get_llm_client()
        if not client:
            return self._rule_based_explanation(phenotype_name, context)
        
        context_str = ""
        if context:
            context_str = f"\nContext: Prevalence in AD cohort: {context.get('ad_prevalence', 'N/A')}%, " \
                         f"Control cohort: {context.get('control_prevalence', 'N/A')}%, " \
                         f"Odds Ratio: {context.get('odds_ratio', 'N/A')}"
        
        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical expert specializing in Alzheimer's disease and computational phenotyping. Provide clear, evidence-based explanations of clinical phenotypes."
                    },
                    {
                        "role": "user",
                        "content": f"""Explain the clinical phenotype: {phenotype_name}{context_str}

Provide a JSON response with:
1. "description": Brief description of the phenotype
2. "clinical_significance": What this phenotype indicates clinically
3. "ad_association": How this relates to Alzheimer's disease
4. "implications": Clinical or research implications
5. "severity": Assessment of severity (Mild/Moderate/Severe)
6. "common_comorbidities": List of commonly associated conditions

Format as JSON only."""
                    }
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                'phenotype': phenotype_name,
                'description': result.get('description', ''),
                'clinical_significance': result.get('clinical_significance', ''),
                'ad_association': result.get('ad_association', ''),
                'implications': result.get('implications', ''),
                'severity': result.get('severity', 'Unknown'),
                'common_comorbidities': result.get('common_comorbidities', []),
                'source': 'LLM'
            }
        except Exception as e:
            print(f"LLM explanation error: {e}")
            return self._rule_based_explanation(phenotype_name, context)
    
    def _rule_based_explanation(
        self,
        phenotype_name: str,
        context: Optional[Dict]
    ) -> Dict:
        """Rule-based phenotype explanation"""
        name_lower = str(phenotype_name).lower()
        
        # Cognitive/Neurological phenotypes
        if any(term in name_lower for term in ['dementia', 'alzheimer', 'cognitive', 'memory']):
            return {
                'phenotype': phenotype_name,
                'description': 'Cognitive impairment or dementia-related condition',
                'clinical_significance': 'Indicates neurological decline affecting memory, thinking, or daily functioning',
                'ad_association': 'Directly associated with Alzheimer\'s disease progression',
                'implications': 'May indicate disease severity, progression rate, or need for cognitive interventions',
                'severity': 'Severe',
                'common_comorbidities': ['Depression', 'Anxiety', 'Sleep disorders'],
                'source': 'Rule-based'
            }
        
        # Cardiovascular phenotypes
        elif any(term in name_lower for term in ['hypertension', 'cardiac', 'heart', 'cardiovascular', 'stroke']):
            return {
                'phenotype': phenotype_name,
                'description': 'Cardiovascular condition affecting heart or blood vessels',
                'clinical_significance': 'May contribute to cognitive decline through vascular mechanisms',
                'ad_association': 'Vascular risk factors are associated with increased AD risk and faster progression',
                'implications': 'Suggests need for cardiovascular risk management in AD patients',
                'severity': 'Moderate to Severe',
                'common_comorbidities': ['Diabetes', 'Hyperlipidemia', 'Atrial fibrillation'],
                'source': 'Rule-based'
            }
        
        # Metabolic phenotypes
        elif any(term in name_lower for term in ['diabetes', 'metabolic', 'glucose', 'insulin', 'obesity']):
            return {
                'phenotype': phenotype_name,
                'description': 'Metabolic disorder affecting glucose regulation or metabolism',
                'clinical_significance': 'Metabolic dysfunction may accelerate cognitive decline',
                'ad_association': 'Type 2 diabetes is a known risk factor for AD and may worsen cognitive outcomes',
                'implications': 'Metabolic control may be important for cognitive health in AD',
                'severity': 'Moderate',
                'common_comorbidities': ['Hypertension', 'Cardiovascular disease', 'Kidney disease'],
                'source': 'Rule-based'
            }
        
        # Psychiatric phenotypes
        elif any(term in name_lower for term in ['depression', 'anxiety', 'psychiatric', 'mental', 'mood']):
            return {
                'phenotype': phenotype_name,
                'description': 'Psychiatric or mood disorder',
                'clinical_significance': 'Common in AD patients, may be early symptom or consequence of disease',
                'ad_association': 'Depression and anxiety are common comorbidities that may precede or accompany AD',
                'implications': 'May require integrated treatment addressing both cognitive and mood symptoms',
                'severity': 'Moderate',
                'common_comorbidities': ['Sleep disorders', 'Cognitive impairment', 'Social isolation'],
                'source': 'Rule-based'
            }
        
        # Default
        else:
            return {
                'phenotype': phenotype_name,
                'description': f'Clinical condition: {phenotype_name}',
                'clinical_significance': 'Requires clinical evaluation to determine significance',
                'ad_association': 'Association with AD may vary based on specific condition',
                'implications': 'Further clinical assessment recommended',
                'severity': 'Unknown',
                'common_comorbidities': [],
                'source': 'Rule-based'
            }
    
    def explain_phenotype_group(
        self,
        phenotypes: List[str],
        group_name: str = "Phenotype Group"
    ) -> Dict:
        """Explain a group of related phenotypes"""
        explanations = [self.explain_phenotype(p) for p in phenotypes]
        
        # Aggregate information
        severities = [e['severity'] for e in explanations]
        categories = []
        
        for e in explanations:
            if 'Cognitive' in e['description']:
                categories.append('Cognitive')
            elif 'Cardiovascular' in e['description']:
                categories.append('Cardiovascular')
            elif 'Metabolic' in e['description']:
                categories.append('Metabolic')
            elif 'Psychiatric' in e['description']:
                categories.append('Psychiatric')
        
        return {
            'group_name': group_name,
            'phenotypes': explanations,
            'primary_categories': list(set(categories)),
            'severity_distribution': {
                'mild': severities.count('Mild'),
                'moderate': severities.count('Moderate'),
                'severe': severities.count('Severe')
            },
            'summary': f'Group contains {len(phenotypes)} phenotypes across {len(set(categories))} categories'
        }
    
    def get_top_phenotypes_with_explanations(
        self,
        association_results: List[Dict],
        top_n: int = 10
    ) -> List[Dict]:
        """Get top significant phenotypes with explanations"""
        results_df = pd.DataFrame(association_results)
        
        # Filter significant and sort by p-value
        significant = results_df[results_df.get('significant', False)]
        if len(significant) == 0:
            significant = results_df.nsmallest(top_n, 'pvalue')
        
        top_phenotypes = significant.nsmallest(top_n, 'pvalue')
        
        explanations = []
        for _, row in top_phenotypes.iterrows():
            phenotype_name = row.get('FullDiagnosisName') or row.get('DiagnosisName', 'Unknown')
            
            context = {
                'ad_prevalence': row.get('Count_ad', 0),
                'control_prevalence': row.get('Count_con', 0),
                'odds_ratio': row.get('odds_ratio', 1.0),
                'pvalue': row.get('pvalue', 1.0)
            }
            
            explanation = self.explain_phenotype(phenotype_name, context)
            explanation['statistics'] = {
                'odds_ratio': float(row.get('odds_ratio', 1.0)),
                'log2_odds_ratio': float(row.get('log2_odds_ratio', 0.0)),
                'pvalue': float(row.get('pvalue', 1.0)),
                'neg_log10_pvalue': float(row.get('neg_log10_pvalue', 0.0)),
                'enriched': row.get('enriched', 'Not Significant')
            }
            
            explanations.append(explanation)
        
        return explanations

