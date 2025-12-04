"""
Association analysis service
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from app.services.data_loader import DataLoader
from app.utils.statistics import perform_statistical_test, bonferroni_correction
from app.utils.icd10 import icd_chapter_to_name


class AssociationService:
    """Service for association analysis between AD and controls"""
    
    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.data_loader = data_loader or DataLoader()
    
    def analyze_diagnosis(
        self,
        diag_key: str = "FullDiagnosisName",
        stratify_by: Optional[str] = None,
        alpha: float = 0.05
    ) -> Dict:
        """
        Analyze diagnosis associations
        
        Args:
            diag_key: Diagnosis level (FullDiagnosisName, Level3_Category, Level2_Category)
            stratify_by: Stratify by 'sex' or None
            alpha: Significance level
        
        Returns:
            Dictionary with analysis results
        """
        # Load data
        ad_diag = self.data_loader.load_diagnosis("ad")
        con_diag = self.data_loader.load_diagnosis("control")
        
        ad_counts = self.data_loader.get_patient_counts("ad")
        con_counts = self.data_loader.get_patient_counts("control")
        
        if stratify_by == 'sex':
            return self._analyze_diagnosis_stratified(ad_diag, con_diag, diag_key, alpha)
        else:
            return self._analyze_diagnosis_overall(ad_diag, con_diag, diag_key, alpha)
    
    def _analyze_diagnosis_overall(
        self,
        ad_diag: pd.DataFrame,
        con_diag: pd.DataFrame,
        diag_key: str,
        alpha: float
    ) -> Dict:
        """Analyze diagnosis without stratification"""
        ad_counts = self.data_loader.get_patient_counts("ad")
        con_counts = self.data_loader.get_patient_counts("control")
        
        # Count diagnoses
        ad_diag_counts = self.data_loader.count_diagnosis(
            ad_diag, ad_counts['total'], diag_key
        )
        con_diag_counts = self.data_loader.count_diagnosis(
            con_diag, con_counts['total'], diag_key
        )
        
        # Merge
        merged = ad_diag_counts.merge(
            con_diag_counts,
            on=diag_key,
            how='outer',
            suffixes=('_ad', '_con')
        ).fillna(0)
        
        # Perform statistical tests
        results = []
        for _, row in merged.iterrows():
            test_result = perform_statistical_test(
                int(row['Count_ad']),
                int(row['Count_r_ad']),
                int(row['Count_con']),
                int(row['Count_r_con'])
            )
            test_result[diag_key] = row[diag_key]
            results.append(test_result)
        
        results_df = pd.DataFrame(results)
        
        # Bonferroni correction
        n_tests = len(results_df)
        corrected_alpha = bonferroni_correction(alpha, n_tests)
        
        # Categorize significance
        results_df['significant'] = results_df['pvalue'] < corrected_alpha
        results_df['enriched'] = 'Not Significant'
        results_df.loc[
            (results_df['significant']) & (results_df['log2_odds_ratio'] > 1),
            'enriched'
        ] = 'Alzheimer Enriched'
        results_df.loc[
            (results_df['significant']) & (results_df['log2_odds_ratio'] < -1),
            'enriched'
        ] = 'Control Enriched'
        
        # Add ICD-10 chapter info
        if 'ValueL' in ad_diag.columns:
            chapter_map = ad_diag[[diag_key, 'ValueL']].drop_duplicates().set_index(diag_key)['ValueL']
            results_df['chapter'] = results_df[diag_key].map(chapter_map)
            results_df['chapter_name'] = results_df['chapter'].apply(icd_chapter_to_name)
        
        return {
            'results': results_df.to_dict('records'),
            'summary': {
                'total_tests': n_tests,
                'corrected_alpha': corrected_alpha,
                'significant_count': int(results_df['significant'].sum()),
                'alzheimer_enriched': int((results_df['enriched'] == 'Alzheimer Enriched').sum()),
                'control_enriched': int((results_df['enriched'] == 'Control Enriched').sum())
            }
        }
    
    def _analyze_diagnosis_stratified(
        self,
        ad_diag: pd.DataFrame,
        con_diag: pd.DataFrame,
        diag_key: str,
        alpha: float
    ) -> Dict:
        """Analyze diagnosis stratified by sex"""
        # Separate by sex
        ad_female = ad_diag[ad_diag['Sex'] == 'Female']
        ad_male = ad_diag[ad_diag['Sex'] == 'Male']
        con_female = con_diag[con_diag['Sex'] == 'Female']
        con_male = con_diag[con_diag['Sex'] == 'Male']
        
        # Get counts
        ad_counts = self.data_loader.get_patient_counts("ad")
        con_counts = self.data_loader.get_patient_counts("control")
        
        # Analyze each group
        female_results = self._analyze_diagnosis_overall(
            ad_female, con_female, diag_key, alpha
        )
        male_results = self._analyze_diagnosis_overall(
            ad_male, con_male, diag_key, alpha
        )
        
        # Combine results
        female_df = pd.DataFrame(female_results['results'])
        male_df = pd.DataFrame(male_results['results'])
        
        merged = female_df.merge(
            male_df,
            on=diag_key,
            how='outer',
            suffixes=('_F', '_M')
        )
        
        return {
            'results': merged.to_dict('records'),
            'female_summary': female_results['summary'],
            'male_summary': male_results['summary']
        }
    
    def analyze_medications(
        self,
        stratify_by: Optional[str] = None,
        alpha: float = 0.05
    ) -> Dict:
        """Analyze medication associations"""
        ad_meds = self.data_loader.load_medications("ad")
        con_meds = self.data_loader.load_medications("control")
        
        ad_counts = self.data_loader.get_patient_counts("ad")
        con_counts = self.data_loader.get_patient_counts("control")
        
        # Count medications
        med_col = 'MedicationGenericName'
        if med_col not in ad_meds.columns:
            med_col = 'SimpleGenericName'
        
        ad_med_counts = (
            ad_meds[['PatientID', med_col]]
            .drop_duplicates()
            .groupby(med_col)['PatientID']
            .nunique()
            .reset_index()
        )
        ad_med_counts.columns = [med_col, 'Count']
        ad_med_counts['Count_r'] = ad_counts['total'] - ad_med_counts['Count']
        
        con_med_counts = (
            con_meds[['PatientID', med_col]]
            .drop_duplicates()
            .groupby(med_col)['PatientID']
            .nunique()
            .reset_index()
        )
        con_med_counts.columns = [med_col, 'Count']
        con_med_counts['Count_r'] = con_counts['total'] - con_med_counts['Count']
        
        # Merge and test
        merged = ad_med_counts.merge(
            con_med_counts,
            on=med_col,
            how='outer',
            suffixes=('_ad', '_con')
        ).fillna(0)
        
        results = []
        for _, row in merged.iterrows():
            test_result = perform_statistical_test(
                int(row['Count_ad']),
                int(row['Count_r_ad']),
                int(row['Count_con']),
                int(row['Count_r_con'])
            )
            test_result[med_col] = row[med_col]
            results.append(test_result)
        
        results_df = pd.DataFrame(results)
        
        # Bonferroni correction
        n_tests = len(results_df)
        corrected_alpha = bonferroni_correction(alpha, n_tests)
        
        results_df['significant'] = results_df['pvalue'] < corrected_alpha
        results_df['enriched'] = 'Not Significant'
        results_df.loc[
            (results_df['significant']) & (results_df['log2_odds_ratio'] > 1),
            'enriched'
        ] = 'Alzheimer Enriched'
        results_df.loc[
            (results_df['significant']) & (results_df['log2_odds_ratio'] < -1),
            'enriched'
        ] = 'Control Enriched'
        
        return {
            'results': results_df.to_dict('records'),
            'summary': {
                'total_tests': n_tests,
                'corrected_alpha': corrected_alpha,
                'significant_count': int(results_df['significant'].sum())
            }
        }
    
    def analyze_lab_results(
        self,
        stratify_by: Optional[str] = None,
        alpha: float = 0.05
    ) -> Dict:
        """Analyze lab result associations"""
        from app.utils.statistics import test_continuous_variable
        
        ad_labs = self.data_loader.load_lab_results("ad")
        con_labs = self.data_loader.load_lab_results("control")
        
        # Combine and process
        all_labs = pd.concat([ad_labs, con_labs], ignore_index=True)
        all_labs = all_labs[all_labs['Values'].notna()]
        
        # Mark AD patients
        ad_patients = set(ad_labs['PatientID'].unique())
        all_labs['isAD'] = all_labs['PatientID'].isin(ad_patients)
        
        # Convert values to numeric
        def to_numeric(val):
            try:
                return float(val)
            except:
                return np.nan
        
        all_labs['numeric_value'] = all_labs['Values'].apply(to_numeric)
        all_labs = all_labs[all_labs['numeric_value'].notna()]
        
        # Calculate median per patient per test
        lab_medians = (
            all_labs.groupby(['PatientID', 'TestName'])['numeric_value']
            .median()
            .reset_index()
        )
        
        # Merge with AD status
        demo = pd.concat([
            self.data_loader.load_demographics("ad"),
            self.data_loader.load_demographics("control")
        ], ignore_index=True)
        
        # Add isAD column if not present
        if 'isAD' not in demo.columns:
            demo['isAD'] = demo['PatientID'].isin(ad_patients)
        
        lab_medians = lab_medians.merge(
            demo[['PatientID', 'isAD']],
            on='PatientID'
        )
        
        # Test each lab
        results = []
        for test_name in lab_medians['TestName'].unique():
            test_data = lab_medians[lab_medians['TestName'] == test_name]
            ad_vals = test_data[test_data['isAD'] == 1]['numeric_value']
            con_vals = test_data[test_data['isAD'] == 0]['numeric_value']
            
            if len(ad_vals) > 0 and len(con_vals) > 0:
                test_result = test_continuous_variable(ad_vals, con_vals)
                test_result['TestName'] = test_name
                results.append(test_result)
        
        results_df = pd.DataFrame(results)
        
        # Bonferroni correction
        n_tests = len(results_df)
        corrected_alpha = bonferroni_correction(alpha, n_tests)
        results_df['significant'] = results_df['pvalue'] < corrected_alpha
        
        return {
            'results': results_df.to_dict('records'),
            'summary': {
                'total_tests': n_tests,
                'corrected_alpha': corrected_alpha,
                'significant_count': int(results_df['significant'].sum())
            }
        }

