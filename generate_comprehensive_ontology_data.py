"""
Comprehensive Ontology-Aligned EHR Data Generation
Following OntoCodex framework with extensive OMOP CDM alignment
Includes: Demographics, Diagnoses, Medications, Labs, Procedures, Imaging, Vital Signs, Social History
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

# Set random seed for reproducibility (changed to introduce more variability)
np.random.seed(123)
random.seed(123)

# Configuration
N_AD_PATIENTS = 200
N_CONTROL_PATIENTS = 200

# OMOP-aligned terminologies
ETHNICITIES = ['Hispanic or Latino', 'Not Hispanic or Latino', 'Unknown', 'Patient Refused']
RACES = ['White', 'Black or African American', 'Asian', 'Native Hawaiian or Other Pacific Islander', 
         'American Indian or Alaska Native', 'Other', 'Unknown']
SEXES = ['Male', 'Female']

# Comprehensive AD-specific diagnoses with SNOMED CT codes
AD_DIAGNOSES = [
    {'name': 'Alzheimer disease', 'snomed': '26929004', 'icd10': 'G30.9', 'category': 'Mental, Behavioral disorders', 
     'subcategory': 'Alzheimer disease', 'omop_concept_id': '378419', 'severity': 'Moderate'},
    {'name': 'Alzheimer disease with early onset', 'snomed': '230270009', 'icd10': 'G30.0', 
     'category': 'Mental, Behavioral disorders', 'subcategory': 'Alzheimer disease', 'omop_concept_id': '378420', 'severity': 'Severe'},
    {'name': 'Alzheimer disease with late onset', 'snomed': '416780008', 'icd10': 'G30.1', 
     'category': 'Mental, Behavioral disorders', 'subcategory': 'Alzheimer disease', 'omop_concept_id': '378421', 'severity': 'Moderate'},
    {'name': 'Dementia in Alzheimer disease', 'snomed': '52448006', 'icd10': 'F00', 
     'category': 'Mental, Behavioral disorders', 'subcategory': 'Dementia', 'omop_concept_id': '4182210', 'severity': 'Severe'},
    {'name': 'Vascular dementia', 'snomed': '429998004', 'icd10': 'F01', 
     'category': 'Mental, Behavioral disorders', 'subcategory': 'Dementia', 'omop_concept_id': '4182211', 'severity': 'Moderate'},
    {'name': 'Mixed dementia', 'snomed': '192075002', 'icd10': 'F03', 
     'category': 'Mental, Behavioral disorders', 'subcategory': 'Dementia', 'omop_concept_id': '4182212', 'severity': 'Severe'},
    {'name': 'Mild cognitive impairment', 'snomed': '386806002', 'icd10': 'G31.84', 
     'category': 'Mental, Behavioral disorders', 'subcategory': 'Cognitive impairment', 'omop_concept_id': '4182210', 'severity': 'Mild'},
    {'name': 'Memory impairment', 'snomed': '386807006', 'icd10': 'R41.3', 
     'category': 'Mental, Behavioral disorders', 'subcategory': 'Cognitive impairment', 'omop_concept_id': '4011630', 'severity': 'Mild'},
    {'name': 'Behavioral disturbance in dementia', 'snomed': '192069005', 'icd10': 'F02.81', 
     'category': 'Mental, Behavioral disorders', 'subcategory': 'Behavioral disorder', 'omop_concept_id': '4310293', 'severity': 'Moderate'},
    {'name': 'Frontotemporal dementia', 'snomed': '230258005', 'icd10': 'G31.09', 
     'category': 'Mental, Behavioral disorders', 'subcategory': 'Dementia', 'omop_concept_id': '4103512', 'severity': 'Severe'},
]

# Comprehensive ADRD risk factors
ADRD_RISK_FACTORS = [
    # Cardiovascular
    {'name': 'Type 2 diabetes mellitus', 'snomed': '44054006', 'icd10': 'E11.9', 'category': 'Endocrine, nutritional disorders',
     'subcategory': 'Diabetes', 'omop_concept_id': '201826', 'risk_weight': 1.5, 'prevalence_ad': 0.35, 'prevalence_control': 0.25},
    {'name': 'Essential hypertension', 'snomed': '59621000', 'icd10': 'I10', 'category': 'Circulatory system diseases',
     'subcategory': 'Hypertension', 'omop_concept_id': '320128', 'risk_weight': 1.3, 'prevalence_ad': 0.7, 'prevalence_control': 0.5},
    {'name': 'Hyperlipidemia', 'snomed': '55822004', 'icd10': 'E78.5', 'category': 'Endocrine, nutritional disorders',
     'subcategory': 'Lipid disorders', 'omop_concept_id': '432867', 'risk_weight': 1.2, 'prevalence_ad': 0.6, 'prevalence_control': 0.5},
    {'name': 'Coronary atherosclerosis', 'snomed': '53741008', 'icd10': 'I25.10', 'category': 'Circulatory system diseases',
     'subcategory': 'Ischemic heart disease', 'omop_concept_id': '319835', 'risk_weight': 1.4, 'prevalence_ad': 0.4, 'prevalence_control': 0.25},
    {'name': 'Atrial fibrillation', 'snomed': '49436004', 'icd10': 'I48.91', 'category': 'Circulatory system diseases',
     'subcategory': 'Cardiac arrhythmias', 'omop_concept_id': '313217', 'risk_weight': 1.6, 'prevalence_ad': 0.3, 'prevalence_control': 0.15},
    {'name': 'Heart failure', 'snomed': '84114007', 'icd10': 'I50.9', 'category': 'Circulatory system diseases',
     'subcategory': 'Heart failure', 'omop_concept_id': '316139', 'risk_weight': 1.5, 'prevalence_ad': 0.25, 'prevalence_control': 0.15},
    
    # Metabolic
    {'name': 'Obesity', 'snomed': '414915002', 'icd10': 'E66.9', 'category': 'Endocrine, nutritional disorders',
     'subcategory': 'Obesity', 'omop_concept_id': '433736', 'risk_weight': 1.3, 'prevalence_ad': 0.35, 'prevalence_control': 0.3},
    {'name': 'Metabolic syndrome', 'snomed': '237602007', 'icd10': 'E88.81', 'category': 'Endocrine, nutritional disorders',
     'subcategory': 'Metabolic disorder', 'omop_concept_id': '4216672', 'risk_weight': 1.4, 'prevalence_ad': 0.3, 'prevalence_control': 0.2},
    {'name': 'Vitamin D deficiency', 'snomed': '34713006', 'icd10': 'E55.9', 'category': 'Endocrine, nutritional disorders',
     'subcategory': 'Vitamin deficiency', 'omop_concept_id': '4082033', 'risk_weight': 1.2, 'prevalence_ad': 0.5, 'prevalence_control': 0.3},
    {'name': 'Vitamin B12 deficiency', 'snomed': '190634005', 'icd10': 'E53.8', 'category': 'Endocrine, nutritional disorders',
     'subcategory': 'Vitamin deficiency', 'omop_concept_id': '435928', 'risk_weight': 1.3, 'prevalence_ad': 0.4, 'prevalence_control': 0.2},
    
    # Neurological
    {'name': 'Cerebrovascular disease', 'snomed': '62914000', 'icd10': 'I67.9', 'category': 'Circulatory system diseases',
     'subcategory': 'Cerebrovascular disease', 'omop_concept_id': '381591', 'risk_weight': 2.0, 'prevalence_ad': 0.4, 'prevalence_control': 0.1},
    {'name': 'Ischemic stroke', 'snomed': '422504002', 'icd10': 'I63.9', 'category': 'Circulatory system diseases',
     'subcategory': 'Stroke', 'omop_concept_id': '443454', 'risk_weight': 2.5, 'prevalence_ad': 0.25, 'prevalence_control': 0.08},
    {'name': 'Traumatic brain injury', 'snomed': '127295002', 'icd10': 'S06.9', 'category': 'Injury, poisoning',
     'subcategory': 'Head injury', 'omop_concept_id': '444410', 'risk_weight': 1.7, 'prevalence_ad': 0.15, 'prevalence_control': 0.05},
    {'name': 'Parkinson disease', 'snomed': '49049000', 'icd10': 'G20', 'category': 'Nervous system diseases',
     'subcategory': 'Parkinson disease', 'omop_concept_id': '381270', 'risk_weight': 2.0, 'prevalence_ad': 0.1, 'prevalence_control': 0.03},
    {'name': 'Epilepsy', 'snomed': '84757009', 'icd10': 'G40.909', 'category': 'Nervous system diseases',
     'subcategory': 'Seizure disorder', 'omop_concept_id': '380378', 'risk_weight': 1.5, 'prevalence_ad': 0.15, 'prevalence_control': 0.05},
    
    # Psychiatric
    {'name': 'Major depressive disorder', 'snomed': '35489007', 'icd10': 'F32.9', 'category': 'Mental, Behavioral disorders',
     'subcategory': 'Depression', 'omop_concept_id': '440383', 'risk_weight': 1.8, 'prevalence_ad': 0.5, 'prevalence_control': 0.2},
    {'name': 'Anxiety disorder', 'snomed': '197480006', 'icd10': 'F41.9', 'category': 'Mental, Behavioral disorders',
     'subcategory': 'Anxiety', 'omop_concept_id': '442077', 'risk_weight': 1.4, 'prevalence_ad': 0.4, 'prevalence_control': 0.2},
    {'name': 'Sleep apnea', 'snomed': '73430006', 'icd10': 'G47.33', 'category': 'Nervous system diseases',
     'subcategory': 'Sleep disorder', 'omop_concept_id': '313459', 'risk_weight': 1.6, 'prevalence_ad': 0.35, 'prevalence_control': 0.2},
    {'name': 'Insomnia', 'snomed': '193462001', 'icd10': 'G47.00', 'category': 'Nervous system diseases',
     'subcategory': 'Sleep disorder', 'omop_concept_id': '439676', 'risk_weight': 1.3, 'prevalence_ad': 0.6, 'prevalence_control': 0.3},
    
    # Other
    {'name': 'Chronic kidney disease', 'snomed': '709044004', 'icd10': 'N18.9', 'category': 'Genitourinary system diseases',
     'subcategory': 'Kidney disease', 'omop_concept_id': '46271022', 'risk_weight': 1.3, 'prevalence_ad': 0.3, 'prevalence_control': 0.15},
    {'name': 'Hypothyroidism', 'snomed': '40930008', 'icd10': 'E03.9', 'category': 'Endocrine, nutritional disorders',
     'subcategory': 'Thyroid disorders', 'omop_concept_id': '138384', 'risk_weight': 1.2, 'prevalence_ad': 0.25, 'prevalence_control': 0.2},
    {'name': 'Hearing impairment', 'snomed': '15188001', 'icd10': 'H91.90', 'category': 'Ear diseases',
     'subcategory': 'Hearing loss', 'omop_concept_id': '377889', 'risk_weight': 1.4, 'prevalence_ad': 0.45, 'prevalence_control': 0.25},
    {'name': 'Visual impairment', 'snomed': '397540003', 'icd10': 'H54.7', 'category': 'Eye diseases',
     'subcategory': 'Vision loss', 'omop_concept_id': '377575', 'risk_weight': 1.3, 'prevalence_ad': 0.4, 'prevalence_control': 0.2},
]

# Control-only diagnoses
CONTROL_DIAGNOSES = [
    {'name': 'Osteoarthritis', 'snomed': '396275006', 'icd10': 'M19.90', 'category': 'Musculoskeletal diseases', 
     'subcategory': 'Arthritis', 'omop_concept_id': '80180', 'prevalence': 0.5},
    {'name': 'Gastroesophageal reflux', 'snomed': '235595009', 'icd10': 'K21.9', 'category': 'Digestive system diseases',
     'subcategory': 'GERD', 'omop_concept_id': '318800', 'prevalence': 0.4},
    {'name': 'Back pain', 'snomed': '161891005', 'icd10': 'M54.5', 'category': 'Musculoskeletal diseases',
     'subcategory': 'Back pain', 'omop_concept_id': '194133', 'prevalence': 0.45},
    {'name': 'Chronic obstructive pulmonary disease', 'snomed': '13645005', 'icd10': 'J44.9',
     'category': 'Respiratory diseases', 'subcategory': 'COPD', 'omop_concept_id': '255573', 'prevalence': 0.15},
    {'name': 'Asthma', 'snomed': '195967001', 'icd10': 'J45.909', 'category': 'Respiratory diseases',
     'subcategory': 'Asthma', 'omop_concept_id': '317009', 'prevalence': 0.2},
    {'name': 'Allergic rhinitis', 'snomed': '61582004', 'icd10': 'J30.9', 'category': 'Respiratory diseases',
     'subcategory': 'Allergic conditions', 'omop_concept_id': '374037', 'prevalence': 0.3},
    {'name': 'Diverticulosis', 'snomed': '398050005', 'icd10': 'K57.90', 'category': 'Digestive system diseases',
     'subcategory': 'Intestinal disorder', 'omop_concept_id': '197236', 'prevalence': 0.25},
    {'name': 'Benign prostatic hyperplasia', 'snomed': '266569009', 'icd10': 'N40.0', 'category': 'Genitourinary system diseases',
     'subcategory': 'Prostate disorders', 'omop_concept_id': '198803', 'prevalence': 0.4},
]

# Comprehensive AD medications
AD_MEDICATIONS = [
    # Cholinesterase inhibitors
    {'name': 'Donepezil', 'rxnorm': '135447', 'ingredient': 'donepezil', 'omop_concept_id': '19078461', 
     'class': 'Cholinesterase Inhibitor', 'route': 'Oral', 'frequency': 'Once daily'},
    {'name': 'Rivastigmine', 'rxnorm': '86124', 'ingredient': 'rivastigmine', 'omop_concept_id': '19010482', 
     'class': 'Cholinesterase Inhibitor', 'route': 'Oral/Patch', 'frequency': 'Twice daily or patch'},
    {'name': 'Galantamine', 'rxnorm': '37617', 'ingredient': 'galantamine', 'omop_concept_id': '19097821', 
     'class': 'Cholinesterase Inhibitor', 'route': 'Oral', 'frequency': 'Twice daily'},
    
    # NMDA antagonists
    {'name': 'Memantine', 'rxnorm': '351266', 'ingredient': 'memantine', 'omop_concept_id': '40166305', 
     'class': 'NMDA Antagonist', 'route': 'Oral', 'frequency': 'Once or twice daily'},
    
    # Anti-amyloid (newer)
    {'name': 'Aducanumab', 'rxnorm': '2556640', 'ingredient': 'aducanumab', 'omop_concept_id': '45775965', 
     'class': 'Anti-Amyloid', 'route': 'IV', 'frequency': 'Monthly'},
    {'name': 'Lecanemab', 'rxnorm': '2601723', 'ingredient': 'lecanemab', 'omop_concept_id': '46275930', 
     'class': 'Anti-Amyloid', 'route': 'IV', 'frequency': 'Bi-weekly'},
]

# Common medications (greatly expanded)
COMMON_MEDICATIONS = [
    # Cardiovascular
    {'name': 'Lisinopril', 'rxnorm': '29046', 'ingredient': 'lisinopril', 'omop_concept_id': '1308216', 
     'class': 'ACE Inhibitor', 'route': 'Oral'},
    {'name': 'Losartan', 'rxnorm': '52175', 'ingredient': 'losartan', 'omop_concept_id': '1367500', 
     'class': 'ARB', 'route': 'Oral'},
    {'name': 'Amlodipine', 'rxnorm': '17767', 'ingredient': 'amlodipine', 'omop_concept_id': '1332418', 
     'class': 'Calcium Channel Blocker', 'route': 'Oral'},
    {'name': 'Metoprolol', 'rxnorm': '6918', 'ingredient': 'metoprolol', 'omop_concept_id': '1307046', 
     'class': 'Beta Blocker', 'route': 'Oral'},
    {'name': 'Atorvastatin', 'rxnorm': '83367', 'ingredient': 'atorvastatin', 'omop_concept_id': '1539403', 
     'class': 'Statin', 'route': 'Oral'},
    {'name': 'Simvastatin', 'rxnorm': '36567', 'ingredient': 'simvastatin', 'omop_concept_id': '1539403', 
     'class': 'Statin', 'route': 'Oral'},
    {'name': 'Aspirin', 'rxnorm': '1191', 'ingredient': 'aspirin', 'omop_concept_id': '1112807', 
     'class': 'Antiplatelet', 'route': 'Oral'},
    {'name': 'Clopidogrel', 'rxnorm': '32968', 'ingredient': 'clopidogrel', 'omop_concept_id': '1322184', 
     'class': 'Antiplatelet', 'route': 'Oral'},
    {'name': 'Warfarin', 'rxnorm': '11289', 'ingredient': 'warfarin', 'omop_concept_id': '1310149', 
     'class': 'Anticoagulant', 'route': 'Oral'},
    {'name': 'Apixaban', 'rxnorm': '1364430', 'ingredient': 'apixaban', 'omop_concept_id': '40244464', 
     'class': 'Anticoagulant', 'route': 'Oral'},
    
    # Diabetes
    {'name': 'Metformin', 'rxnorm': '6809', 'ingredient': 'metformin', 'omop_concept_id': '1503297', 
     'class': 'Biguanide', 'route': 'Oral'},
    {'name': 'Insulin glargine', 'rxnorm': '274783', 'ingredient': 'insulin glargine', 'omop_concept_id': '1530014', 
     'class': 'Long-acting insulin', 'route': 'Subcutaneous'},
    {'name': 'Glipizide', 'rxnorm': '4815', 'ingredient': 'glipizide', 'omop_concept_id': '1559684', 
     'class': 'Sulfonylurea', 'route': 'Oral'},
    
    # Psychiatric
    {'name': 'Sertraline', 'rxnorm': '36437', 'ingredient': 'sertraline', 'omop_concept_id': '755695', 
     'class': 'SSRI', 'route': 'Oral'},
    {'name': 'Escitalopram', 'rxnorm': '321988', 'ingredient': 'escitalopram', 'omop_concept_id': '715939', 
     'class': 'SSRI', 'route': 'Oral'},
    {'name': 'Citalopram', 'rxnorm': '2556', 'ingredient': 'citalopram', 'omop_concept_id': '739138', 
     'class': 'SSRI', 'route': 'Oral'},
    {'name': 'Lorazepam', 'rxnorm': '6470', 'ingredient': 'lorazepam', 'omop_concept_id': '740910', 
     'class': 'Benzodiazepine', 'route': 'Oral'},
    {'name': 'Zolpidem', 'rxnorm': '39993', 'ingredient': 'zolpidem', 'omop_concept_id': '19034726', 
     'class': 'Hypnotic', 'route': 'Oral'},
    {'name': 'Quetiapine', 'rxnorm': '35636', 'ingredient': 'quetiapine', 'omop_concept_id': '717136', 
     'class': 'Antipsychotic', 'route': 'Oral'},
    
    # Other common
    {'name': 'Levothyroxine', 'rxnorm': '10582', 'ingredient': 'levothyroxine', 'omop_concept_id': '1718', 
     'class': 'Thyroid hormone', 'route': 'Oral'},
    {'name': 'Omeprazole', 'rxnorm': '7646', 'ingredient': 'omeprazole', 'omop_concept_id': '923645', 
     'class': 'PPI', 'route': 'Oral'},
    {'name': 'Vitamin D', 'rxnorm': '11253', 'ingredient': 'cholecalciferol', 'omop_concept_id': '40163924', 
     'class': 'Vitamin', 'route': 'Oral'},
    {'name': 'Vitamin B12', 'rxnorm': '2551', 'ingredient': 'cyanocobalamin', 'omop_concept_id': '19133701', 
     'class': 'Vitamin', 'route': 'Oral'},
    {'name': 'Multivitamin', 'rxnorm': '89905', 'ingredient': 'multivitamin', 'omop_concept_id': '19127443', 
     'class': 'Vitamin', 'route': 'Oral'},
]

# Comprehensive lab tests with LOINC codes
LAB_TESTS = [
    # Cognitive/Neurological
    {'name': 'Mini-Mental State Examination', 'loinc': '72107-6', 'omop_concept_id': '44791053', 
     'unit': 'score', 'normal_range': (24, 30), 'category': 'Cognitive'},
    {'name': 'Montreal Cognitive Assessment', 'loinc': '72172-0', 'omop_concept_id': '40758558', 
     'unit': 'score', 'normal_range': (26, 30), 'category': 'Cognitive'},
    {'name': 'Apolipoprotein E genotype', 'loinc': '21636-6', 'omop_concept_id': '40762499', 
     'unit': '', 'normal_range': None, 'category': 'Genetic'},
    
    # Metabolic
    {'name': 'Hemoglobin A1c', 'loinc': '4548-4', 'omop_concept_id': '3004410', 
     'unit': '%', 'normal_range': (4.0, 5.7), 'category': 'Metabolic'},
    {'name': 'Fasting glucose', 'loinc': '1558-6', 'omop_concept_id': '3004501', 
     'unit': 'mg/dL', 'normal_range': (70, 100), 'category': 'Metabolic'},
    {'name': 'Total cholesterol', 'loinc': '2093-3', 'omop_concept_id': '3027114', 
     'unit': 'mg/dL', 'normal_range': (125, 200), 'category': 'Lipid'},
    {'name': 'LDL cholesterol', 'loinc': '18262-6', 'omop_concept_id': '3028437', 
     'unit': 'mg/dL', 'normal_range': (50, 100), 'category': 'Lipid'},
    {'name': 'HDL cholesterol', 'loinc': '18263-4', 'omop_concept_id': '3007070', 
     'unit': 'mg/dL', 'normal_range': (40, 60), 'category': 'Lipid'},
    {'name': 'Triglycerides', 'loinc': '2571-8', 'omop_concept_id': '3022192', 
     'unit': 'mg/dL', 'normal_range': (50, 150), 'category': 'Lipid'},
    
    # Vitamins
    {'name': 'Vitamin B12', 'loinc': '2132-9', 'omop_concept_id': '3013762', 
     'unit': 'pg/mL', 'normal_range': (200, 900), 'category': 'Vitamin'},
    {'name': 'Folate', 'loinc': '2284-8', 'omop_concept_id': '3007435', 
     'unit': 'ng/mL', 'normal_range': (2.7, 17), 'category': 'Vitamin'},
    {'name': 'Vitamin D, 25-hydroxy', 'loinc': '1989-3', 'omop_concept_id': '3038288', 
     'unit': 'ng/mL', 'normal_range': (30, 100), 'category': 'Vitamin'},
    
    # Thyroid
    {'name': 'TSH', 'loinc': '3016-3', 'omop_concept_id': '3016723', 
     'unit': 'mIU/L', 'normal_range': (0.5, 4.5), 'category': 'Thyroid'},
    {'name': 'Free T4', 'loinc': '3024-7', 'omop_concept_id': '3007359', 
     'unit': 'ng/dL', 'normal_range': (0.8, 1.8), 'category': 'Thyroid'},
    
    # Kidney function
    {'name': 'Creatinine', 'loinc': '2160-0', 'omop_concept_id': '3016723', 
     'unit': 'mg/dL', 'normal_range': (0.7, 1.3), 'category': 'Renal'},
    {'name': 'BUN', 'loinc': '3094-0', 'omop_concept_id': '3013682', 
     'unit': 'mg/dL', 'normal_range': (7, 20), 'category': 'Renal'},
    {'name': 'eGFR', 'loinc': '33914-3', 'omop_concept_id': '40762352', 
     'unit': 'mL/min/1.73m2', 'normal_range': (60, 120), 'category': 'Renal'},
    
    # Liver function
    {'name': 'ALT', 'loinc': '1742-6', 'omop_concept_id': '3006923', 
     'unit': 'U/L', 'normal_range': (7, 56), 'category': 'Hepatic'},
    {'name': 'AST', 'loinc': '1920-8', 'omop_concept_id': '3013721', 
     'unit': 'U/L', 'normal_range': (10, 40), 'category': 'Hepatic'},
    {'name': 'Albumin', 'loinc': '1751-7', 'omop_concept_id': '3024561', 
     'unit': 'g/dL', 'normal_range': (3.5, 5.5), 'category': 'Hepatic'},
    
    # Inflammation
    {'name': 'C-reactive protein', 'loinc': '1988-5', 'omop_concept_id': '3020460', 
     'unit': 'mg/L', 'normal_range': (0, 3), 'category': 'Inflammation'},
    {'name': 'ESR', 'loinc': '4537-7', 'omop_concept_id': '3010813', 
     'unit': 'mm/hr', 'normal_range': (0, 20), 'category': 'Inflammation'},
    
    # Complete panels
    {'name': 'Complete Blood Count', 'loinc': '58410-2', 'omop_concept_id': '3000963', 
     'unit': '', 'normal_range': None, 'category': 'Hematology'},
    {'name': 'Comprehensive Metabolic Panel', 'loinc': '24323-8', 'omop_concept_id': '3006140', 
     'unit': '', 'normal_range': None, 'category': 'Chemistry'},
    {'name': 'Lipid Panel', 'loinc': '57698-3', 'omop_concept_id': '3019900', 
     'unit': '', 'normal_range': None, 'category': 'Lipid'},
    
    # AD-specific biomarkers
    {'name': 'Tau protein, CSF', 'loinc': '14639-5', 'omop_concept_id': '40761511', 
     'unit': 'pg/mL', 'normal_range': (0, 300), 'category': 'AD Biomarker'},
    {'name': 'Beta-amyloid 42, CSF', 'loinc': '14638-7', 'omop_concept_id': '40761512', 
     'unit': 'pg/mL', 'normal_range': (500, 1000), 'category': 'AD Biomarker'},
    {'name': 'Homocysteine', 'loinc': '13965-5', 'omop_concept_id': '3024540', 
     'unit': 'umol/L', 'normal_range': (5, 15), 'category': 'AD Risk'},
]

# Imaging procedures with CPT codes
IMAGING_PROCEDURES = [
    {'name': 'MRI Brain without contrast', 'cpt': '70551', 'snomed': '241615005', 
     'omop_concept_id': '4013636', 'category': 'Neuroimaging'},
    {'name': 'MRI Brain with contrast', 'cpt': '70552', 'snomed': '241621009', 
     'omop_concept_id': '4013637', 'category': 'Neuroimaging'},
    {'name': 'CT Head without contrast', 'cpt': '70450', 'snomed': '77477000', 
     'omop_concept_id': '4059386', 'category': 'Neuroimaging'},
    {'name': 'PET Brain FDG', 'cpt': '78608', 'snomed': '44491008', 
     'omop_concept_id': '4051282', 'category': 'Neuroimaging'},
    {'name': 'Amyloid PET scan', 'cpt': '78814', 'snomed': '722041001', 
     'omop_concept_id': '45763606', 'category': 'Neuroimaging'},
    {'name': 'Chest X-ray', 'cpt': '71046', 'snomed': '399208008', 
     'omop_concept_id': '4058331', 'category': 'Diagnostic'},
    {'name': 'Echocardiogram', 'cpt': '93306', 'snomed': '40701008', 
     'omop_concept_id': '4013637', 'category': 'Cardiac'},
    {'name': 'Carotid ultrasound', 'cpt': '93880', 'snomed': '241614009', 
     'omop_concept_id': '4059385', 'category': 'Vascular'},
]

# Treatment procedures with CPT codes
TREATMENT_PROCEDURES = [
    {'name': 'Cognitive behavioral therapy', 'cpt': '90832', 'snomed': '385898002', 
     'omop_concept_id': '4130086', 'category': 'Psychotherapy'},
    {'name': 'Physical therapy evaluation', 'cpt': '97161', 'snomed': '91251008', 
     'omop_concept_id': '2721042', 'category': 'Rehabilitation'},
    {'name': 'Occupational therapy', 'cpt': '97165', 'snomed': '310128004', 
     'omop_concept_id': '2721043', 'category': 'Rehabilitation'},
    {'name': 'Speech therapy', 'cpt': '92507', 'snomed': '80092005', 
     'omop_concept_id': '2721044', 'category': 'Rehabilitation'},
    {'name': 'Neuropsychological testing', 'cpt': '96132', 'snomed': '443730003', 
     'omop_concept_id': '2721045', 'category': 'Assessment'},
    {'name': 'Memory training', 'cpt': '97532', 'snomed': '410172006', 
     'omop_concept_id': '4059388', 'category': 'Cognitive'},
    {'name': 'Medication management', 'cpt': '99215', 'snomed': '182834008', 
     'omop_concept_id': '581477', 'category': 'Management'},
]

# Vital signs
VITAL_SIGNS = [
    {'name': 'Systolic Blood Pressure', 'loinc': '8480-6', 'unit': 'mmHg', 'normal_range': (90, 120)},
    {'name': 'Diastolic Blood Pressure', 'loinc': '8462-4', 'unit': 'mmHg', 'normal_range': (60, 80)},
    {'name': 'Heart Rate', 'loinc': '8867-4', 'unit': 'beats/min', 'normal_range': (60, 100)},
    {'name': 'Body Temperature', 'loinc': '8310-5', 'unit': 'F', 'normal_range': (97.0, 99.0)},
    {'name': 'Respiratory Rate', 'loinc': '9279-1', 'unit': 'breaths/min', 'normal_range': (12, 20)},
    {'name': 'Oxygen Saturation', 'loinc': '2708-6', 'unit': '%', 'normal_range': (95, 100)},
    {'name': 'Body Weight', 'loinc': '29463-7', 'unit': 'kg', 'normal_range': (50, 100)},
    {'name': 'Height', 'loinc': '8302-2', 'unit': 'cm', 'normal_range': (150, 190)},
    {'name': 'BMI', 'loinc': '39156-5', 'unit': 'kg/m2', 'normal_range': (18.5, 25)},
]

# Social history
SMOKING_STATUS = ['Never smoker', 'Former smoker', 'Current smoker', 'Unknown']
ALCOHOL_USE = ['None', 'Occasional', 'Moderate', 'Heavy', 'Unknown']
EDUCATION_LEVEL = ['Less than high school', 'High school', 'Some college', 'College degree', 'Graduate degree', 'Unknown']
MARITAL_STATUS = ['Married', 'Divorced', 'Widowed', 'Never married', 'Unknown']


def generate_demographics(n_patients, cohort, base_age=70):
    """Generate comprehensive demographics with realistic overlap"""
    demographics = []
    base_date = datetime.now()
    
    for i in range(n_patients):
        patient_id = f"{cohort}_{i+1:04d}"
        sex = random.choice(SEXES)
        ethnicity = random.choice(ETHNICITIES)
        race = np.random.choice(RACES, p=[0.6, 0.15, 0.1, 0.03, 0.02, 0.05, 0.05])
        
        # Add HEAVY age overlap: AD and controls very similar
        if cohort == 'AD':
            # Much wider age range - early onset AD
            age = int(np.clip(np.random.normal(73, 12), 50, 95))
        else:
            # Controls can be very old
            age = int(np.clip(np.random.normal(71, 12), 50, 95))
        
        birth_date = base_date - timedelta(days=age*365.25)
        
        # Death status
        if cohort == 'AD':
            death_status = random.choices(['Alive', 'Deceased'], weights=[0.75, 0.25])[0]
        else:
            death_status = random.choices(['Alive', 'Deceased'], weights=[0.9, 0.1])[0]
        
        # Social history
        if cohort == 'AD':
            # AD patients more likely to be former smokers, less educated
            smoking = np.random.choice(SMOKING_STATUS, p=[0.4, 0.35, 0.2, 0.05])
            education = np.random.choice(EDUCATION_LEVEL, p=[0.15, 0.35, 0.25, 0.15, 0.05, 0.05])
        else:
            smoking = np.random.choice(SMOKING_STATUS, p=[0.5, 0.3, 0.15, 0.05])
            education = np.random.choice(EDUCATION_LEVEL, p=[0.1, 0.3, 0.25, 0.2, 0.1, 0.05])
        
        alcohol = np.random.choice(ALCOHOL_USE, p=[0.3, 0.35, 0.25, 0.05, 0.05])
        marital = np.random.choice(MARITAL_STATUS, p=[0.5, 0.15, 0.2, 0.1, 0.05])
        
        demographics.append({
            'PatientID': patient_id,
            'Sex': sex,
            'Ethnicity': ethnicity,
            'Race': race,
            'BirthDate': birth_date.strftime('%Y-%m-%d'),
            'DeathStatus': death_status,
            'Age': age,
            'SmokingStatus': smoking,
            'AlcoholUse': alcohol,
            'EducationLevel': education,
            'MaritalStatus': marital,
            'OMOP_PersonID': f'P{100000+i}'
        })
    
    return pd.DataFrame(demographics)


def generate_diagnoses_with_dates(demographics_df, cohort):
    """Generate comprehensive diagnosis data with temporal patterns and realistic noise"""
    diagnoses = []
    base_date = datetime.now()
    
    for _, patient in demographics_df.iterrows():
        patient_id = patient['PatientID']
        age = patient['Age']
        
        # Diagnosis start date (1-10 years ago for AD, 1-5 for control)
        if cohort == 'AD':
            years_back = random.uniform(2, 10)
        else:
            years_back = random.uniform(1, 5)
        
        start_date = base_date - timedelta(days=years_back*365.25)
        
        if cohort == 'AD':
            # MUCH MORE NOISE: 30% of AD patients might not have AD diagnosis in records yet
            if random.random() > 0.30:
                n_ad = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
                ad_diags = random.sample(AD_DIAGNOSES, k=n_ad)
            else:
                ad_diags = []  # 30% have no AD diagnosis yet (early/undiagnosed/misdiagnosed)
            
            # Add risk factors with HEAVY noise (+/- 35%)
            risk_diags = []
            for risk in ADRD_RISK_FACTORS:
                noise = random.uniform(-0.35, 0.35)
                prevalence = np.clip(risk['prevalence_ad'] + noise, 0, 1)
                if random.random() < prevalence:
                    risk_diags.append(risk)
            
            patient_diagnoses = ad_diags + risk_diags
        else:
            # CRITICAL: MUCH MORE OVERLAP - 35% of controls have cognitive issues
            if random.random() < 0.35:
                mci_diag = [d for d in AD_DIAGNOSES if 'Mild cognitive impairment' in d['name'] or 'Memory impairment' in d['name']]
                if mci_diag:
                    # Some controls get full AD diagnoses (misdiagnosis, early detection)
                    if random.random() < 0.15:
                        patient_diagnoses = [random.choice(AD_DIAGNOSES)]
                    else:
                        patient_diagnoses = [random.choice(mci_diag)]
                else:
                    patient_diagnoses = []
            else:
                patient_diagnoses = []
            
            # Control patients: risk factors with HEAVY noise (+/- 35%)
            risk_diags = []
            for risk in ADRD_RISK_FACTORS:
                noise = random.uniform(-0.35, 0.35)
                # Some controls have very high prevalence of risk factors
                prevalence = np.clip(risk['prevalence_control'] + noise, 0.1, 1)
                if random.random() < prevalence:
                    risk_diags.append(risk)
            
            # Add other conditions with more noise
            other_diags = []
            for ctrl in CONTROL_DIAGNOSES:
                noise = random.uniform(-0.35, 0.35)
                prevalence = np.clip(ctrl['prevalence'] + noise, 0, 1)
                if random.random() < prevalence:
                    other_diags.append(ctrl)
            
            patient_diagnoses += risk_diags + other_diags
        
        # Generate temporal pattern for each diagnosis
        for diag in patient_diagnoses:
            # Number of occurrences
            n_occurrences = random.randint(2, 15)
            
            for j in range(n_occurrences):
                # Spread diagnoses over time
                days_offset = random.uniform(0, years_back*365.25)
                service_date = start_date + timedelta(days=days_offset)
                
                diagnoses.append({
                    'PatientID': patient_id,
                    'DateOfService': service_date.strftime('%Y-%m-%d'),
                    'FullDiagnosisName': diag['name'],
                    'ICD10_Code': diag['icd10'],
                    'SNOMED_Code': diag['snomed'],
                    'Level2_Category': diag['category'],
                    'Level3_Category': diag['subcategory'],
                    'OMOP_ConceptID': diag['omop_concept_id'],
                    'Severity': diag.get('severity', 'Moderate')
                })
    
    return pd.DataFrame(diagnoses).sort_values(['PatientID', 'DateOfService'])


def generate_medications_with_dates(demographics_df, cohort):
    """Generate comprehensive medication data with realistic noise"""
    medications = []
    base_date = datetime.now()
    
    for _, patient in demographics_df.iterrows():
        patient_id = patient['PatientID']
        
        if cohort == 'AD':
            years_back = random.uniform(2, 10)
        else:
            years_back = random.uniform(1, 5)
        
        start_date = base_date - timedelta(days=years_back*365.25)
        
        if cohort == 'AD':
            # MUCH MORE NOISE: only 35% have AD-specific medication
            if random.random() < 0.35:
                n_ad = random.choices([1, 2], weights=[0.8, 0.2])[0]
                ad_meds = random.sample(AD_MEDICATIONS, k=n_ad)
            else:
                ad_meds = []  # 65% don't have AD meds (undiagnosed, refuse treatment, early stage)
            
            # Common medications with high variability
            n_common = random.randint(0, 8)
            if n_common > 0:
                common_meds = random.sample(COMMON_MEDICATIONS, k=min(n_common, len(COMMON_MEDICATIONS)))
            else:
                common_meds = []
            
            patient_meds = ad_meds + common_meds
        else:
            # Control: 12% might have memory drugs (HIGH overlap)
            if random.random() < 0.12:
                n_ad = 1
                ad_meds = random.sample(AD_MEDICATIONS, k=n_ad)
            else:
                ad_meds = []
            
            n_common = random.randint(0, 7)  # Many have no meds
            if n_common > 0:
                patient_meds = ad_meds + random.sample(COMMON_MEDICATIONS, k=n_common)
            else:
                patient_meds = ad_meds
        
        for med in patient_meds:
            # Multiple prescriptions/refills over time
            n_orders = random.randint(3, 20)
            
            for j in range(n_orders):
                # Medication orders spread over time (typically monthly refills)
                days_offset = random.uniform(0, years_back*365.25)
                order_date = start_date + timedelta(days=days_offset)
                
                medications.append({
                    'PatientID': patient_id,
                    'DateOfService': order_date.strftime('%Y-%m-%d'),
                    'MedicationName': med['name'],
                    'MedicationGenericName': med['ingredient'],
                    'RxNorm_Code': med['rxnorm'],
                    'OMOP_ConceptID': med['omop_concept_id'],
                    'MedicationClass': med['class'],
                    'Route': med['route']
                })
    
    return pd.DataFrame(medications).sort_values(['PatientID', 'DateOfService'])


def generate_lab_results_with_dates(demographics_df, cohort):
    """Generate comprehensive lab results with realistic noise and missing data"""
    lab_results = []
    base_date = datetime.now()
    
    for _, patient in demographics_df.iterrows():
        patient_id = patient['PatientID']
        
        if cohort == 'AD':
            years_back = random.uniform(2, 10)
        else:
            years_back = random.uniform(1, 5)
        
        start_date = base_date - timedelta(days=years_back*365.25)
        
        # Select lab tests with HIGH variability and missing data
        if cohort == 'AD':
            n_tests = random.randint(5, 22)  # Much more variability, many have few tests
        else:
            n_tests = random.randint(3, 15)  # Even more variability
        
        patient_labs = random.sample(LAB_TESTS, k=min(n_tests, len(LAB_TESTS)))
        
        # Only 45% of AD patients get cognitive tests (MUCH missing data)
        if cohort == 'AD' and random.random() < 0.45:
            cognitive_tests = [t for t in LAB_TESTS if t['category'] == 'Cognitive']
            for test in cognitive_tests:
                if test not in patient_labs and random.random() < 0.4:  # Only 40% get the test
                    patient_labs.append(test)
        
        # 25% of controls also get cognitive tests (HIGH screening rate)
        if cohort == 'CTRL' and random.random() < 0.25:
            cognitive_tests = [t for t in LAB_TESTS if t['category'] == 'Cognitive']
            if cognitive_tests:
                # Some get multiple tests
                n_cog_tests = random.randint(1, 2)
                for _ in range(n_cog_tests):
                    if cognitive_tests:
                        test = random.choice(cognitive_tests)
                        if test not in patient_labs:
                            patient_labs.append(test)
        
        for test in patient_labs:
            # Multiple results over time with some missing data
            if test['category'] in ['Cognitive', 'Genetic', 'AD Biomarker']:
                n_results = random.randint(0, 3)  # Some patients have 0 results (missing data)
            else:
                n_results = random.randint(1, 10)
            
            for j in range(n_results):
                days_offset = random.uniform(0, years_back*365.25)
                test_date = start_date + timedelta(days=days_offset)
                
                # Generate realistic values with noise and overlap
                if test['normal_range']:
                    mean_val = np.mean(test['normal_range'])
                    std_val = (test['normal_range'][1] - test['normal_range'][0]) / 4
                    
                    # Add HEAVY noise and HIGH overlap between groups
                    if cohort == 'AD':
                        if test['name'] in ['Mini-Mental State Examination', 'Montreal Cognitive Assessment']:
                            # HIGH overlap: many AD patients score well (early stage, high education)
                            mean_val = test['normal_range'][0] + (test['normal_range'][1] - test['normal_range'][0]) * random.uniform(0.3, 0.75)
                            std_val *= 2.0  # Much more variability
                        elif test['name'] in ['Hemoglobin A1c', 'Homocysteine', 'C-reactive protein']:
                            # Many AD patients have normal values
                            mean_val = mean_val + random.uniform(-std_val*1.5, std_val*2.5)
                        elif test['name'] == 'Beta-amyloid 42, CSF':
                            # 40% of AD patients have normal biomarkers
                            mean_val = test['normal_range'][0] + (test['normal_range'][1] - test['normal_range'][0]) * random.uniform(0.3, 0.9)
                    else:
                        # Control patients often have abnormal values (aging, comorbidities)
                        if test['name'] in ['Mini-Mental State Examination', 'Montreal Cognitive Assessment']:
                            # Many controls have impairment (aging, education)
                            mean_val = test['normal_range'][0] + (test['normal_range'][1] - test['normal_range'][0]) * random.uniform(0.5, 0.92)
                        else:
                            # Controls can have metabolic abnormalities
                            mean_val = mean_val + random.uniform(-std_val, std_val*1.5)
                    
                    # Add HIGH measurement error/noise
                    value = np.random.normal(mean_val, std_val * 1.8)  # Much more variability
                    value = np.clip(value, test['normal_range'][0] * 0.2, test['normal_range'][1] * 2.2)
                    
                    if test['unit']:
                        result = f"{value:.2f} {test['unit']}"
                    else:
                        result = f"{value:.2f}"
                else:
                    # Qualitative results with noise
                    if cohort == 'AD':
                        # 60% abnormal (not 100%)
                        if random.random() < 0.6:
                            result = random.choice(['Abnormal', 'Borderline'])
                        else:
                            result = 'Normal'
                    else:
                        # 20% of controls have abnormal results
                        if random.random() < 0.2:
                            result = random.choice(['Abnormal', 'Borderline'])
                        else:
                            result = random.choice(['Normal', 'Within normal limits'])
                
                lab_results.append({
                    'PatientID': patient_id,
                    'DateOfService': test_date.strftime('%Y-%m-%d'),
                    'TestName': test['name'],
                    'TestResult': result,
                    'LOINC_Code': test['loinc'],
                    'OMOP_ConceptID': test['omop_concept_id'],
                    'Category': test['category']
                })
    
    return pd.DataFrame(lab_results).sort_values(['PatientID', 'DateOfService'])


def generate_imaging_procedures(demographics_df, cohort):
    """Generate imaging procedure records"""
    procedures = []
    base_date = datetime.now()
    
    for _, patient in demographics_df.iterrows():
        patient_id = patient['PatientID']
        
        if cohort == 'AD':
            years_back = random.uniform(2, 10)
            # AD patients get more neuroimaging
            n_procedures = random.randint(3, 8)
            neuro_weight = 0.8
        else:
            years_back = random.uniform(1, 5)
            n_procedures = random.randint(1, 4)
            neuro_weight = 0.3
        
        start_date = base_date - timedelta(days=years_back*365.25)
        
        # Select procedures
        neuro_procedures = [p for p in IMAGING_PROCEDURES if p['category'] == 'Neuroimaging']
        other_procedures = [p for p in IMAGING_PROCEDURES if p['category'] != 'Neuroimaging']
        
        patient_procedures = []
        for _ in range(n_procedures):
            if random.random() < neuro_weight and neuro_procedures:
                patient_procedures.append(random.choice(neuro_procedures))
            elif other_procedures:
                patient_procedures.append(random.choice(other_procedures))
        
        for proc in patient_procedures:
            days_offset = random.uniform(0, years_back*365.25)
            proc_date = start_date + timedelta(days=days_offset)
            
            procedures.append({
                'PatientID': patient_id,
                'DateOfService': proc_date.strftime('%Y-%m-%d'),
                'ProcedureName': proc['name'],
                'CPT_Code': proc['cpt'],
                'SNOMED_Code': proc['snomed'],
                'OMOP_ConceptID': proc['omop_concept_id'],
                'Category': proc['category']
            })
    
    return pd.DataFrame(procedures).sort_values(['PatientID', 'DateOfService'])


def generate_treatment_procedures(demographics_df, cohort):
    """Generate treatment/therapy procedure records"""
    procedures = []
    base_date = datetime.now()
    
    for _, patient in demographics_df.iterrows():
        patient_id = patient['PatientID']
        
        if cohort == 'AD':
            years_back = random.uniform(2, 10)
            # AD patients get more cognitive therapies
            n_procedures = random.randint(5, 15)
        else:
            years_back = random.uniform(1, 5)
            n_procedures = random.randint(1, 5)
        
        start_date = base_date - timedelta(days=years_back*365.25)
        
        # Select appropriate procedures
        if cohort == 'AD':
            # More cognitive/rehab procedures
            weights = [0.2, 0.15, 0.15, 0.15, 0.1, 0.15, 0.1]
        else:
            weights = [0.2, 0.2, 0.1, 0.05, 0.05, 0.05, 0.35]
        
        patient_procedures = np.random.choice(
            TREATMENT_PROCEDURES, 
            size=n_procedures, 
            replace=True,
            p=weights
        )
        
        for proc in patient_procedures:
            days_offset = random.uniform(0, years_back*365.25)
            proc_date = start_date + timedelta(days=days_offset)
            
            procedures.append({
                'PatientID': patient_id,
                'DateOfService': proc_date.strftime('%Y-%m-%d'),
                'ProcedureName': proc['name'],
                'CPT_Code': proc['cpt'],
                'SNOMED_Code': proc['snomed'],
                'OMOP_ConceptID': proc['omop_concept_id'],
                'Category': proc['category']
            })
    
    return pd.DataFrame(procedures).sort_values(['PatientID', 'DateOfService'])


def generate_vital_signs(demographics_df, cohort):
    """Generate vital signs records"""
    vitals = []
    base_date = datetime.now()
    
    for _, patient in demographics_df.iterrows():
        patient_id = patient['PatientID']
        age = patient['Age']
        
        if cohort == 'AD':
            years_back = random.uniform(2, 10)
            n_visits = random.randint(10, 30)
        else:
            years_back = random.uniform(1, 5)
            n_visits = random.randint(5, 15)
        
        start_date = base_date - timedelta(days=years_back*365.25)
        
        # Generate baseline patient characteristics
        baseline_sbp = np.random.normal(130 if cohort == 'AD' else 125, 15)
        baseline_dbp = np.random.normal(80, 10)
        baseline_hr = np.random.normal(75, 10)
        baseline_weight = np.random.normal(75, 15)
        baseline_height = np.random.normal(170, 10)
        
        for visit in range(n_visits):
            days_offset = random.uniform(0, years_back*365.25)
            visit_date = start_date + timedelta(days=days_offset)
            
            # Generate vital signs with some variation
            for vital in VITAL_SIGNS:
                if vital['name'] == 'Systolic Blood Pressure':
                    value = baseline_sbp + np.random.normal(0, 10)
                elif vital['name'] == 'Diastolic Blood Pressure':
                    value = baseline_dbp + np.random.normal(0, 8)
                elif vital['name'] == 'Heart Rate':
                    value = baseline_hr + np.random.normal(0, 8)
                elif vital['name'] == 'Body Weight':
                    value = baseline_weight + np.random.normal(0, 2)
                elif vital['name'] == 'Height':
                    value = baseline_height
                elif vital['name'] == 'BMI':
                    value = baseline_weight / ((baseline_height/100) ** 2)
                else:
                    mean_val = np.mean(vital['normal_range'])
                    std_val = (vital['normal_range'][1] - vital['normal_range'][0]) / 6
                    value = np.random.normal(mean_val, std_val)
                
                vitals.append({
                    'PatientID': patient_id,
                    'DateOfService': visit_date.strftime('%Y-%m-%d'),
                    'VitalSign': vital['name'],
                    'Value': f"{value:.1f} {vital['unit']}",
                    'LOINC_Code': vital['loinc']
                })
    
    return pd.DataFrame(vitals).sort_values(['PatientID', 'DateOfService'])


def main():
    """Generate comprehensive ontology-aligned data"""
    print("="*80)
    print("COMPREHENSIVE ONTOLOGY-ALIGNED EHR DATA GENERATION")
    print("Following OntoCodex Framework & OMOP CDM Standards")
    print("="*80)
    print()
    
    # Generate AD cohort
    print(f"Generating AD cohort ({N_AD_PATIENTS} patients)...")
    ad_demographics = generate_demographics(N_AD_PATIENTS, 'AD')
    ad_diagnoses = generate_diagnoses_with_dates(ad_demographics, 'AD')
    ad_medications = generate_medications_with_dates(ad_demographics, 'AD')
    ad_labs = generate_lab_results_with_dates(ad_demographics, 'AD')
    ad_imaging = generate_imaging_procedures(ad_demographics, 'AD')
    ad_treatments = generate_treatment_procedures(ad_demographics, 'AD')
    ad_vitals = generate_vital_signs(ad_demographics, 'AD')
    
    # Generate Control cohort
    print(f"Generating Control cohort ({N_CONTROL_PATIENTS} patients)...")
    control_demographics = generate_demographics(N_CONTROL_PATIENTS, 'CTRL')
    control_diagnoses = generate_diagnoses_with_dates(control_demographics, 'CTRL')
    control_medications = generate_medications_with_dates(control_demographics, 'CTRL')
    control_labs = generate_lab_results_with_dates(control_demographics, 'CTRL')
    control_imaging = generate_imaging_procedures(control_demographics, 'CTRL')
    control_treatments = generate_treatment_procedures(control_demographics, 'CTRL')
    control_vitals = generate_vital_signs(control_demographics, 'CTRL')
    
    # Save to CSV files
    print("\nSaving to CSV files...")
    data_dir = 'Data'
    
    # AD Cohort
    ad_demographics.to_csv(f'{data_dir}/ad_demographics.csv', index=False)
    ad_diagnoses.to_csv(f'{data_dir}/ad_diagnosis.csv', index=False)
    ad_medications.to_csv(f'{data_dir}/ad_medications.csv', index=False)
    ad_labs.to_csv(f'{data_dir}/ad_labresults.csv', index=False)
    ad_imaging.to_csv(f'{data_dir}/ad_imaging.csv', index=False)
    ad_treatments.to_csv(f'{data_dir}/ad_treatments.csv', index=False)
    ad_vitals.to_csv(f'{data_dir}/ad_vitals.csv', index=False)
    
    # Control Cohort
    control_demographics.to_csv(f'{data_dir}/control_demographics.csv', index=False)
    control_diagnoses.to_csv(f'{data_dir}/control_diagnosis.csv', index=False)
    control_medications.to_csv(f'{data_dir}/control_medications.csv', index=False)
    control_labs.to_csv(f'{data_dir}/control_labresults.csv', index=False)
    control_imaging.to_csv(f'{data_dir}/control_imaging.csv', index=False)
    control_treatments.to_csv(f'{data_dir}/control_treatments.csv', index=False)
    control_vitals.to_csv(f'{data_dir}/control_vitals.csv', index=False)
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("DATA GENERATION SUMMARY")
    print("="*80)
    
    print(f"\n🔵 AD COHORT (n={N_AD_PATIENTS}):")
    print(f"  Demographics:        {len(ad_demographics):,} patients")
    print(f"  Diagnoses:           {len(ad_diagnoses):,} records")
    print(f"  Medications:         {len(ad_medications):,} orders")
    print(f"  Lab Results:         {len(ad_labs):,} tests")
    print(f"  Imaging Procedures:  {len(ad_imaging):,} procedures")
    print(f"  Treatment Procedures:{len(ad_treatments):,} sessions")
    print(f"  Vital Signs:         {len(ad_vitals):,} measurements")
    print(f"  📊 Total Records:    {len(ad_diagnoses) + len(ad_medications) + len(ad_labs) + len(ad_imaging) + len(ad_treatments) + len(ad_vitals):,}")
    
    print(f"\n🟢 CONTROL COHORT (n={N_CONTROL_PATIENTS}):")
    print(f"  Demographics:        {len(control_demographics):,} patients")
    print(f"  Diagnoses:           {len(control_diagnoses):,} records")
    print(f"  Medications:         {len(control_medications):,} orders")
    print(f"  Lab Results:         {len(control_labs):,} tests")
    print(f"  Imaging Procedures:  {len(control_imaging):,} procedures")
    print(f"  Treatment Procedures:{len(control_treatments):,} sessions")
    print(f"  Vital Signs:         {len(control_vitals):,} measurements")
    print(f"  📊 Total Records:    {len(control_diagnoses) + len(control_medications) + len(control_labs) + len(control_imaging) + len(control_treatments) + len(control_vitals):,}")
    
    total_records = (len(ad_diagnoses) + len(ad_medications) + len(ad_labs) + len(ad_imaging) + len(ad_treatments) + len(ad_vitals) +
                    len(control_diagnoses) + len(control_medications) + len(control_labs) + len(control_imaging) + len(control_treatments) + len(control_vitals))
    
    print(f"\n📈 GRAND TOTAL:      {total_records:,} clinical records")
    
    # Save comprehensive metadata
    metadata = {
        'generation_date': datetime.now().isoformat(),
        'framework': 'OntoCodex-aligned',
        'version': '2.0-Comprehensive',
        'standards': {
            'CDM': 'OMOP CDM v5',
            'terminologies': ['SNOMED CT', 'ICD-10', 'RxNorm', 'LOINC', 'CPT'],
            'ontology': 'MCC-CDO aligned'
        },
        'cohorts': {
            'AD': {'n_patients': N_AD_PATIENTS, 'total_records': len(ad_diagnoses) + len(ad_medications) + len(ad_labs) + len(ad_imaging) + len(ad_treatments) + len(ad_vitals)},
            'Control': {'n_patients': N_CONTROL_PATIENTS, 'total_records': len(control_diagnoses) + len(control_medications) + len(control_labs) + len(control_imaging) + len(control_treatments) + len(control_vitals)}
        },
        'data_types': ['Demographics', 'Diagnoses', 'Medications', 'Lab Results', 'Imaging', 'Treatments', 'Vital Signs'],
        'ad_diagnoses_count': len(AD_DIAGNOSES),
        'risk_factors_count': len(ADRD_RISK_FACTORS),
        'medications_count': len(AD_MEDICATIONS) + len(COMMON_MEDICATIONS),
        'lab_tests_count': len(LAB_TESTS),
        'imaging_procedures_count': len(IMAGING_PROCEDURES),
        'treatment_procedures_count': len(TREATMENT_PROCEDURES)
    }
    
    with open(f'{data_dir}/comprehensive_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ COMPREHENSIVE DATA GENERATION COMPLETE!")
    print("="*80)
    print("\n📋 Features:")
    print("  ✓ All records include DateOfService (temporal patterns)")
    print("  ✓ OMOP CDM v5 aligned")
    print("  ✓ Standard medical codes (SNOMED CT, ICD-10, RxNorm, LOINC, CPT)")
    print("  ✓ Social history included")
    print("  ✓ Comprehensive clinical data (7 data types)")
    print("  ✓ Realistic ADRD risk factor distributions")
    print("  ✓ AD-specific biomarkers and cognitive assessments")
    print("  ✓ Ready for knowledge graph construction")
    print("="*80)


if __name__ == "__main__":
    main()

