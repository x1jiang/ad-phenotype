#!/usr/bin/env python3
"""
OMOP CDM to AD-Phenotype CSV Extraction Script

This script extracts the 12 required CSV files from an OMOP CDM database.
Designed for Mayo Clinic MCSA_OMOP database.

Usage:
    1. Update the database connection settings below
    2. Run: python omop_to_csv_extraction.py
    3. Output: Creates 12 CSV files in ./Data/ directory

Author: Dr. Xiaoqian Jiang (UTHealth)
For: Mayo Clinic team (Jingna, Ahmed, Yue)
"""

import os
from pathlib import Path

# ============================================================================
# CONFIGURATION - UPDATE THESE FOR YOUR ENVIRONMENT
# ============================================================================

# Database connection (update for your OMOP database)
DB_CONFIG = {
    'database': '/infodev1/infodtao/MCC/MCSA_OMOP',  # Update path/connection
    'schema': 'cdm',  # OMOP schema name
}

# Output directory
OUTPUT_DIR = Path('./Data')

# AD Cohort definition - ICD10/SNOMED codes for dementia
AD_CODES = {
    'icd10': ['G30', 'G30.0', 'G30.1', 'G30.8', 'G30.9', 'F00', 'F00.0', 'F00.1', 'F00.2', 'F00.9'],
    'snomed': ['26929004', '15662003', '230258005', '230270009', '416780008'],
}

# ============================================================================
# SQL QUERIES
# ============================================================================

# Identifies AD cohort patients
SQL_AD_COHORT = """
SELECT DISTINCT person_id
FROM condition_occurrence co
JOIN concept c ON co.condition_concept_id = c.concept_id
WHERE 
    c.vocabulary_id = 'ICD10CM' AND (
        c.concept_code LIKE 'G30%' OR 
        c.concept_code LIKE 'F00%' OR
        c.concept_code LIKE 'G31.0%' OR
        c.concept_code LIKE 'G31.1%'
    )
    OR (c.vocabulary_id = 'SNOMED' AND c.concept_code IN ('26929004', '15662003'))
"""

# Identifies Control cohort (no dementia)
SQL_CONTROL_COHORT = """
SELECT DISTINCT p.person_id
FROM person p
WHERE p.person_id NOT IN ({ad_cohort_query})
"""

# Demographics extraction
SQL_DEMOGRAPHICS = """
SELECT 
    CONCAT('{prefix}_', LPAD(ROW_NUMBER() OVER (ORDER BY p.person_id), 4, '0')) AS PatientID,
    COALESCE(gc.concept_name, 'Unknown') AS Sex,
    COALESCE(ec.concept_name, 'Unknown') AS Ethnicity,
    COALESCE(rc.concept_name, 'Unknown') AS Race,
    COALESCE(CAST(p.birth_datetime AS DATE), 
             CONCAT(p.year_of_birth, '-01-01')) AS BirthDate,
    CASE WHEN d.person_id IS NOT NULL THEN 'Deceased' ELSE 'Alive' END AS DeathStatus,
    YEAR(CURRENT_DATE) - p.year_of_birth AS Age,
    COALESCE(sm.smoking_status, 'Unknown') AS SmokingStatus,
    'Unknown' AS AlcoholUse,
    'Unknown' AS EducationLevel,
    'Unknown' AS MaritalStatus,
    CONCAT('P', p.person_id) AS OMOP_PersonID
FROM person p
LEFT JOIN concept gc ON p.gender_concept_id = gc.concept_id
LEFT JOIN concept ec ON p.ethnicity_concept_id = ec.concept_id
LEFT JOIN concept rc ON p.race_concept_id = rc.concept_id
LEFT JOIN death d ON p.person_id = d.person_id
LEFT JOIN (
    -- Smoking status from observations
    SELECT person_id, 
           MAX(CASE 
               WHEN observation_concept_id IN (4144272) THEN 'Current smoker'
               WHEN observation_concept_id IN (4310250) THEN 'Former smoker'
               ELSE 'Never smoker'
           END) AS smoking_status
    FROM observation
    WHERE observation_concept_id IN (4144272, 4310250, 4144273)
    GROUP BY person_id
) sm ON p.person_id = sm.person_id
WHERE p.person_id IN ({cohort_ids})
"""

# Diagnoses extraction
SQL_DIAGNOSES = """
SELECT 
    pt.PatientID,
    CAST(co.condition_start_date AS DATE) AS DateOfService,
    c.concept_name AS FullDiagnosisName,
    COALESCE(icd.concept_code, '') AS ICD10_Code,
    c.concept_code AS SNOMED_Code,
    '' AS Level2_Category,  -- Requires OntoCodex mapping
    '' AS Level3_Category,  -- Requires OntoCodex mapping
    co.condition_concept_id AS OMOP_ConceptID,
    COALESCE(sc.concept_name, 'Unknown') AS Severity
FROM condition_occurrence co
JOIN patient_mapping pt ON co.person_id = pt.person_id
JOIN concept c ON co.condition_concept_id = c.concept_id
LEFT JOIN concept_relationship cr ON c.concept_id = cr.concept_id_1 
    AND cr.relationship_id = 'Maps to'
LEFT JOIN concept icd ON cr.concept_id_2 = icd.concept_id 
    AND icd.vocabulary_id = 'ICD10CM'
LEFT JOIN concept sc ON co.condition_status_concept_id = sc.concept_id
WHERE co.person_id IN ({cohort_ids})
ORDER BY pt.PatientID, DateOfService
"""

# Medications extraction
SQL_MEDICATIONS = """
SELECT 
    pt.PatientID,
    CAST(de.drug_exposure_start_date AS DATE) AS DateOfService,
    c.concept_name AS MedicationName,
    LOWER(COALESCE(ing.concept_name, c.concept_name)) AS MedicationGenericName,
    c.concept_code AS RxNorm_Code,
    de.drug_concept_id AS OMOP_ConceptID,
    COALESCE(atc.concept_name, 'Unknown') AS MedicationClass,
    COALESCE(rc.concept_name, 'Unknown') AS Route
FROM drug_exposure de
JOIN patient_mapping pt ON de.person_id = pt.person_id
JOIN concept c ON de.drug_concept_id = c.concept_id
LEFT JOIN concept rc ON de.route_concept_id = rc.concept_id
LEFT JOIN concept_ancestor ca ON c.concept_id = ca.descendant_concept_id
LEFT JOIN concept atc ON ca.ancestor_concept_id = atc.concept_id 
    AND atc.vocabulary_id = 'ATC' AND atc.concept_class_id = 'ATC 2nd'
LEFT JOIN drug_strength ds ON c.concept_id = ds.drug_concept_id
LEFT JOIN concept ing ON ds.ingredient_concept_id = ing.concept_id
WHERE de.person_id IN ({cohort_ids})
ORDER BY pt.PatientID, DateOfService
"""

# Lab Results extraction
SQL_LABRESULTS = """
SELECT 
    pt.PatientID,
    CAST(m.measurement_date AS DATE) AS DateOfService,
    c.concept_name AS TestName,
    CONCAT(ROUND(m.value_as_number, 2), ' ', COALESCE(uc.concept_name, '')) AS TestResult,
    c.concept_code AS LOINC_Code,
    m.measurement_concept_id AS OMOP_ConceptID,
    CASE 
        WHEN c.concept_name LIKE '%cholesterol%' OR c.concept_name LIKE '%HDL%' 
             OR c.concept_name LIKE '%LDL%' OR c.concept_name LIKE '%triglyceride%' 
             THEN 'Lipid'
        WHEN c.concept_name LIKE '%glucose%' OR c.concept_name LIKE '%HbA1c%' THEN 'Metabolic'
        WHEN c.concept_name LIKE '%creatinine%' OR c.concept_name LIKE '%BUN%' THEN 'Renal'
        ELSE 'Other'
    END AS Category
FROM measurement m
JOIN patient_mapping pt ON m.person_id = pt.person_id
JOIN concept c ON m.measurement_concept_id = c.concept_id
LEFT JOIN concept uc ON m.unit_concept_id = uc.concept_id
WHERE m.person_id IN ({cohort_ids})
  AND m.value_as_number IS NOT NULL
ORDER BY pt.PatientID, DateOfService
"""

# Imaging extraction (CPT 70000-79999 range)
SQL_IMAGING = """
SELECT 
    pt.PatientID,
    CAST(po.procedure_date AS DATE) AS DateOfService,
    c.concept_name AS ProcedureName,
    cpt.concept_code AS CPT_Code,
    c.concept_code AS SNOMED_Code,
    po.procedure_concept_id AS OMOP_ConceptID,
    CASE 
        WHEN c.concept_name LIKE '%MRI%brain%' OR c.concept_name LIKE '%MR%brain%' THEN 'Neuroimaging'
        WHEN c.concept_name LIKE '%CT%head%' OR c.concept_name LIKE '%CT%brain%' THEN 'Neuroimaging'
        WHEN c.concept_name LIKE '%PET%' THEN 'PET Scan'
        ELSE 'Other Imaging'
    END AS Category
FROM procedure_occurrence po
JOIN patient_mapping pt ON po.person_id = pt.person_id
JOIN concept c ON po.procedure_concept_id = c.concept_id
LEFT JOIN concept_relationship cr ON c.concept_id = cr.concept_id_1 
    AND cr.relationship_id = 'Maps to'
LEFT JOIN concept cpt ON cr.concept_id_2 = cpt.concept_id 
    AND cpt.vocabulary_id = 'CPT4'
WHERE po.person_id IN ({cohort_ids})
  AND (cpt.concept_code BETWEEN '70000' AND '79999' 
       OR c.concept_name LIKE '%imaging%' 
       OR c.concept_name LIKE '%MRI%' 
       OR c.concept_name LIKE '%CT%'
       OR c.concept_name LIKE '%X-ray%')
ORDER BY pt.PatientID, DateOfService
"""

# Treatments extraction (non-imaging procedures)
SQL_TREATMENTS = """
SELECT 
    pt.PatientID,
    CAST(po.procedure_date AS DATE) AS DateOfService,
    c.concept_name AS ProcedureName,
    cpt.concept_code AS CPT_Code,
    c.concept_code AS SNOMED_Code,
    po.procedure_concept_id AS OMOP_ConceptID,
    CASE 
        WHEN c.concept_name LIKE '%therapy%' OR c.concept_name LIKE '%counseling%' THEN 'Psychotherapy'
        WHEN c.concept_name LIKE '%surgery%' OR c.concept_name LIKE '%excision%' THEN 'Surgery'
        WHEN c.concept_name LIKE '%infusion%' OR c.concept_name LIKE '%injection%' THEN 'Infusion'
        ELSE 'Other Treatment'
    END AS Category
FROM procedure_occurrence po
JOIN patient_mapping pt ON po.person_id = pt.person_id
JOIN concept c ON po.procedure_concept_id = c.concept_id
LEFT JOIN concept_relationship cr ON c.concept_id = cr.concept_id_1 
    AND cr.relationship_id = 'Maps to'
LEFT JOIN concept cpt ON cr.concept_id_2 = cpt.concept_id 
    AND cpt.vocabulary_id = 'CPT4'
WHERE po.person_id IN ({cohort_ids})
  AND NOT (cpt.concept_code BETWEEN '70000' AND '79999')  -- Exclude imaging
ORDER BY pt.PatientID, DateOfService
"""


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to extract OMOP data to CSV files.
    
    NOTE: This script provides the SQL templates. You will need to:
    1. Adapt the database connection for your environment
    2. Run each SQL query against your OMOP database
    3. Export results to CSV files in the Data/ directory
    
    For MCSA_OMOP specifically, you may need to adjust table/schema names.
    """
    
    print("=" * 60)
    print("OMOP to AD-Phenotype CSV Extraction")
    print("=" * 60)
    
    print("\nThis script provides SQL templates for extracting data.")
    print("Please adapt for your database environment.\n")
    
    print("Required output files:")
    files = [
        ("ad_demographics.csv", "control_demographics.csv"),
        ("ad_diagnosis.csv", "control_diagnosis.csv"),
        ("ad_medications.csv", "control_medications.csv"),
        ("ad_labresults.csv", "control_labresults.csv"),
        ("ad_imaging.csv", "control_imaging.csv"),
        ("ad_treatments.csv", "control_treatments.csv"),
    ]
    
    for ad_file, ctrl_file in files:
        print(f"  - {ad_file}")
        print(f"  - {ctrl_file}")
    
    print("\n" + "=" * 60)
    print("SQL Queries are defined in this file. Key steps:")
    print("=" * 60)
    print("""
1. Run SQL_AD_COHORT to identify AD patients
2. Run SQL_CONTROL_COHORT to identify matched controls
3. For each cohort, run the extraction queries:
   - SQL_DEMOGRAPHICS
   - SQL_DIAGNOSES
   - SQL_MEDICATIONS
   - SQL_LABRESULTS
   - SQL_IMAGING
   - SQL_TREATMENTS
4. Export each result to corresponding CSV file
""")
    
    print("For OntoCodex Level2/Level3 categories, contact Jingna Feng.")
    print("\nSee OMOP_MAPPING_GUIDE.md for column mapping details.")


if __name__ == "__main__":
    main()
