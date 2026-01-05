# OMOP CDM to AD-Phenotype CSV Mapping Guide

**Purpose**: Map `MCSA_OMOP` database to the 12 required CSV files

---

## Quick Overview

The pipeline requires **12 CSV files** (6 AD cohort + 6 Control cohort):

| File Type    | AD Cohort               | Control Cohort               |
| ------------ | ----------------------- | ---------------------------- |
| Demographics | `ad_demographics.csv` | `control_demographics.csv` |
| Diagnoses    | `ad_diagnosis.csv`    | `control_diagnosis.csv`    |
| Medications  | `ad_medications.csv`  | `control_medications.csv`  |
| Lab Results  | `ad_labresults.csv`   | `control_labresults.csv`   |
| Imaging      | `ad_imaging.csv`      | `control_imaging.csv`      |
| Treatments   | `ad_treatments.csv`   | `control_treatments.csv`   |

---

## Column Mappings from OMOP CDM v5

### 1. Demographics (`ad_demographics.csv`, `control_demographics.csv`)

| CSV Column         | OMOP Table      | OMOP Column                                           | Notes                                |
| ------------------ | --------------- | ----------------------------------------------------- | ------------------------------------ |
| `PatientID`      | `person`      | `person_id`                                         | Prefix with "AD_" or "CON_"          |
| `Sex`            | `person`      | `gender_concept_id`                                 | Join with `concept` table for name |
| `Ethnicity`      | `person`      | `ethnicity_concept_id`                              | Join with `concept`                |
| `Race`           | `person`      | `race_concept_id`                                   | Join with `concept`                |
| `BirthDate`      | `person`      | `birth_datetime` or computed from `year_of_birth` |                                      |
| `DeathStatus`    | `death`       | Check existence                                       | "Deceased" if record exists          |
| `Age`            | Computed        | `YEAR(CURRENT_DATE) - year_of_birth`                |                                      |
| `SmokingStatus`  | `observation` | `observation_concept_id`                            | SNOMED: 77176002, 8517006, etc.      |
| `AlcoholUse`     | `observation` | `observation_concept_id`                            | SNOMED: 228273003                    |
| `EducationLevel` | `observation` | `observation_concept_id`                            | Optional                             |
| `MaritalStatus`  | `observation` | `observation_concept_id`                            | Optional                             |
| `OMOP_PersonID`  | `person`      | `person_id`                                         | Original OMOP ID                     |

---

### 2. Diagnoses (`ad_diagnosis.csv`, `control_diagnosis.csv`)

| CSV Column            | OMOP Table               | OMOP Column                                            | Notes                            |
| --------------------- | ------------------------ | ------------------------------------------------------ | -------------------------------- |
| `PatientID`         | `condition_occurrence` | `person_id`                                          |                                  |
| `DateOfService`     | `condition_occurrence` | `condition_start_date`                               |                                  |
| `FullDiagnosisName` | `concept`              | `concept_name`                                       | Join on `condition_concept_id` |
| `ICD10_Code`        | `concept_relationship` | Map via `concept_code` where vocabulary_id='ICD10CM' |                                  |
| `SNOMED_Code`       | `concept`              | `concept_code`                                       | Where vocabulary_id='SNOMED'     |
| `Level2_Category`   | Manual mapping           | See OntoCodex ontology categories                      |                                  |
| `Level3_Category`   | Manual mapping           | See OntoCodex ontology categories                      |                                  |
| `OMOP_ConceptID`    | `condition_occurrence` | `condition_concept_id`                               |                                  |
| `Severity`          | `condition_occurrence` | `condition_status_concept_id`                        | Or derive from modifiers         |

---

### 3. Medications (`ad_medications.csv`, `control_medications.csv`)

| CSV Column                | OMOP Table           | OMOP Column                                | Notes                        |
| ------------------------- | -------------------- | ------------------------------------------ | ---------------------------- |
| `PatientID`             | `drug_exposure`    | `person_id`                              |                              |
| `DateOfService`         | `drug_exposure`    | `drug_exposure_start_date`               |                              |
| `MedicationName`        | `concept`          | `concept_name`                           | Join on `drug_concept_id`  |
| `MedicationGenericName` | `concept`          | Lower-cased concept_name or via ingredient |                              |
| `RxNorm_Code`           | `concept`          | `concept_code`                           | Where vocabulary_id='RxNorm' |
| `OMOP_ConceptID`        | `drug_exposure`    | `drug_concept_id`                        |                              |
| `MedicationClass`       | `concept_ancestor` | Join to ATC class                          |                              |
| `Route`                 | `drug_exposure`    | `route_concept_id` → concept_name       |                              |

---

### 4. Lab Results (`ad_labresults.csv`, `control_labresults.csv`)

| CSV Column         | OMOP Table      | OMOP Column                                       | Notes                              |
| ------------------ | --------------- | ------------------------------------------------- | ---------------------------------- |
| `PatientID`      | `measurement` | `person_id`                                     |                                    |
| `DateOfService`  | `measurement` | `measurement_date`                              |                                    |
| `TestName`       | `concept`     | `concept_name`                                  | Join on `measurement_concept_id` |
| `TestResult`     | `measurement` | `value_as_number` + `unit_concept_id`         | Format as "55.49 mg/dL"            |
| `LOINC_Code`     | `concept`     | `concept_code`                                  | Where vocabulary_id='LOINC'        |
| `OMOP_ConceptID` | `measurement` | `measurement_concept_id`                        |                                    |
| `Category`       | Manual mapping  | Based on LOINC hierarchy (Lipid, Metabolic, etc.) |                                    |

---

### 5. Imaging (`ad_imaging.csv`, `control_imaging.csv`)

| CSV Column         | OMOP Table               | OMOP Column                               | Notes                            |
| ------------------ | ------------------------ | ----------------------------------------- | -------------------------------- |
| `PatientID`      | `procedure_occurrence` | `person_id`                             |                                  |
| `DateOfService`  | `procedure_occurrence` | `procedure_date`                        |                                  |
| `ProcedureName`  | `concept`              | `concept_name`                          | Join on `procedure_concept_id` |
| `CPT_Code`       | `concept`              | `concept_code`                          | Where vocabulary_id='CPT4'       |
| `SNOMED_Code`    | `concept_relationship` | Map via SNOMED vocabulary                 |                                  |
| `OMOP_ConceptID` | `procedure_occurrence` | `procedure_concept_id`                  |                                  |
| `Category`       | Filter                   | Only imaging procedures (CPT 70000-79999) |                                  |

---

### 6. Treatments (`ad_treatments.csv`, `control_treatments.csv`)

| CSV Column         | OMOP Table               | OMOP Column                                     | Notes                      |
| ------------------ | ------------------------ | ----------------------------------------------- | -------------------------- |
| `PatientID`      | `procedure_occurrence` | `person_id`                                   |                            |
| `DateOfService`  | `procedure_occurrence` | `procedure_date`                              |                            |
| `ProcedureName`  | `concept`              | `concept_name`                                |                            |
| `CPT_Code`       | `concept`              | `concept_code`                                | Where vocabulary_id='CPT4' |
| `SNOMED_Code`    | `concept_relationship` | Map via SNOMED                                  |                            |
| `OMOP_ConceptID` | `procedure_occurrence` | `procedure_concept_id`                        |                            |
| `Category`       | Filter                   | Non-imaging procedures (therapy, surgery, etc.) |                            |

---

## AD vs Control Cohort Identification

**AD Cohort**: Patients with any of these conditions:

- ICD10: G30.x (Alzheimer's), F00.x (Dementia in AD), G31.x (Other dementias)
- SNOMED: 26929004 (Alzheimer's disease)
- OMOP Concept IDs: 378419, 4182210, etc.

**Control Cohort**: Matched patients WITHOUT dementia diagnoses, matched on:

- Age (±2 years)
- Sex
- Race/Ethnicity

---

## Sample SQL - Demographics

```sql
SELECT 
    CONCAT('AD_', LPAD(ROW_NUMBER() OVER (ORDER BY p.person_id), 4, '0')) AS PatientID,
    COALESCE(gc.concept_name, 'Unknown') AS Sex,
    COALESCE(ec.concept_name, 'Unknown') AS Ethnicity,
    COALESCE(rc.concept_name, 'Unknown') AS Race,
    COALESCE(p.birth_datetime, CONCAT(p.year_of_birth, '-01-01')) AS BirthDate,
    CASE WHEN d.person_id IS NOT NULL THEN 'Deceased' ELSE 'Alive' END AS DeathStatus,
    YEAR(CURRENT_DATE) - p.year_of_birth AS Age,
    'Unknown' AS SmokingStatus,  -- Needs observation lookup
    'Unknown' AS AlcoholUse,
    'Unknown' AS EducationLevel,
    'Unknown' AS MaritalStatus,
    CONCAT('P', p.person_id) AS OMOP_PersonID
FROM person p
LEFT JOIN concept gc ON p.gender_concept_id = gc.concept_id
LEFT JOIN concept ec ON p.ethnicity_concept_id = ec.concept_id
LEFT JOIN concept rc ON p.race_concept_id = rc.concept_id
LEFT JOIN death d ON p.person_id = d.person_id
WHERE p.person_id IN (
    -- AD cohort: patients with dementia/AD diagnoses
    SELECT DISTINCT person_id FROM condition_occurrence co
    JOIN concept c ON co.condition_concept_id = c.concept_id
    WHERE c.concept_code LIKE 'G30%' OR c.concept_code LIKE 'F00%'
);
```
