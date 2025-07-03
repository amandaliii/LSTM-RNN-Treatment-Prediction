import pandas as pd
from datasets import Dataset
import os

# LOAD AND CLEAN DATA
def load_mimic3_data(mimic_3data, max_hadm_ids=100):
    # INPUT DATA - READ CSV.GZ FILES
    admissions = pd.read_csv(os.path.join(mimic_3data, "ADMISSIONS.csv.gz"), compression="gzip", nrows=100)
    chart_events = pd.read_csv(os.path.join(mimic_3data, "CHARTEVENTS.csv.gz"), compression="gzip", nrows=100)
    lab_events = pd.read_csv(os.path.join(mimic_3data, "LABEVENTS.csv.gz"), compression="gzip", nrows=100)
    note_events = pd.read_csv(os.path.join(mimic_3data, "NOTEEVENTS.csv.gz"), compression="gzip", nrows=100)
    patients = pd.read_csv(os.path.join(mimic_3data, "PATIENTS.csv.gz"), compression="gzip", nrows=100)

    # OUTPUT DATA - READ CSV.GZ FILES
    diagnoses_icd = pd.read_csv(os.path.join(mimic_3data, "DIAGNOSES_ICD.csv.gz"), compression="gzip", nrows=100)
    prescriptions = pd.read_csv(os.path.join(mimic_3data, "PRESCRIPTIONS.csv.gz"), compression="gzip", nrows=100)
    procedures_icd = pd.read_csv(os.path.join(mimic_3data, "PROCEDURES_ICD.csv.gz"), compression="gzip", nrows=100)

    #OUTPUT DATA AGGREGATED
    # numerical data
    # aggregate diagnoses_icd (output) ICD9 codes/diagnoses per admission
    diagnoses_agg = diagnoses_icd.groupby(['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].agg(lambda x: ';'.join(x.astype(str).dropna())).reset_index()
    diagnoses_agg.rename(columns={'ICD9_CODE': 'DIAGNOSES'}, inplace=True)

    # aggregate procedures_icd (output) ICD9 codes/diagnoses per admission
    procedures_agg = procedures_icd.groupby(['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].agg(lambda x: ';'.join(x.astype(str).dropna())).reset_index()
    procedures_agg.rename(columns={'ICD9_CODE': 'PROCEDURES'}, inplace=True)

    # text data
    # aggregate prescriptions; drug names per admission
    prescriptions_agg = prescriptions.groupby(['SUBJECT_ID', 'HADM_ID'])['DRUG'].agg(lambda x: ';'.join(x.astype(str).dropna())).reset_index()
    prescriptions_agg.rename(columns={'DRUG': 'MEDICATIONS'}, inplace=True)

    # aggregate chart events
    chart_events_agg = chart_events.groupby(['SUBJECT_ID', 'HADM_ID']).agg({
        'ITEMID': lambda x: ';'.join(x.astype(str).dropna()),
        'VALUE': lambda x: ';'.join(x.astype(str).dropna())
    }).reset_index()
    chart_events_agg.rename(columns={'ITEMID': 'VITAL_ITEMS', 'VALUE': 'VITAL_VALUES'}, inplace=True)

    # aggregate lab events
    lab_events_agg = lab_events.groupby(['SUBJECT_ID', 'HADM_ID']).agg({
        'ITEMID': lambda x: ';'.join(x.astype(str).dropna()),
        'VALUE': lambda x: ';'.join(x.astype(str).dropna())
    }).reset_index()
    lab_events_agg.rename(columns={'ITEMID': 'LAB_ITEMS', 'VALUE': 'LAB_VALUES'}, inplace=True)

    # aggregate note events
    note_events_agg = note_events.groupby(['SUBJECT_ID', 'HADM_ID'])['TEXT'].agg(lambda x: ' '.join(x.dropna())).reset_index()
    note_events_agg.rename(columns={'TEXT': 'NOTES'}, inplace=True)

    # MERGE ALL TABLES WITH ADMISSION AS BASE
    merged_data = admissions[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DIAGNOSIS']]
    merged_data = merged_data.merge(patients[['SUBJECT_ID', 'GENDER', 'DOB']], on='SUBJECT_ID', how='left')
    merged_data = merged_data.merge(diagnoses_agg, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    merged_data = merged_data.merge(procedures_agg, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    merged_data = merged_data.merge(prescriptions_agg, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    merged_data = merged_data.merge(chart_events_agg, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    merged_data = merged_data.merge(lab_events_agg, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    merged_data = merged_data.merge(note_events_agg, on=['SUBJECT_ID', 'HADM_ID'], how='left')

    # MISSING VALUES (fill NaN with placeholders)
    merged_data.fillna({
        'DIAGNOSIS': 'No diagnosis',
        'PROCEDURES': 'No procedure',
        'MEDICATIONS': 'No medication',
        'VITAL_ITEMS': 'No vitals',
        'VITAL_VALUES': 'No values',
        'LAB_ITEMS': 'No labs',
        'LAB_VALUES': 'No values',
        'NOTES': 'No notes',
        'GENDER': 'Unknown',
        'DOB': 'Unknown'
    }, inplace=True)

    # CREATE INPUT-OUTPUT PAIRS FOR BERT
    data = []
    for _, row in merged_data.iterrows():
        input_text = (
            f"Patient: Gender: {row['GENDER']}, DOB: {row['DOB']}"
            f"Admission: {row['ADMITTIME']}, Diagnosis Text: {row['DIAGNOSIS']}"
            f"Clinical Notes: {row['NOTES']} Diagnoses: {row['DIAGNOSIS']}"
            f"Vitals: {row['VITAL_ITEMS']}={row['VITAL_VALUES']}"
            f"Labs: {row['LAB_ITEMS']}={row['LAB_VALUES']}"
        )
        output_text = f"Treatments: Procedures: {row['PROCEDURES']} Medications: {row['MEDICATIONS']} Lab Tests: {row['LAB_ITEMS']}"
        data.append({'input_text': input_text, 'output_text': output_text})

    full_dataset = Dataset.from_list(data)
    return full_dataset

if __name__ == "__main__":
    mimic_data = "/Users/amandali/Downloads/Mimic III"
    full_data = load_mimic3_data(mimic_data, max_hadm_ids=100)
    if full_data is None:
        print("full data loading and processing complete. ")
