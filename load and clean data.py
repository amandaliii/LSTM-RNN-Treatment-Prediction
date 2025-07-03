import pandas as pd
from datasets import Dataset
import os

""" LOADING AND CLEANING DATA """
def load_mimic3_data(mimic_data):
    """INPUT DATA"""
    admissions = pd.read_csv(os.path.join(mimic_data, "ADMISSIONS.csv.gz"), compression="gzip")
    chart_events = pd.read_csv(os.path.join(mimic_data, "CHARTEVENTS.csv.gz"), compression="gzip")
    lab_events = pd.read_csv(os.path.join(mimic_data, "LABEVENTS.csv.gz"), compression="gzip")
    note_events = pd.read_csv(os.path.join(mimic_data, "NOTEVENTS.csv.gz"), compression="gzip")
    patients = pd.read_csv(os.path.join(mimic_data, "PATIENTS.csv.gz"), compression="gzip")
    print("input data loaded")

    """OUTPUT DATA"""
    diagnoses_icd = pd.read_csv(os.path.join(mimic_data, "D_ICD_DIAGNOSES.csv.gz"), compression="gzip")
    prescriptions = pd.read_csv(os.path.join(mimic_data, "PRESCRIPTIONS.csv.gz"), compression="gzip")
    procedures_icd = pd.read_csv(os.path.join(mimic_data, "D_ICD_PROCEDURES.csv.gz"), compression="gzip")
    print("output data loaded")

    """OUTPUT DATA AGGREGATED"""
    # numerical data
    # aggregate diagnoses_icd (output) ICD9 codes/diagnoses per admission
    diagnoses_agg = diagnoses_icd.groupby(['SUBJECT_ID', 'HADM_ID'])['ICV9_CODE'].agg(lambda x: ';'.join(x.dropna())).reset_index()
    diagnoses_agg.rename(columns={'ICD9C_CODE': 'DIAGNOSES'}, inplace=True)

    # aggregate procedures_icd (output) ICD9 codes/diagnoses per admission
    procedures_agg = procedures_icd.groupby(['SUBJECT_ID', 'HADM_ID'])['ICV9_CODE'].agg(lambda x: ';'.join(x.dropna())).reset_index()
    diagnoses_agg.rename(columns={'ICD9C_CODE': 'PROCEDURES'}, inplace=True)

    # text data
    # aggregate prescriptions; drug names per admission
    prescriptions_agg = prescriptions.groupby(['SUBJECT_ID', 'HADM_ID'])['ICV9_CODE'].agg(lambda x: ';'.join(x.dropna())).reset_index()
    diagnoses_agg.rename(columns={'ICD9C_CODE': 'PRESCRIPTIONS'}, inplace=True)
    print("data aggregation done")

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
    note_events_agg.rename(columns={'ITEMID': 'NOTES'}, inplace=True)
    print("event data aggregation complete")

    """MERGE ALL TABLES WITH ADMISSION AS BASE"""
    merged_data = admissions[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DIAGNOSIS']]
    merged_data = merged_data.merge(patients[['SUBJECT_ID', 'GENDER', 'DOB']], on='SUBJECT_ID', how='left')
    merged_data = merged_data.merge(diagnoses_agg, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    merged_data = merged_data.merge(procedures_agg, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    merged_data = merged_data.merge(prescriptions_agg, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    merged_data = merged_data.merge(chart_events_agg, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    merged_data = merged_data.merge(lab_events_agg, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    merged_data = merged_data.merge(note_events_agg, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    print("data merge complete")

    """MISSING VALUES (fill NaN with placeholders)"""
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
    print("null value replacement complete")

    """CREATE INPUT-OUTPUT PAIRS FOR BERT"""
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

    fullData = Dataset.from_list(data)
    print("input output pairs and data prep for bert complete")
    return fullData

if __name__ == "__main__":
    mimic_data = "/Users/amandali/Downloads/Mimic III"
    fullData = load_mimic3_data(mimic_data)
    if fullData is None:
        print("full data loading and processing complete. fullData size:", len(fullData))