import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import os

# LOAD AND CLEAN DATA
def load_mimic3_data(mimic_3data, max_hadm_ids=100):
    # INPUT DATA - READ CSV.GZ FILES
    admissions = pd.read_csv(os.path.join(mimic_3data, "ADMISSIONS.csv.gz"), compression="gzip")
    chart_events = pd.read_csv(os.path.join(mimic_3data, "CHARTEVENTS.csv.gz"), compression="gzip")
    lab_events = pd.read_csv(os.path.join(mimic_3data, "LABEVENTS.csv.gz"), compression="gzip")
    note_events = pd.read_csv(os.path.join(mimic_3data, "NOTEEVENTS.csv.gz"), compression="gzip")
    patients = pd.read_csv(os.path.join(mimic_3data, "PATIENTS.csv.gz"), compression="gzip")

    # OUTPUT DATA - READ CSV.GZ FILES
    diagnoses_icd = pd.read_csv(os.path.join(mimic_3data, "DIAGNOSES_ICD.csv.gz"), compression="gzip")
    prescriptions = pd.read_csv(os.path.join(mimic_3data, "PRESCRIPTIONS.csv.gz"), compression="gzip")
    procedures_icd = pd.read_csv(os.path.join(mimic_3data, "PROCEDURES_ICD.csv.gz"), compression="gzip")

    # FILTER TO COMMON HADMS_IDS
    common_hadm_ids = set(admissions['HADM_ID']).intersection(
    note_events['HADM_ID'], diagnoses_icd['HADM_ID'], prescriptions['HADM_ID'], procedures_icd['HADM_ID']
    )
    # make sure HADM_IDs are in each dataset
    common_hadm_ids = list(common_hadm_ids)[:max_hadm_ids]
    admissions = admissions[admissions['HADM_ID'].isin(common_hadm_ids)]
    chart_events = chart_events[chart_events['HADM_ID'].isin(common_hadm_ids)]
    lab_events = lab_events[lab_events['HADM_ID'].isin(common_hadm_ids)]
    note_events = note_events[note_events['HADM_ID'].isin(common_hadm_ids)]
    diagnoses_icd = diagnoses_icd[diagnoses_icd['HADM_ID'].isin(common_hadm_ids)]
    prescriptions = prescriptions[prescriptions['HADM_ID'].isin(common_hadm_ids)]
    procedures_icd = procedures_icd[procedures_icd['HADM_ID'].isin(common_hadm_ids)]

    #OUTPUT DATA AGGREGATED
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
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    data = []
    for _, row in merged_data.iterrows():
        input_text = (
            f"Patient: Gender: {row['GENDER']}, DOB: {row['DOB']}"
            f"Admission: {row['ADMITTIME']}, Diagnosis Text: {row['DIAGNOSIS']}"
            f"Clinical Notes: {row['NOTES'][:100]}"
            f"Vital Signs: {row['VITAL_VALUES']}. Lab Results: {row['LAB_VALUES']}"
        )
        output_text = f"Procedures: {row['PROCEDURES']}; Medications: {row['MEDICATIONS']}; Lab Tests: {row['LAB_ITEMS']}"

        # tokenize to ensure within ClinicalBERT's limit
        inputs = tokenizer(input_text, truncation=True, max_length=512, return_tensors="pt")
        input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)

        data.append({'input_text': input_text, 'output_text': output_text})

    full_dataset = Dataset.from_list(data)

    # SPLIT DATASET
    train_test = full_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test['train']
    test_dataset = train_test['test']

    return train_dataset, test_dataset

if __name__ == "__main__":
    mimic_data = "/Users/amandali/Downloads/Mimic III"
    train_dataset, test_dataset = load_mimic3_data(mimic_data, max_hadm_ids=100)

    # print first few entries
    print("Train Dataset Sample:")
    for i in range(min(3, len(train_dataset))):
        print(train_dataset[i])

    print("\nTest Dataset Sample:")
    for i in range(min(3, len(test_dataset))):
        print(test_dataset[i])
