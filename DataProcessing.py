import pandas as pd
import os
import json

# mimic_3data is the directory path for the data in my local file
def load_mimic3_data(mimic_3data, nrows):
    # data dictionary mapping - file name, sort columns, and grouping column
    data_files = {
        'chart_events': ('CHARTEVENTS.csv.gz', ['HADM_ID', 'CHARTTIME'], 'ITEMID'),
        'input_events': ('INPUTEVENTS_MV.csv.gz', ['HADM_ID', 'STARTTIME'], 'ITEMID'),
        'lab_events': ('LABEVENTS.csv.gz', ['HADM_ID', 'CHARTTIME'], 'ITEMID'),
        'microbiology_events': ('MICROBIOLOGYEVENTS.csv.gz', ['HADM_ID', 'CHARTTIME'], 'SPEC_ITEMID'),
        'prescriptions': ('PRESCRIPTIONS.csv.gz', ['HADM_ID', 'STARTDATE'], 'DRUG'),
        'procedure_events': ('PROCEDUREEVENTS_MV.csv.gz', ['HADM_ID', 'STARTTIME'], 'ITEMID'),
    }

    # store grouped data for each category
    data_dicts = {}
    for key, (file_name, sort_cols, group_col) in data_files.items():
        file_path = os.path.join(mimic_3data, file_name)
        # check if file exists at specified path - if not, assign empty and move on
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            data_dicts[key] = {}
            continue
        # try and read and process other files that are found
        try:
            df = pd.read_csv(file_path, compression='gzip', nrows=nrows)
            # check if dataframe is empty
            if df.empty:
                print(f"Empty DataFrame: {file_path}")
                # if dataframe is empty then assign empty and move on
                data_dicts[key] = {}
            else:
                # check if required columns exist
                missing_cols = [col for col in sort_cols + [group_col] if col not in df.columns]
                # if any required columns are missing
                if missing_cols:
                    print(f"Missing columns: {missing_cols} in {file_name}")
                    data_dicts[key] = {}
                else:
                    # sort dataframe by specified columns
                    sorted_df = df.sort_values(by=sort_cols)
                    # group by HADM_ID and collect the group_col into lists, convert to dict
                    data_dicts[key] = sorted_df.groupby('HADM_ID')[group_col].apply(list).to_dict()
                    print(f"Loaded {file_name} with {len(df)} rows")
        # catch exceptions/errors
        except Exception as e:
            print(f"Error reading {file_name} : {e}")
            data_dicts[key] = {}


    # empty dict for merged results
    merged_dict = {}
    # stores all unique hadm_ids
    all_hadm_ids = set()
    # add all HADM_IDs from the dict to the set
    for d in data_dicts.values():
        all_hadm_ids.update(d.keys())

    for hadm_id in all_hadm_ids:
        current_dict = {
            # get list of ITEMID for chart events, default to empty list if not found
            'chart_items': data_dicts['chart_events'].get(hadm_id, []),
            'input_items': data_dicts['input_events'].get(hadm_id, []),
            'lab_items': data_dicts['lab_events'].get(hadm_id, []),
            'microbiology_items': data_dicts['microbiology_events'].get(hadm_id, []),
            'prescriptions_items': data_dicts['prescriptions'].get(hadm_id, []),
            'procedure_items': data_dicts['procedure_events'].get(hadm_id, [])
        }
        # only include the HADM_ID if all categories have non-empty lists
        if all(len(items) > 0 for items in current_dict.values()):
            merged_dict[hadm_id] = current_dict

    return merged_dict

mimic_data_dir = '/Users/amandali/Downloads/Mimic III'

try:
    result = load_mimic3_data(mimic_data_dir, nrows=100000)
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"Error in load mimic3 data: {e}")
