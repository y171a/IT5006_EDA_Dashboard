import pandas as pd
import streamlit as st
from Hello import load_data

# Define ICD-9 chapter ranges
ICD9_CHAPTERS = {
    'Infectious and Parasitic Diseases': (1, 139),
    'Neoplasms': (140, 239),
    'Endocrine, Nutritional, Metabolic': (240, 279),
    'Blood and Blood-Forming Organs': (280, 289),
    'Mental Disorders': (290, 319),
    'Nervous System and Sense Organs': (320, 389),
    'Circulatory System': (390, 459),
    'Respiratory System': (460, 519),
    'Digestive System': (520, 579),
    'Genitourinary System': (580, 629),
    'Pregnancy and Childbirth': (630, 679),
    'Skin and Subcutaneous Tissue': (680, 709),
    'Musculoskeletal System': (710, 739),
    'Congenital Anomalies': (740, 759),
    'Perinatal Conditions': (760, 779),
    'Symptoms and Ill-Defined Conditions': (780, 799),
    'Injury and Poisoning': (800, 999),
    'Supplementary Factors (V codes)': ('V01', 'V91'),
    'External Causes (E codes)': ('E000', 'E999')
}

def map_icd9(code):
    """Map ICD-9 code to its chapter category."""
    try:
        if code.startswith('V'):
            return 'Supplementary Factors (V codes)'
        elif code.startswith('E'):
            return 'External Causes (E codes)'
        else:
            num = float(code)
            for chapter, (low, high) in ICD9_CHAPTERS.items():
                if isinstance(low, (int, float)) and low <= num <= high:
                    return chapter
    except:
        return 'Unknown'
    return 'Unknown'

@st.cache_data
def get_processed_dataframe():
    """Load and process the diabetes dataset with all cleaning steps."""
    
    # Load original data
    df = load_data()
    
    # Drop weight column (97% missing)
    df.drop(columns='weight', inplace=True)
    
    # Drop constant drug features
    constant_drugs = ['examide', 'citoglipton']
    existing_constants = [col for col in constant_drugs if col in df.columns]
    if existing_constants:
        df.drop(columns=existing_constants, inplace=True)
    
    # Drop near-zero variance features
    ID_COLUMNS = ['encounter_id', 'patient_nbr']
    NZV_THRESHOLD = 0.01
    low_var_features = []
    
    analysis_cols = [col for col in df.columns if col not in ID_COLUMNS]
    
    for col in analysis_cols:
        freq = df[col].value_counts(normalize=True, dropna=False)
        if len(freq) == 1:
            low_var_features.append(col)
        elif len(freq) > 0 and freq.iloc[0] > (1 - NZV_THRESHOLD):
            low_var_features.append(col)
    
    if low_var_features:
        df.drop(columns=low_var_features, inplace=True)
    
    # Load IDs mapping
    df_ids_mapping = pd.read_csv("IDS_mapping.csv")
    
    # Add descriptive names for IDs
    df["discharge_disposition_name"] = df["discharge_disposition_id"].apply(
        lambda x: df_ids_mapping[(df_ids_mapping["Type"]=="discharge_disposition_id") & 
                                 (df_ids_mapping["ID"] == x)].description.values[0]
    )
    df["admission_type_name"] = df["admission_type_id"].apply(
        lambda x: df_ids_mapping[(df_ids_mapping["Type"]=="admission_type_id") & 
                                 (df_ids_mapping["ID"] == x)].description.values[0]
    )
    df["admission_source_name"] = df["admission_source_id"].apply(
        lambda x: df_ids_mapping[(df_ids_mapping["Type"]=="admission_source_id") & 
                                 (df_ids_mapping["ID"] == x)].description.values[0]
    )
    
    # Apply ICD-9 mapping to diagnosis columns
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[f'{col}_category'] = df[col].apply(map_icd9)
    
    return df