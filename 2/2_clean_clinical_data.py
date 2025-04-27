import pandas as pd
import numpy as np
import os
from pathlib import Path

# ===== Create output folder =====
os.makedirs('2_data', exist_ok=True)

def clean_clinical_data():
    """
    Clean clinical data:
    1. Match Patient IDs with RNA-seq data.
    2. Handle missing values.
    3. Save cleaned data.
    """
    # ===== Load data =====
    print("Loading data...")
    patient_ids = pd.read_csv('2_data/rna_ids.csv')
    clinical_data = pd.read_csv('2/clinical2.csv')
    
    print(f"Patient IDs: {patient_ids.shape}")
    print(f"Clinical data: {clinical_data.shape}")
    
    # ===== Align IDs =====
    clinical_data['Case ID'] = clinical_data['Case ID'].str.replace('AMC-', 'R01-')
    patient_ids['PatientID'] = patient_ids['PatientID'].str.replace('R01-1-1-', 'R01-')
    
    # ===== Filter data =====
    matched_data = clinical_data[clinical_data['Case ID'].isin(patient_ids['PatientID'])]
    print(f"Matched clinical data shape: {matched_data.shape}")
    
    # Check unmatched IDs
    unmatched_ids = set(patient_ids['PatientID']) - set(matched_data['Case ID'])
    if unmatched_ids:
        print(f"Unmatched Patient IDs: {unmatched_ids}")
    
    # ===== Handle missing values =====
    print("Handling missing values...")
    numeric_cols = matched_data.select_dtypes(include=[np.number]).columns
    categorical_cols = matched_data.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        if matched_data[col].isnull().sum() > 0:
            median_value = matched_data[col].median()
            matched_data[col] = matched_data[col].fillna(median_value)
            print(f"Filled missing numeric '{col}' with median: {median_value}")
    
    for col in categorical_cols:
        if matched_data[col].isnull().sum() > 0:
            mode_value = matched_data[col].mode()[0]
            matched_data[col] = matched_data[col].fillna(mode_value)
            print(f"Filled missing categorical '{col}' with mode: {mode_value}")
    
    if matched_data.isnull().sum().sum() == 0:
        print("All missing values handled.")
    else:
        print("Warning: Some missing values remain.")

    # ===== Save cleaned data =====
    output_file = '2_data/cleaned_clinical2.csv'
    matched_data.to_csv(output_file, index=False)
    print(f"Cleaned clinical data saved to: {output_file}")
    
    # ===== Summary =====
    print(f"Final data shape: {matched_data.shape}")
    if 'Survival Status' in matched_data.columns:
        print("Survival Status distribution:")
        print(matched_data['Survival Status'].value_counts(normalize=True))
    
    return matched_data

if __name__ == "__main__":
    clean_clinical_data()
