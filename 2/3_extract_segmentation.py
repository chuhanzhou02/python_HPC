import pandas as pd
from pathlib import Path

def extract_segmentation_info():
    """
    Extract segmentation (SEG) file info for selected patients.
    1. Filter metadata for SEG modality.
    2. Match patient IDs.
    3. Save filtered segmentation info.
    """
    # ===== Load data =====
    patient_ids = pd.read_csv('2_data/rna_ids.csv')                     # RNA-seq patient IDs
    metadata = pd.read_csv('2/metadata.csv')                            # Metadata file
    
    # ===== Filter SEG modality =====
    seg_data = metadata[metadata['Modality'] == 'SEG']
    
    # ===== Filter matching Subject IDs =====
    filtered_data = seg_data[seg_data['Subject ID'].isin(patient_ids['PatientID'])]
    
    # ===== Select relevant columns =====
    result = filtered_data[['Subject ID', 'Series Description', 'Manufacturer', 'File Size', 'File Location']]
    
    # ===== Save results =====
    output_file = '2_data/segmentation_info.csv'
    result.to_csv(output_file, index=False)
    
    # ===== Summary =====
    print(f"Processing completed! Found {len(result)} SEG records.")
    print(f"Results saved to: {output_file}")
    print("\nStatistics:")
    print(f"Total number of patients: {len(patient_ids)}")
    print(f"Patients with SEG records: {len(result['Subject ID'].unique())}")
    print("\nManufacturer distribution:")
    print(result['Manufacturer'].value_counts())
    print("\nFile size statistics:")
    print(result['File Size'].describe())

if __name__ == "__main__":
    extract_segmentation_info()
