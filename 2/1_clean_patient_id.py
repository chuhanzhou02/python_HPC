import csv
import os
from pathlib import Path

# ===== File paths =====
input_file = './2/rnaseq.txt'                       # Input RNA-seq data
output_file = './2_data/rna_ids.csv'     # Output patient IDs

# ===== Create output folder =====
os.makedirs(Path(output_file).parent, exist_ok=True)

# ===== Read patient IDs from RNA-seq file =====
with open(input_file, 'r') as infile:
    reader = csv.reader(infile, delimiter='\t')
    header = next(reader)       # First row contains patient IDs

# Extract patient IDs (skip the first column, which is gene names)
patient_ids = header[1:]

# ===== Write patient IDs to CSV =====
with open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['PatientID'])    # Add header
    for pid in patient_ids:
        writer.writerow([pid])

print(f"Patient IDs extracted and saved to: {output_file}")
