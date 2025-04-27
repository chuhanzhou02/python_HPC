import os
import pydicom
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage import measure

# Configure paths
root_dir = '/user/home/ms13525/scratch/mshds-ml-data-2025/dataset2/'
seg_csv_path = '/user/home/do24422/result2/seg.csv'
output_csv = '/user/home/do24422/result2/segmentation_features.csv'

# Read seg.csv
seg_info = pd.read_csv(seg_csv_path)

# Store results
results = []

# Iterate through each row
for idx, row in seg_info.iterrows():
    subject_id = row['Subject ID']
    seg_path = row['File Location']

    seg_full_path = os.path.join(root_dir, seg_path)

    # Load SEG file
    if not os.path.exists(seg_full_path):
        print(f"Path does not exist for {subject_id}: {seg_full_path}")
        continue

    seg_files = [os.path.join(seg_full_path, f) for f in os.listdir(seg_full_path) if f.endswith('.dcm')]
    if not seg_files:
        print(f"No DICOM files found for {subject_id}")
        continue

    # Assume there is only one SEG file
    try:
        seg = sitk.ReadImage(seg_files[0])
    except Exception as e:
        print(f"Failed to read DICOM for {subject_id}: {e}")
        continue

    # Convert segmentation to numpy array
    seg_array = sitk.GetArrayFromImage(seg)

    # Binarize (some SEG files store different label values, here we only take values > 0)
    mask = (seg_array > 0).astype(np.uint8)

    # Voxel size (mm)
    spacing = seg.GetSpacing()  # (x, y, z)

    voxel_volume = spacing[0] * spacing[1] * spacing[2]  # Single voxel volume (mm³)

    # Calculate volume
    volume_m = np.sum(mask) * voxel_volume  # Total volume

    # Calculate surface area (using marching cubes)
    try:
        verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=spacing)

        # Surface area
        surface_mr = measure.mesh_surface_area(verts, faces)

        # Calculate maximum diameter
        from scipy.spatial.distance import pdist
        if verts.shape[0] > 0:
            max_diamt = np.max(pdist(verts))
        else:
            max_diamt = 0

        # Compactness
        if volume_m > 0:
            compactn = (surface_mr ** 1.5) / volume_m
        else:
            compactn = 0

    except Exception as e:
        print(f"Marching cubes failed for {subject_id}: {e}")
        surface_mr = 0
        max_diamt = 0
        compactn = 0

    # Save results
    results.append({
        'subject_id': subject_id,
        'volume_m': volume_m,
        'surface_mr': surface_mr,
        'max_diamt': max_diamt,
        'compactn': compactn
    })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

print(f"Feature extraction finished. Output saved to {output_csv}")
