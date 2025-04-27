import os
import pandas as pd
import pydicom
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import measure
from tqdm import tqdm

# ==== Path Configuration for HPC ====
root_dir = '/user/home/ms13525/scratch/mshds-ml-data-2025/dataset1/'
OUTPUT_CSV = '/user/home/kv24343/ML/tumor_features.csv'   

def load_ct_volume(folder_path, verbose=True):
    """
    Load CT volume data, automatically filter CT type files, and handle corrupted slices.
    
    Parameters:
        folder_path (str): Path to DICOM folder
        verbose (bool): Whether to print detailed information
    Returns:
        images (ndarray): CT image data, shape: (slices, height, width)
        slices (list): List of loaded pydicom objects
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"[Error] Path does not exist: {folder_path}")
        
    if not os.path.isdir(folder_path):
        raise ValueError(f"[Error] Not a directory: {folder_path}")
        
    dcm_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".dcm")]
    if not dcm_files:
        raise ValueError(f"[Error] No DICOM files in directory: {folder_path}")
        
    if verbose:
        print(f"[Info] Found {len(dcm_files)} DICOM files in {folder_path}")
        print(f"[Info] DICOM files: {dcm_files}")
        
    slices = []
    skipped_files = 0
    invalid_files = []
    
    for filename in sorted(dcm_files):
        file_path = os.path.join(folder_path, filename)
        try:
            if not os.access(file_path, os.R_OK):
                print(f"[Warning] File not readable: {file_path}")
                skipped_files += 1
                invalid_files.append((filename, "Not readable"))
                continue
                
            dcm = pydicom.dcmread(file_path, stop_before_pixels=True)  # Don't read pixels first to speed up
            sop_class_uid = getattr(dcm, 'SOPClassUID', None)
            modality = getattr(dcm, 'Modality', None)
            
            if verbose:
                print(f"[Debug] File {filename}: SOPClassUID={sop_class_uid}, Modality={modality}")

            # Skip RTSTRUCT files
            if sop_class_uid == '1.2.840.10008.5.1.4.1.1.481.3' and modality == 'RTSTRUCT':
                if verbose:
                    print(f"[Info] Skipping RTSTRUCT file: {file_path}")
                continue

            # Only accept CT Image Storage
            if sop_class_uid == '1.2.840.10008.5.1.4.1.1.2' and modality == 'CT':
                # Actually read the pixels
                dcm = pydicom.dcmread(file_path)
                slices.append(dcm)
            else:
                if verbose:
                    print(f"[Warning] Not a CT file: {file_path}, SOPClassUID={sop_class_uid}, Modality={modality}")
                skipped_files += 1
                invalid_files.append((filename, f"Not a CT file (SOPClassUID={sop_class_uid}, Modality={modality})"))
        except Exception as e:
            skipped_files += 1
            if verbose:
                print(f"[Warning] Failed to read: {file_path}, error: {e}")
            invalid_files.append((filename, f"Read failed: {str(e)}"))

    if not slices:
        error_msg = f"[Error] No valid CT slices found in {folder_path}.\n"
        error_msg += f"Skipped {skipped_files} files.\n"
        error_msg += "Invalid files details:\n"
        for filename, reason in invalid_files:
            error_msg += f"- {filename}: {reason}\n"
        raise ValueError(error_msg)

    if verbose:
        print(f"[Info] Successfully read {len(slices)} CT slices, skipped {skipped_files} invalid files. Sorting...")

    # ===== Sorting logic =====
    try:
        slices_with_pos = [s for s in slices if hasattr(s, 'ImagePositionPatient')]
        if slices_with_pos:
            slices_with_pos.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            slices = slices_with_pos
            if verbose:
                print("[Info] Successfully sorted by ImagePositionPatient")
        else:
            slices_with_inst = [s for s in slices if hasattr(s, 'InstanceNumber')]
            if slices_with_inst:
                slices_with_inst.sort(key=lambda x: int(x.InstanceNumber))
                slices = slices_with_inst
                if verbose:
                    print("[Info] Successfully sorted by InstanceNumber")
            else:
                raise ValueError(f"[Error] Missing sorting information in {folder_path}.")
    except Exception as e:
        raise ValueError(f"[Error] Sorting failed: {e}")

    # ===== Stack image data =====
    try:
        images = np.stack([s.pixel_array for s in slices])
    except Exception as e:
        raise ValueError(f"[Error] Failed to stack image data: {e}")

    return images, slices


def find_segmentation(subject_path):
    """Find SEG file"""
    for subfolder, dirs, files in os.walk(subject_path):
        if "Segmentation" in subfolder:
            dcm_files = [f for f in files if f.endswith(".dcm")]
            if dcm_files:
                return os.path.join(subfolder, dcm_files[0])
    return None

def load_segmentation_mask(seg_path):
    """Load SEG mask"""
    try:
        ds = pydicom.dcmread(seg_path)
        seg = ds.pixel_array
        if seg.ndim == 4:
            seg = seg[0]  # Take the first label
        return seg.astype(bool)
    except Exception as e:
        print(f"Failed to load SEG: {seg_path}, error: {e}")
        return None

def extract_basic_features(volume):
    """Extract basic statistical features from the volume"""
    return {
        'mean': np.mean(volume),
        'std': np.std(volume),
        'min': np.min(volume),
        'max': np.max(volume),
        'size': volume.size,
        'shape': str(volume.shape)  # Convert to string for saving
    }

def extract_texture_features(volume, levels=64):
    """Extract texture features using GLCM"""
    num_slices = volume.shape[0]
    sampled_indices = np.linspace(0, num_slices-1, min(10, num_slices)).astype(int)
    textures = {'contrast': [], 'correlation': [], 'dissimilarity': [], 'homogeneity': []}

    for idx in sampled_indices:
        img = volume[idx]
        img = np.clip(img, 0, np.max(img))
        if np.max(img) != np.min(img):
            img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * (levels-1)).astype(np.uint8)
            glcm = graycomatrix(img, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)
            for prop in textures.keys():
                textures[prop].append(graycoprops(glcm, prop)[0, 0])

    features = {k: np.mean(v) if v else 0 for k, v in textures.items()}
    return features

def extract_morphological_features(volume, slices, mask):
    """Extract morphological features from the volume"""
    try:
        pixel_spacing = slices[0].PixelSpacing  # mm
        slice_thickness = float(slices[0].SliceThickness)  # mm
        spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])
    except:
        spacing = (1.0, 1.0, 1.0)

    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    volume_m = np.sum(mask) * voxel_volume

    try:
        verts, faces, _, _ = measure.marching_cubes(mask.astype(np.uint8), level=0.5, spacing=spacing)
        surface_mr = measure.mesh_surface_area(verts, faces)
    except:
        surface_mr = 0

    coords = np.argwhere(mask)
    if coords.shape[0] > 0:
        max_diamt = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)) * np.mean(spacing)
    else:
        max_diamt = 0

    compactn = (volume_m) / (surface_mr ** (1.5)) if surface_mr > 0 else 0

    return {
        'volume_m': volume_m,
        'surface_mr': surface_mr,
        'max_diamt': max_diamt,
        'compactn': compactn
    }

def find_ct_folder(base_folder):
    """
    Recursively search for a folder containing valid CT files.
    
    Parameters:
        base_folder (str): Base folder path to start searching
    Returns:
        str: Path to folder containing valid CT files, or None if not found
    """
    # First try the base folder
    try:
        dcm_files = [f for f in os.listdir(base_folder) if f.lower().endswith(".dcm")]
        if dcm_files:
            # Check if any of these files are CT
            for filename in dcm_files:
                file_path = os.path.join(base_folder, filename)
                try:
                    dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
                    if (getattr(dcm, 'SOPClassUID', None) == '1.2.840.10008.5.1.4.1.1.2' and 
                        getattr(dcm, 'Modality', None) == 'CT'):
                        return base_folder
                except:
                    continue
    except:
        pass

    # If no CT found in base folder, search subfolders
    try:
        for item in os.listdir(base_folder):
            item_path = os.path.join(base_folder, item)
            if os.path.isdir(item_path):
                result = find_ct_folder(item_path)
                if result is not None:
                    return result
    except:
        pass

    return None

# ==== Main Program ====
all_features = []
processed_count = 0
MAX_PROCESS = 60  # Process 6 samples

for subject in tqdm(os.listdir(root_dir)):
    if processed_count >= MAX_PROCESS:
        print(f"Reached maximum processing count {MAX_PROCESS}, stopping")
        break
        
    subject_path = os.path.join(root_dir, subject)
    if not os.path.isdir(subject_path):
        continue

    print(f"\nProcessing {subject}...")
    
    # Find the first study directory
    study_dirs = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
    if not study_dirs:
        print(f"No study directories found for {subject}")
        continue
        
    study_path = os.path.join(subject_path, study_dirs[0])
    
    # Find CT folder
    ct_folder = find_ct_folder(study_path)
    if ct_folder is None:
        print(f"No valid CT folder found for {subject}")
        continue

    try:
        volume, slices = load_ct_volume(ct_folder, verbose=False)
    except Exception as e:
        print(f"Skipping {subject}, CT loading failed, error: {e}")
        continue

    seg_path = find_segmentation(subject_path)
    mask = None
    if seg_path:
        mask = load_segmentation_mask(seg_path)
        if mask is not None:
            print(f"Using real SEG segmentation: {seg_path}")
        else:
            print(f"SEG loading failed, using fallback")
    else:
        print(f"No SEG found, using fallback")

    if mask is None:
        mask = volume > -600  # Fallback mask

    feats_basic = extract_basic_features(volume)
    feats_texture = extract_texture_features(volume)
    feats_morpho = extract_morphological_features(volume, slices, mask)

    feats_all = {**feats_basic, **feats_texture, **feats_morpho}
    feats_all['subject_id'] = subject
    feats_all['folder'] = ct_folder
    all_features.append(feats_all)
    
    processed_count += 1
    print(f"Completed {processed_count}/{MAX_PROCESS} samples")

# Save results
if all_features:
    df = pd.DataFrame(all_features)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to: {OUTPUT_CSV}")
    print(f"Processed {len(df)} samples in total")
else:
    print("No features extracted") 