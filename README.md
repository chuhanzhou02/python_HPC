# Medical Image Analysis Project

This project provides a comprehensive data processing and analysis pipeline for medical image analysis, focusing on tumor feature extraction and patient survival prediction using multi-modal data including CT scans, clinical data, and RNA sequencing data.

## Project Overview

The project consists of two main datasets:
1. Dataset 1: Basic analysis pipeline
2. Dataset 2: Advanced multi-modal analysis

## Environment Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/chuhanzhou02/python_HPC.git
cd python_HPC
```

2. Create and activate a virtual environment (recommended):
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Processing Pipeline

### Dataset 1 Processing
1. `1_preprocess_dataset1.py`: Clinical Data Preprocessing
   - Missing value handling
   - Continuous feature standardization
   - Categorical feature encoding
   - Data merging and cleaning

2. `2_extract_features.py`: Medical Image Feature Extraction
   - CT image loading and preprocessing
   - Segmentation mask processing
   - Basic statistical feature extraction
   - Texture feature extraction
   - Morphological feature extraction

3. `3_ML.py`: Machine Learning Analysis
   - Data loading and preprocessing
   - Feature importance analysis
   - SHAP value analysis
   - Model training and evaluation

### Dataset 2 Processing
1. `1_clean_patient_id.py`: Patient ID Standardization
2. `2_clean_clinical_data.py`: Clinical Data Processing
3. `3_extract_segmentation.py`: Segmentation Data Processing
4. `4_extract_imaging_features.py`: Advanced Imaging Feature Extraction
5. `5_clean_rnaseq_data.py`: RNA Sequencing Data Processing
   - Data cleaning and normalization
   - Feature selection using variance threshold
   - Mutual information analysis
   - SHAP value analysis
6. `6_xgboost_model_analysis.py`: Advanced Model Analysis
   - Multi-modal data integration
   - Feature interaction analysis
   - Cross-validation
   - Performance evaluation
   - Model interpretability analysis

## Running the Code

### Dataset 1 Execution Order
1. Run clinical data preprocessing:
```bash
python 1/1_preprocess_dataset1.py
```

2. Extract imaging features:
```bash
python 1/2_extract_features.py
```

3. Perform machine learning analysis:
```bash
python 1/3_ML.py
```

### Dataset 2 Execution Order
1. Clean and standardize patient IDs:
```bash
python 2/1_clean_patient_id.py
```

2. Process clinical data:
```bash
python 2/2_clean_clinical_data.py
```

3. Process segmentation data:
```bash
python 2/3_extract_segmentation.py
```

4. Extract imaging features:
```bash
python 2/4_extract_imaging_features.py
```

5. Process RNA sequencing data:
```bash
python 2/5_clean_rnaseq_data.py
```

6. Run advanced model analysis:
```bash
python 2/6_xgboost_model_analysis.py
```

## Key Features

- Multi-modal Data Integration
  - Clinical data processing
  - CT image analysis
  - RNA sequencing analysis
  - Feature interaction analysis

- Advanced Feature Engineering
  - Automated DICOM CT image processing
  - Multi-dimensional tumor feature extraction
  - RNA-seq feature selection
  - Feature interaction creation

- Machine Learning Analysis
  - XGBoost-based survival prediction
  - Feature importance analysis
  - SHAP value interpretation
  - Cross-validation and performance evaluation

## Data Requirements

- Clinical Data: CSV format containing patient ID, clinical features, and survival information
- Imaging Data: DICOM format CT scan images
- Segmentation Data: DICOM format segmentation masks
- RNA Sequencing Data: Tab-separated text file with gene expression data

## Output

The pipeline generates:
- Preprocessed clinical data
- Extracted tumor features
- Processed RNA sequencing data
- Feature importance analysis plots
- SHAP value analysis plots
- Model performance metrics
- Cross-validation results

## Notes

- Ensure proper organization of input data files
- Process data in the correct sequence
- For detailed information about each component, please refer to the respective Python files
- The project requires significant computational resources for processing medical images and RNA sequencing data
- Make sure all required data files are in the correct directories before running the scripts
- Check the output directories for generated results and visualizations 