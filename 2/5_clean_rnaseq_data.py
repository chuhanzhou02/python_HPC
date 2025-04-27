import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===== Create output folders =====
os.makedirs('2_data', exist_ok=True)
os.makedirs('2_plot', exist_ok=True)

# ===== RNA-seq cleaning =====
def clean_rnaseq_data(file_path):
    print("Loading RNA-seq data...")
    rnaseq_data = pd.read_csv(file_path, sep='\t', index_col=0, na_values=['NA', ''])
    print(f"Original RNA-seq data shape: {rnaseq_data.shape}")

    # Remove rows with >50% missing values
    na_ratio = rnaseq_data.isna().mean(axis=1)
    rnaseq_data = rnaseq_data[na_ratio <= 0.5]
    print(f"Data shape after removing rows with >50% NA: {rnaseq_data.shape}")

    # Remove columns that are all NA
    rnaseq_data = rnaseq_data.dropna(axis=1, how='all')
    print(f"Data shape after removing all-NA columns: {rnaseq_data.shape}")

    # Fill remaining NAs with mean
    rnaseq_data = rnaseq_data.fillna(rnaseq_data.mean())

    # Save cleaned data
    rnaseq_data.to_csv('2_data/rnaseq_cleaned.txt', sep='\t')
    print("Cleaned RNA-seq data saved to 2_data/rnaseq_cleaned.txt")
    return rnaseq_data

# ===== Feature selection =====
def select_features(rnaseq_data, clinical_path, max_features=20):
    # Load clinical data
    clinical_data = pd.read_csv(clinical_path)
    clinical_data = clinical_data.rename(columns={'Case ID': 'PatientID'})
    clinical_data['PatientID'] = clinical_data['PatientID'].str.replace('R0', 'R01-').str.replace('R01-1-', 'R01-')
    clinical_data['deadstatus.event'] = (clinical_data['Survival Status'] == 'Dead').astype(int)

    # Align RNA-seq Patient IDs
    rnaseq_data = rnaseq_data.T.reset_index().rename(columns={'index': 'PatientID'})
    rnaseq_data['PatientID'] = rnaseq_data['PatientID'].str.replace('R0', 'R01-').str.replace('R01-1-', 'R01-')

    # Merge
    merged_data = pd.merge(rnaseq_data, clinical_data[['PatientID', 'deadstatus.event']], on='PatientID', how='inner')
    print(f"Merged data shape: {merged_data.shape}")

    X = merged_data.drop(['PatientID', 'deadstatus.event'], axis=1)
    y = merged_data['deadstatus.event']

    # Variance threshold
    selector = VarianceThreshold(threshold=0.1)
    X_selected = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Features after variance selection: {len(selected_features)}")

    # Mutual information (if needed)
    if len(selected_features) > max_features:
        mi_scores = mutual_info_classif(X[selected_features], y)
        mi_df = pd.DataFrame({'feature': selected_features, 'mi_score': mi_scores})
        mi_df = mi_df.sort_values('mi_score', ascending=False)
        selected_features = mi_df.head(max_features)['feature'].tolist()

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='mi_score', y='feature', data=mi_df.head(max_features))
        plt.title('Top RNA-seq Features by Mutual Information')
        plt.tight_layout()
        plt.savefig('2_plot/rna_feature_importance.png')
        plt.close()
        print("Feature importance plot saved to 2_plot/rna_feature_importance.png")

        # SHAP analysis
        model = XGBClassifier(random_state=42)
        model.fit(X[selected_features], y)
        explainer = shap.Explainer(model)
        shap_values = explainer(X[selected_features])

        # SHAP plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values.values, X[selected_features], feature_names=selected_features, show=False)
        plt.tight_layout()
        plt.savefig('2_plot/shap_summary.png')
        plt.close()
        print("SHAP summary plot saved to 2_plot/shap_summary.png")

        # Save feature importance data
        mi_df.to_csv('2_data/rna_feature_importance.csv', index=False)
        print("Feature importance scores saved to 2_data/rna_feature_importance.csv")

    # Save selected features
    selected_data = rnaseq_data[['PatientID'] + selected_features]
    selected_data.to_csv('2_data/rnaseq_selected.txt', sep='\t', index=False)
    print(f"Selected RNA-seq features saved to 2_data/rnaseq_selected.txt")

    return selected_features

# ===== Main =====
def main():
    rnaseq_cleaned = clean_rnaseq_data('2/rnaseq.txt')
    selected_features = select_features(rnaseq_cleaned, '2/clinical2.csv')
    print(f"Final selected features ({len(selected_features)}): {selected_features}")

if __name__ == "__main__":
    main()
