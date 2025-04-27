import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folders
os.makedirs('1_data', exist_ok=True)
os.makedirs('1_plot', exist_ok=True)

# Load data
def load_data():
    tumor = pd.read_csv('1_data/tumor_features_v2.csv')
    clinical = pd.read_csv('1_data/clinical1_preprocessed.csv')
    
    tumor = tumor.rename(columns={'subject_id': 'PatientID'})
    tumor['PatientID'] = tumor['PatientID'].astype(str)
    clinical['PatientID'] = clinical['PatientID'].astype(str)
    
    data = pd.merge(tumor, clinical, on='PatientID', how='inner')
    print(f"Merged data shape: {data.shape}")
    
    data['deadstatus.event'] = data['deadstatus.event_1']
    print(f"Target distribution:\n{data['deadstatus.event'].value_counts()}")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    
    exclude = ['PatientID', 'deadstatus.event', 'deadstatus.event_1', 'Survival.time', 'Survival.time_raw', 'age_raw']
    feature_cols = [col for col in data.columns if col not in exclude]
    
    X = data[feature_cols]
    y = data['deadstatus.event']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, X.columns.tolist()

# Feature importance
def feature_importance(X, y, feature_names):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    model = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, scale_pos_weight=len(y[y==0])/len(y[y==1]))
    model.fit(X_res, y_res)
    
    importance = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
    importance = importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance.head(10))
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig('1_plot/feature_importance.png')
    plt.close()
    
    importance.to_csv('1_data/feature_importance.csv', index=False)
    print("Feature importance saved.")
    
    return model, X_res, y_res

# SHAP analysis
def shap_analysis(model, X_res, feature_names):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_res)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values, X_res, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig('1_plot/shap_summary.png')
    plt.close()
    
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_features = [feature_names[i] for i in np.argsort(mean_abs_shap)[-3:][::-1]]
    
    for feat in top_features:
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(feat, shap_values.values, X_res, feature_names=feature_names, show=False)
        plt.title(f'SHAP Dependence Plot: {feat}')
        plt.tight_layout()
        plt.savefig(f'1_plot/shap_dependence_{feat}.png')
        plt.close()

# Main
def main():
    print("Loading data...")
    X, y, feature_names = load_data()
    print("Feature importance analysis...")
    model, X_res, y_res = feature_importance(X, y, feature_names)
    print("SHAP analysis...")
    shap_analysis(model, X_res, feature_names)
    print("Analysis completed.")

if __name__ == "__main__":
    main()
