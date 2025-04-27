import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import shap
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif

os.makedirs('2_plot', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('2_plot/xgboost_analysis.log'),
        logging.StreamHandler()
    ]
)

def create_interaction_features(X):

    interaction_features = pd.DataFrame()
    
    imaging_features = ['volume_m', 'surface_mr', 'max_diamt']
    available_imaging = [f for f in imaging_features if f in X.columns]
    
    for i in range(len(available_imaging)):
        for j in range(i+1, len(available_imaging)):
            interaction_features[f'{available_imaging[i]}_{available_imaging[j]}'] = X[available_imaging[i]] * X[available_imaging[j]]
    
    clinical_features = ['Age at Histological Diagnosis']
    available_clinical = [f for f in clinical_features if f in X.columns]
    gender_features = [col for col in X.columns if col.startswith('Gender_')]
    
    for feature in available_clinical:
        for img_feature in available_imaging:
            interaction_features[f'{feature}_{img_feature}'] = X[feature] * X[img_feature]
    
    for gender_feature in gender_features:
        for img_feature in available_imaging:
            interaction_features[f'{gender_feature}_{img_feature}'] = X[gender_feature] * X[img_feature]
    
    logging.info(f"The number of created interaction features: {len(interaction_features.columns)}")
    if len(interaction_features.columns) > 0:
        logging.info("List of interaction characteristics:")
        logging.info(interaction_features.columns.tolist())
    
    return interaction_features

def evaluate_model(y_true, y_pred, y_prob=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    return metrics

def load_and_preprocess_data():
    try:
        tumor_features = pd.read_csv('2_data/segmentation_features.csv')
        clinical_data = pd.read_csv('2_data/cleaned_clinical2.csv')
        rnaseq_data = pd.read_csv('2_data/rnaseq_selected.txt', sep='\t')
        
        
        tumor_features = tumor_features.rename(columns={'subject_id': 'PatientID'})
        
        tumor_features['PatientID'] = tumor_features['PatientID'].str.replace('R01-1-', 'R01-')
        clinical_data['Case ID'] = clinical_data['Case ID'].str.replace('R01-1-1-', 'R01-')
        rnaseq_data['PatientID'] = rnaseq_data['PatientID'].str.replace('R01-1-1-', 'R01-')
        
        merged_data = pd.merge(tumor_features, clinical_data, left_on='PatientID', right_on='Case ID', how='left')
      
        merged_data = pd.merge(merged_data, rnaseq_data, on='PatientID', how='left')
     
        missing_stats = merged_data.isnull().sum()
        missing_stats = missing_stats[missing_stats > 0]
        if not missing_stats.empty:
           
            logging.warning(missing_stats)
          
            numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
            merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data[numeric_cols].median())
          
        merged_data['deadstatus.event'] = (merged_data['Survival Status'] == 'Dead').astype(int)
        
        logging.info("目标变量分布:")
        logging.info(merged_data['deadstatus.event'].value_counts(normalize=True))
        
        excluded_features = [
            'PatientID', 'Case ID',  
            'deadstatus.event',  
            'Survival Status',  
            'Time to Death (days)', 'Date of Death', 'Date of Last Known Alive',  
            'Recurrence', 'Recurrence Location', 'Date of Recurrence'  
        ]
    
        existing_excluded_features = [col for col in excluded_features if col in merged_data.columns]
        logging.info(f"The actual excluded features: {existing_excluded_features}")
        
        
        numeric_features = merged_data.select_dtypes(include=[np.number]).columns
        categorical_features = merged_data.select_dtypes(include=['object']).columns
        
        for feature in categorical_features:
            if feature not in existing_excluded_features:
                dummies = pd.get_dummies(merged_data[feature], prefix=feature)
            
                dummies.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_') for col in dummies.columns]
                merged_data = pd.concat([merged_data, dummies], axis=1)
                merged_data = merged_data.drop(feature, axis=1)
        
        logging.info("Starting to standardize numeric features...")
        numeric_features = [col for col in numeric_features if col not in existing_excluded_features]
        if len(numeric_features) > 0:
            scaler = StandardScaler()
            merged_data[numeric_features] = scaler.fit_transform(merged_data[numeric_features])
            logging.info("Numeric features standardized.")
        
        X = merged_data.drop(existing_excluded_features, axis=1)
        interaction_features = create_interaction_features(X)
        
        if not interaction_features.empty:
            X = pd.concat([X, interaction_features], axis=1)
            interaction_scaler = StandardScaler()
            X[interaction_features.columns] = interaction_scaler.fit_transform(X[interaction_features.columns])
            logging.info("Interaction features standardized.")
        
        y = merged_data['deadstatus.event']
    
        X.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_') for col in X.columns]
        
        logging.info(f"Total number of features used for modeling:{X.shape[1]}")
        logging.info("Feature type statistics:")
        logging.info(f"Numeric features: {len(numeric_features)}")
        logging.info(f"Categorical features:{len(categorical_features)}")
        logging.info(f"Interaction features: {len(interaction_features.columns)}")
        
        return X, y, X.columns.tolist()
    
    except Exception as e:
        logging.error(f"Error occurred during data loading and preprocessing: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Error details: {str(e)}")
        raise

def single_modality_analysis(X, y, feature_names):

    results = {}
    
    try:
        imaging_features = [col for col in feature_names if col in [
            'volume_m', 'surface_mr', 'max_diamt', 'compactn',
            'volume_m_surface_mr', 'volume_m_max_diamt', 'surface_mr_max_diamt'
        ]]
        
        clinical_features = [col for col in feature_names if col not in imaging_features and not col.startswith('ENSG')]
        rnaseq_features = [col for col in feature_names if col.startswith('ENSG')]
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        class_weights = {}
        for label in [0, 1]:
            count = len(y[y == label])
            if count > 0:
                class_weights[label] = len(y) / (2 * count)
            else:
                class_weights[label] = 1.0
        
        xgb_model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.01,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            scale_pos_weight=class_weights.get(1, 1.0)
        )
        
        if imaging_features:
            X_imaging = X[imaging_features]
            results['imaging'] = evaluate_modality(X_imaging, y, xgb_model, cv)
            logging.info("Imaging features evaluation results:")
            for metric, stats in results['imaging'].items():
                logging.info(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        if clinical_features:
            X_clinical = X[clinical_features]
            results['clinical'] = evaluate_modality(X_clinical, y, xgb_model, cv)
            logging.info("Clinical features evaluation results:")
            for metric, stats in results['clinical'].items():
                logging.info(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        if rnaseq_features:
            X_rnaseq = X[rnaseq_features]
            results['rnaseq'] = evaluate_modality(X_rnaseq, y, xgb_model, cv)
            logging.info("RNA-seq features evaluation results:")
            for metric, stats in results['rnaseq'].items():
                logging.info(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return results
    
    except Exception as e:
        logging.error(f"Error during single modality analysis: {str(e)}")
        raise

def evaluate_modality(X, y, model, cv):
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        for train_idx, test_idx in cv.split(X_resampled, y_resampled):
            X_train, X_test = X_resampled.iloc[train_idx], X_resampled.iloc[test_idx]
            y_train, y_test = y_resampled.iloc[train_idx], y_resampled.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            fold_metrics = evaluate_model(y_test, y_pred, y_prob)
            for metric, value in fold_metrics.items():
                if metric != 'confusion_matrix':
                    metrics[metric].append(value)
        
        return {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in metrics.items()
        }
    
    except Exception as e:
        logging.error(f"Error during modality evaluation: {str(e)}")
        raise

def multimodal_fusion(X, y):
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        class_weights = {}
        for label in [0, 1]:
            count = len(y[y == label])
            if count > 0:
                class_weights[label] = len(y) / (2 * count)
            else:
                class_weights[label] = 1.0
        
        xgb_model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.01,  
            max_depth=3,  
            min_child_weight=5,  
            subsample=0.8,  
            colsample_bytree=0.8,  
            reg_alpha=0.1,  
            reg_lambda=1.0,  
            random_state=42,
            scale_pos_weight=class_weights.get(1, 1.0)
        )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {}
        
        fusion_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        for train_idx, test_idx in cv.split(X_resampled, y_resampled):
            X_train, X_test = X_resampled.iloc[train_idx], X_resampled.iloc[test_idx]
            y_train, y_test = y_resampled.iloc[train_idx], y_resampled.iloc[test_idx]
            
            xgb_model.fit(X_train, y_train)
            y_pred = xgb_model.predict(X_test)
            y_prob = xgb_model.predict_proba(X_test)[:, 1]
            
            fold_metrics = evaluate_model(y_test, y_pred, y_prob)
            for metric, value in fold_metrics.items():
                if metric != 'confusion_matrix':
                    fusion_metrics[metric].append(value)
        
        results = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in fusion_metrics.items()
        }
        
        logging.info("Multimodal fusion evaluation results:")
        for metric, stats in results.items():
            logging.info(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return results
    
    except Exception as e:
        logging.error(f"Error during multimodal fusion analysis: {str(e)}")
        raise

def feature_importance_analysis(X, y, feature_names):

    try:

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            scale_pos_weight=len(y[y==0])/len(y[y==1])
        )
        
        xgb.fit(X_resampled, y_resampled)
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb.feature_importances_
        })
        
        importance = importance.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importance.head(20))
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig('2_plot/feature_importance.png')
        
        importance.to_csv('2_data/feature_importance.csv', index=False)
        
        logging.info("Feature importance analysis completed")
        logging.info("Top 10 important features:")
        logging.info(importance.head(10))
        
        return importance
    
    except Exception as e:
        logging.error(f"Error during feature importance analysis: {str(e)}")
        raise

def model_interpretability_analysis(X, y, feature_names):
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            scale_pos_weight=len(y[y==0])/len(y[y==1])
        )

        xgb.fit(X_resampled, y_resampled)
        
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_resampled)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_resampled, feature_names=feature_names, show=False)
        plt.title('SHAP Summary Plot')
        plt.tight_layout()
        plt.savefig('2_plot/shap_summary.png')
        
        top_features = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=False).head(3)['feature'].tolist()
        
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature,
                shap_values,
                X_resampled,
                feature_names=feature_names,
                show=False
            )
            plt.title(f'SHAP Dependence Plot for {feature}')
            plt.tight_layout()
            plt.savefig(f'2_plot/shap_dependence_{feature}.png')
    
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.to_csv('2_plot/shap_values.csv', index=False)

        
        importance = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': np.abs(shap_values).mean(0)
        }).sort_values('shap_importance', ascending=False)
        
        logging.info("Model interpretability analysis completed")
        logging.info("Top 10 important features (based on SHAP values):")
        logging.info(importance.head(10))
        
        return importance
    
    except Exception as e:
        logging.error(f"Error during model interpretability analysis: {str(e)}")
        raise

def main():
    try:
        logging.info("Starting to load and preprocess data...")
        X, y, feature_names = load_and_preprocess_data()
        
        logging.info("Performing single modality analysis...")
        single_results = single_modality_analysis(X, y, feature_names)
        logging.info(f"Single modality analysis results: {single_results}")
        
        logging.info("Performing multimodal fusion analysis...")
        fusion_results = multimodal_fusion(X, y)
        logging.info(f"Multimodal fusion analysis results: {fusion_results}")
        
        logging.info("Performing feature importance analysis...")
        feature_importance = feature_importance_analysis(X, y, feature_names)
        
        logging.info("Performing model interpretability analysis...")
        model_interpretability = model_interpretability_analysis(X, y, feature_names)
        
        
        results = {
            'single_modality': single_results,
            'multimodal_fusion': fusion_results,
            'feature_importance': feature_importance.to_dict(),
            'model_interpretability': model_interpretability.to_dict()
        }
        
        pd.DataFrame(results).to_csv('2_data/analysis_results.csv')
        logging.info("Analysis completed. Results saved to 2_data/analysis_results.csv")
    
    except Exception as e:
        logging.error(f"Error during program execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
