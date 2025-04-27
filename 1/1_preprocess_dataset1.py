import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def preprocess_clinical_data():
    # Load dataset
    file_path = Path("1/clinical1.csv")
    df = pd.read_csv(file_path)
    
    # Overview of dataset
    print("=== Dataset Overview ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Handle continuous features
    cont_features = ['age', 'Survival.time']
    df[cont_features] = df[cont_features].apply(lambda x: x.fillna(x.median()))
    print("\nFilled missing values in continuous features with median.")

    # Handle categorical features
    cat_features = ['clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage', 
                    'Overall.Stage', 'Histology', 'gender', 'deadstatus.event']
    for col in cat_features:
        df[col] = df[col].astype(str).fillna('Unknown').replace('nan', 'Unknown')
    
    print("\nFilled missing values in categorical features with 'Unknown'.")

    # Standardize continuous features (keep original)
    scaler = StandardScaler()
    for col in cont_features:
        df[f"{col}_raw"] = df[col]
        df[col] = scaler.fit_transform(df[[col]])
        print(f"\n{col} standardized: Mean={df[col].mean():.4f}, Std={df[col].std():.4f}")

    # One-hot encode categorical features (drop first level)
    df_encoded = pd.get_dummies(df, columns=cat_features, prefix=cat_features, drop_first=True)
    added_cols = [col for col in df_encoded.columns if col not in df.columns]

    print("\n=== One-hot Encoding Summary ===")
    print(f"Total new dummy variables: {len(added_cols)}")
    print("Example encoded columns:", added_cols[:5], "...")

    # Save processed data
    output_dir = Path("1_data")
    output_dir.mkdir(exist_ok=True)
    df_encoded.to_csv(output_dir / "clinical1_preprocessed.csv", index=False)
    
    print("\nProcessed data saved to '1_data/clinical1_preprocessed.csv'.")
    print(f"Final shape: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns.")

    return df_encoded

if __name__ == "__main__":
    final_df = preprocess_clinical_data()

