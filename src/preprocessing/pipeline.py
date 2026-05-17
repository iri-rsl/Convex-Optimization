"""
Data preprocessing pipeline for Student Depression and Lifestyle dataset.
Handles loading, cleaning, encoding, scaling, and transformation.

Dataset Schema:
- Student_ID (Integer): Unique identifier for each student
- Age (Integer): Age of student (18-24)
- Gender (String): Male/Female
- Department (String): Engineering, Business, Arts, etc.
- CGPA (Float): Cumulative Grade Point Average (0.0-4.0)
- Sleep_Duration (Float): Average hours of sleep per night
- Study_Hours (Float): Average hours spent studying per day
- Social_Media_Hours (Float): Average hours on social media per day
- Physical_Activity (Integer): Average minutes of activity per week
- Stress_Level (Integer): Self-reported stress (0-10)
- Transportation_Time (Integer): Simulated daily commute time
- Student_Debt (Integer): Binary socio-economic indicator
- Part_Time_Job (Integer): Binary socio-economic indicator
- Living_Status (String): Family or Seul
- Depression (Boolean): True = Probable Depression, False = Healthy
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import kagglehub


def create_directories():
    """Create necessary project directories if they don't exist."""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    print("[LOG] Directories created: data/raw, data/processed")


def load_dataset():
    """Load the Student Depression and Lifestyle dataset from Kaggle."""
    print("[LOG] Loading dataset from Kaggle...")
    
    # Download latest version
    path = kagglehub.dataset_download("aldinwhyudii/student-depression-and-lifestyle-100k-data")
    print(f"[LOG] Path to dataset files: {path}")
    
    # Load the dataset
    df = pd.read_csv(f"{path}/student_lifestyle_100k.csv")
    print(f"[LOG] Dataset loaded with shape: {df.shape}")
    return df


def inject_socio_economic_features(df):
    """
    Simulate and inject socio-economic variables from the project brief.
    """
    print("[SIMULATION] Injecting socio-economic variables...")
    np.random.seed(42)
    n_samples = len(df)

    # 1. Daily transportation time (log-normal around ~35 mins, capped at 120)
    base_transport = np.random.lognormal(mean=3.4, sigma=0.4, size=n_samples)
    stress_modifier = df['Stress_Level'].values * 2.5
    transport_time = base_transport + stress_modifier
    df['Transportation_Time'] = np.clip(transport_time, 10, 120).astype(int)

    # Realistic effect: long commutes reduce sleep duration
    long_commute_mask = df['Transportation_Time'] > 60
    df.loc[long_commute_mask, 'Sleep_Duration'] = (
        df.loc[long_commute_mask, 'Sleep_Duration']
        - np.random.uniform(0.5, 1.5, size=long_commute_mask.sum())
    )
    df['Sleep_Duration'] = np.clip(df['Sleep_Duration'], 3, 12)

    # 2. Student debt / bank loan indicator
    proba_debt = np.where(df['Stress_Level'].values > 7, 0.55, 0.25)
    df['Student_Debt'] = np.random.binomial(1, proba_debt).astype(int)

    # 3. Part-time job indicator
    proba_job = np.where((df['Student_Debt'].values == 1) & (df['Stress_Level'].values > 5), 0.60, 0.20)
    df['Part_Time_Job'] = np.random.binomial(1, proba_job).astype(int)

    # Realistic effect: part-time job slightly lowers CGPA
    job_mask = df['Part_Time_Job'] == 1
    df.loc[job_mask, 'CGPA'] = (
        df.loc[job_mask, 'CGPA']
        - np.random.uniform(0.1, 0.4, size=job_mask.sum())
    )
    df['CGPA'] = np.clip(df['CGPA'], 0.0, 4.0)

    # 4. Living status
    living_choices = ['Famille', 'Coloc', 'Seul']
    df['Living_Status'] = 'Famille'

    high_constraint = (df['Part_Time_Job'] == 1) | (df['Student_Debt'] == 1)
    df.loc[high_constraint, 'Living_Status'] = np.random.choice(
        living_choices,
        size=high_constraint.sum(),
        p=[0.2, 0.5, 0.3]
    )
    df.loc[~high_constraint, 'Living_Status'] = np.random.choice(
        living_choices,
        size=(~high_constraint).sum(),
        p=[0.6, 0.2, 0.2]
    )

    print("[SIMULATION] Added: Transportation_Time, Student_Debt, Part_Time_Job, Living_Status")
    return df


def quality_control(df):
    """
    Perform quality control: check missing values and drop Student_ID.
    """
    print("[LOG] Running quality control...")
    print(f"[LOG] Missing values per column:\n{df.isnull().sum()}")
    
    # Drop Student_ID column
    if 'Student_ID' in df.columns:
        df = df.drop(columns=['Student_ID'])
        print("[LOG] Dropped 'Student_ID' column")
    
    return df


def encode_categorical(df):
    """
    One-Hot encode categorical columns: Gender, Department, and Living_Status.
    Uses drop='first' to avoid multicollinearity.
    """
    print("[LOG] Encoding categorical features...")
    categorical_cols = ['Gender', 'Department', 'Living_Status']
    
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_features = encoder.fit_transform(df[categorical_cols])
    
    # Get feature names from encoder
    feature_names = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
    
    # Drop original categorical columns and concatenate encoded features
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)
    
    print(f"[LOG] Categorical encoding complete. New shape: {df.shape}")
    print(f"[LOG] Encoded columns: {list(feature_names)}")
    
    return df


def standardize_numeric_features(df):
    """
    Apply StandardScaler to numeric features for convergence acceleration.
    Essential for SGD-based algorithms.
    """
    print("[LOG] Standardizing numeric features...")
    numeric_cols = [
        'Age', 'CGPA', 'Sleep_Duration', 'Study_Hours',
        'Social_Media_Hours', 'Physical_Activity', 'Stress_Level',
        'Transportation_Time'
    ]
    
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    print("[LOG] Numeric standardization complete.")
    print(f"[LOG] Feature statistics after scaling:\n{df[numeric_cols].describe().round(2)}")
    
    return df


def transform_target_variable(df):
    """
    Convert Depression column (Boolean) to binary {-1, 1} format.
    Required for Hinge Loss formulation: max(0, 1 - y_i(w^T x_i + b))
    """
    print("[LOG] Transforming target variable to {-1, 1}...")
    
    # Map: True -> 1, False -> -1
    df['Depression'] = df['Depression'].map({True: 1, False: -1})
    
    print(f"[LOG] Target variable transformation complete.")
    print(f"[LOG] Class distribution:\n{df['Depression'].value_counts()}")
    
    return df


def save_processed_data(df, output_path="data/processed/clean_student_data.csv"):
    """
    Save the processed DataFrame to CSV format.
    Reorders columns to place Depression (target) at the end.
    """
    # Reorder columns: move Depression to the end
    cols = [col for col in df.columns if col != 'Depression']
    df = df[cols + ['Depression']]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[LOG] Processed data saved to: {output_path}")
    print(f"[LOG] Final dataset shape: {df.shape}")
    print(f"[LOG] Column order: {list(df.columns)}")


def main():
    """
    Execute the complete preprocessing pipeline.
    """
    print("\n" + "="*70)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Load dataset
    df = load_dataset()
    
    # Step 3: Inject socio-economic features
    df = inject_socio_economic_features(df)

    # Step 4: Quality control
    df = quality_control(df)
    
    # Step 5: Encode categorical features
    df = encode_categorical(df)
    
    # Step 6: Standardize numeric features
    df = standardize_numeric_features(df)
    
    # Step 7: Transform target variable
    df = transform_target_variable(df)
    
    # Step 8: Save processed data
    save_processed_data(df)
    
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")
    
    return df


if __name__ == "__main__":
    df = main()
