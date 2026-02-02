"""
Machine Learning Assignment 2 - Model Training Script
Train and evaluate 6 classification models on Telco Customer Churn dataset
"""


import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

def load_and_preprocess_data(filepath):
    """Load and preprocess the Telco Customer Churn dataset"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.shape[1] - 1}")
    print(f"Instances: {df.shape[0]}")
    
    # Handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Drop customerID as it's not a feature
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Balance the dataset by oversampling the minority class
    print("\nBalancing the dataset by oversampling the minority class...")
    df_full = X.copy()
    df_full['Churn'] = y
    class_counts = df_full['Churn'].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    n_majority = class_counts.max()
    n_minority = class_counts.min()
    df_majority = df_full[df_full['Churn'] == majority_class]
    df_minority = df_full[df_full['Churn'] == minority_class]
    df_minority_upsampled = df_minority.sample(n=n_majority, replace=True, random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    X = df_balanced.drop('Churn', axis=1)
    y = df_balanced['Churn']

    print(f"\nFeatures after preprocessing: {X.shape[1]}")
    print(f"Class distribution after balancing:\n{y.value_counts()}")
    
    return X, y, label_encoders

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train all 6 models and calculate evaluation metrics"""
    
    # Scale features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define all models with improved hyperparameters and class_weight
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced', C=0.7),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=14, min_samples_split=3, class_weight='balanced'),
        'KNN': KNeighborsClassifier(n_neighbors=9, weights='distance', p=2),
        'Naive Bayes': GaussianNB(var_smoothing=1e-2),
        'Random Forest': RandomForestClassifier(n_estimators=250, random_state=42, max_depth=12, class_weight='balanced_subsample'),
        'XGBoost': XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss', learning_rate=0.07)
    }

    results = {}
    trained_models = {}

    print("\n" + "="*80)
    print("TRAINING AND EVALUATING MODELS (with tuning)")
    print("="*80)

    for name, model in models.items():
        print(f"\n{'='*80}")
        print(f"Training {name}...")
        print(f"{'='*80}")

        # Train model
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Store results
        results[name] = {
            'Accuracy': accuracy,
            'AUC': auc,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'MCC': mcc,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }

        trained_models[name] = model

        # Print metrics
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"AUC Score: {auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"MCC Score: {mcc:.4f}")
        print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return results, trained_models, scaler

def save_models(trained_models, scaler, label_encoders):
    """Save trained models and preprocessors"""
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    
    for name, model in trained_models.items():
        filename = f"model/{name.replace(' ', '_').lower()}_model.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved: {filename}")
    
    # Save scaler
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved: model/scaler.pkl")
    
    # Save label encoders
    with open('model/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("Saved: model/label_encoders.pkl")

def print_results_table(results):
    """Print formatted results table"""
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    
    df_results = pd.DataFrame(results).T
    df_results = df_results[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]
    
    print("\n" + df_results.to_string())
    
    # Save to CSV
    df_results.to_csv('model/model_results.csv')
    print("\nResults saved to: model/model_results.csv")
    
    return df_results

def main():
    """Main execution function"""
    print("="*80)
    print("MACHINE LEARNING ASSIGNMENT 2")
    print("Telco Customer Churn Prediction")
    print("="*80)
    
    # Load and preprocess data
    X, y, label_encoders = load_and_preprocess_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Split data
    print("\nSplitting data into train and test sets (80-20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train and evaluate models
    results, trained_models, scaler = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )
    
    # Print results table
    df_results = print_results_table(results)
    
    # Save models
    save_models(trained_models, scaler, label_encoders)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nAll models have been trained and saved successfully.")
    print("You can now run the Streamlit app using: streamlit run app.py")

if __name__ == "__main__":
    main()
