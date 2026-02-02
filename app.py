"""
Streamlit Web Application for Telco Customer Churn Prediction
Machine Learning Assignment 2
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_name):
    """Load a trained model from pickle file"""
    try:
        model_path = f"model/{model_name.replace(' ', '_').lower()}_model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Load the fitted scaler"""
    try:
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

def preprocess_data(df):
    """Preprocess the uploaded dataset"""
    # Handle missing values
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Drop customerID if present
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Encode target variable if present
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        y = df['Churn']
        X = df.drop('Churn', axis=1)
    else:
        y = None
        X = df.copy()
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    return X, y

def display_metrics(y_true, y_pred, y_pred_proba, model_name):
    """Display evaluation metrics"""
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Display metrics in columns
    st.subheader("📊 Evaluation Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("AUC Score", f"{auc:.4f}")
    
    with col2:
        st.metric("Precision", f"{precision:.4f}")
        st.metric("Recall", f"{recall:.4f}")
    
    with col3:
        st.metric("F1 Score", f"{f1:.4f}")
        st.metric("MCC Score", f"{mcc:.4f}")
    
    return accuracy, auc, precision, recall, f1, mcc

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'],
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    
    st.pyplot(fig)
    
    return cm

def display_classification_report(y_true, y_pred):
    """Display classification report"""
    report = classification_report(y_true, y_pred, 
                                   target_names=['No Churn', 'Churn'],
                                   output_dict=True)
    
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.format("{:.4f}"))

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Telco Customer Churn Prediction</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Machine Learning Assignment 2")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.markdown("### Model Selection")
    
    # Model selection dropdown
    model_options = [
        'Logistic Regression',
        'Decision Tree',
        'KNN',
        'Naive Bayes',
        'Random Forest',
        'XGBoost'
    ]
    
    selected_model = st.sidebar.selectbox(
        "Choose a Classification Model:",
        model_options
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    This application predicts customer churn using various machine learning models.
    
    **Dataset:** Telco Customer Churn
    
    **Models Available:**
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors
    - Naive Bayes
    - Random Forest (Ensemble)
    - XGBoost (Ensemble)
    """)
    
    # Main content
    st.subheader("📁 Upload Test Dataset")
    st.markdown("Upload your CSV file containing customer data for churn prediction.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload the test dataset in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            # Show data preview
            with st.expander("👁️ View Data Preview"):
                st.dataframe(df.head(10))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Dataset Info:**")
                    st.write(f"- Rows: {df.shape[0]}")
                    st.write(f"- Columns: {df.shape[1]}")
                
                with col2:
                    st.write("**Columns:**")
                    st.write(df.columns.tolist())
            
            # Preprocess data
            with st.spinner("Preprocessing data..."):
                X, y = preprocess_data(df)
            
            if y is None:
                st.warning("⚠️ No 'Churn' column found. Cannot evaluate model performance.")
                st.info("The uploaded dataset should contain a 'Churn' column for evaluation.")
                return
            
            # Load model and scaler
            with st.spinner(f"Loading {selected_model} model..."):
                model = load_model(selected_model)
                scaler = load_scaler()
            
            if model is None or scaler is None:
                st.error("❌ Failed to load model or scaler. Please ensure models are trained.")
                st.info("Run `python model/train_models.py` to train the models first.")
                return
            
            # Make predictions
            with st.spinner("Making predictions..."):
                X_scaled = scaler.transform(X)
                y_pred = model.predict(X_scaled)
                y_pred_proba = model.predict_proba(X_scaled)[:, 1]
            
            st.success("✅ Predictions completed!")
            
            # Display results
            st.markdown("---")
            st.header(f"📈 Results for {selected_model}")
            
            # Display metrics
            metrics = display_metrics(y, y_pred, y_pred_proba, selected_model)
            
            st.markdown("---")
            
            # Display confusion matrix and classification report
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🔢 Confusion Matrix")
                cm = plot_confusion_matrix(y, y_pred, selected_model)
            
            with col2:
                st.subheader("📋 Classification Report")
                display_classification_report(y, y_pred)
            
            # Prediction distribution
            st.markdown("---")
            st.subheader("📊 Prediction Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                pred_counts = pd.Series(y_pred).value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                pred_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
                ax.set_xticklabels(['No Churn', 'Churn'], rotation=0)
                ax.set_xlabel('Prediction')
                ax.set_ylabel('Count')
                ax.set_title('Predicted Classes')
                st.pyplot(fig)
            
            with col2:
                actual_counts = y.value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                actual_counts.plot(kind='bar', ax=ax, color=['#3498db', '#e67e22'])
                ax.set_xticklabels(['No Churn', 'Churn'], rotation=0)
                ax.set_xlabel('Actual')
                ax.set_ylabel('Count')
                ax.set_title('Actual Classes')
                st.pyplot(fig)
            
            # Download predictions
            st.markdown("---")
            st.subheader("💾 Download Predictions")
            
            result_df = df.copy()
            result_df['Predicted_Churn'] = ['Yes' if p == 1 else 'No' for p in y_pred]
            result_df['Churn_Probability'] = y_pred_proba
            
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f"predictions_{selected_model.replace(' ', '_').lower()}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("Please ensure your CSV file has the correct format.")
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a CSV file to begin prediction.")
        
        st.markdown("### 📝 Dataset Requirements:")
        st.markdown("""
        - The dataset should be in CSV format
        - It should contain customer features similar to the Telco Churn dataset
        - Include a 'Churn' column (Yes/No) for evaluation
        - Minimum recommended size: 100 rows for meaningful evaluation
        """)
        
        st.markdown("### 🎯 Expected Columns:")
        expected_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges', 'Churn'
        ]
        st.code(", ".join(expected_cols))

if __name__ == "__main__":
    main()
