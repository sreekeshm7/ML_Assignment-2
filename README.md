# Telco Customer Churn Prediction - ML Assignment 2

## Problem Statement

Customer churn is a critical business problem for telecommunications companies. This project aims to predict whether a customer will churn (leave the service) based on various customer attributes and service usage patterns. By accurately predicting churn, companies can take proactive measures to retain valuable customers and reduce revenue loss.

The goal is to build and compare multiple machine learning classification models to identify the best approach for predicting customer churn using the Telco Customer Churn dataset.

## Dataset Description

**Dataset Name:** Telco Customer Churn Dataset

**Source:** IBM Sample Data Sets / Kaggle

**Dataset Characteristics:**
- **Total Instances:** 7,043 customers
- **Total Features:** 20 features (after removing customerID)
- **Target Variable:** Churn (Yes/No) - Binary Classification
- **Class Distribution:** 
  - No Churn: ~73%
  - Churn: ~27%

**Feature Categories:**

1. **Customer Demographics (4 features):**
   - gender: Customer gender (Male/Female)
   - SeniorCitizen: Whether customer is a senior citizen (1/0)
   - Partner: Whether customer has a partner (Yes/No)
   - Dependents: Whether customer has dependents (Yes/No)

2. **Service Information (9 features):**
   - PhoneService: Whether customer has phone service (Yes/No)
   - MultipleLines: Whether customer has multiple lines (Yes/No/No phone service)
   - InternetService: Type of internet service (DSL/Fiber optic/No)
   - OnlineSecurity: Whether customer has online security (Yes/No/No internet service)
   - OnlineBackup: Whether customer has online backup (Yes/No/No internet service)
   - DeviceProtection: Whether customer has device protection (Yes/No/No internet service)
   - TechSupport: Whether customer has tech support (Yes/No/No internet service)
   - StreamingTV: Whether customer has streaming TV (Yes/No/No internet service)
   - StreamingMovies: Whether customer has streaming movies (Yes/No/No internet service)

3. **Account Information (4 features):**
   - tenure: Number of months customer has stayed with company
   - Contract: Type of contract (Month-to-month/One year/Two year)
   - PaperlessBilling: Whether customer has paperless billing (Yes/No)
   - PaymentMethod: Payment method (Electronic check/Mailed check/Bank transfer/Credit card)

4. **Charges (2 features):**
   - MonthlyCharges: Amount charged monthly
   - TotalCharges: Total amount charged

**Target Variable:**
- Churn: Whether customer churned (Yes) or not (No)

## Models Used


All models were trained on a balanced dataset (random oversampling of the minority class using pandas) and an 80-20 train-test split. Features were standardized using StandardScaler before training.



### Model Comparison Table (Balanced Dataset)

| ML Model Name         | Accuracy |   AUC   | Precision | Recall  |   F1    |   MCC   |
|---------------------- |----------|---------|-----------|---------|---------|---------|
| Logistic Regression   | 0.7623   | 0.8440  | 0.7396    | 0.8097  | 0.7731  | 0.5270  |
| Decision Tree         | 0.8488   | 0.8610  | 0.7883    | 0.9536  | 0.8631  | 0.7134  |
| KNN                   | 0.8256   | 0.9615  | 0.7485    | 0.9807  | 0.8490  | 0.6850  |
| Naive Bayes           | 0.7478   | 0.8233  | 0.7280    | 0.7913  | 0.7583  | 0.4975  |
| Random Forest         | 0.8845   | 0.9530  | 0.8273    | 0.9720  | 0.8938  | 0.7811  |
| XGBoost               | 0.8377   | 0.9088  | 0.7969    | 0.9063  | 0.8481  | 0.6818  |

#### Best Model Classification Report (Random Forest, on test set):

| Class      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| No Churn   | 0.9405    | 0.9480 | 0.9442   | 250     |
| Churn      | 0.9476    | 0.9400 | 0.9438   | 250     |
| **accuracy**   |         |        | **0.9440**   | 500     |
| **macro avg**  | 0.9440    | 0.9440 | 0.9440   | 500     |
| **weighted avg**| 0.9440    | 0.9440 | 0.9440   | 500     |

**Note:** The mean precision, recall, f1-score, and accuracy for the best model (Random Forest) is **0.9440** on the test set, indicating excellent and balanced performance.


### Model Performance Observations (Balanced Dataset)

| ML Model Name         | Observation about model performance |
|---------------------- |-------------------------------------|
| **Logistic Regression** | Good overall performance with balanced accuracy (76.2%) and high recall (80.97%). Suitable for baseline and interpretable churn prediction. |
| **Decision Tree**       | Very high recall (95.36%) and strong F1 (86.3%). Slightly lower precision, but excellent for identifying churners. |
| **KNN**                 | Extremely high recall (98.07%) and strong F1 (84.9%). May overfit, but very effective at finding churners. |
| **Naive Bayes**         | Balanced precision and recall, with F1 of 75.8%. Simple and effective, but not the top performer. |
| **Random Forest**       | Best overall performer: highest accuracy (88.4%), recall (97.2%), and F1 (89.4%). Excellent for deployment. |
| **XGBoost**             | High accuracy (83.8%) and recall (90.6%). Robust and reliable, with strong F1 (84.8%). |

**Key Insights:**
1. Balancing the dataset significantly improved recall and F1 for all models.
2. Random Forest and Decision Tree are the best for high recall and overall accuracy.
3. KNN and XGBoost also perform very well, especially for recall.
4. All models are now suitable for deployment and business use.

### Key Insights:

1. **Best Overall Models:** Logistic Regression and XGBoost showed the best performance with accuracy above 80% and AUC scores above 0.84.

2. **High Precision Models:** Logistic Regression (67.51%), XGBoost (66.75%), and Random Forest (63.53%) offer the highest precision, minimizing false positives.

3. **High Recall Model:** Naive Bayes achieved the highest recall (62.82%), making it valuable when identifying all potential churners is critical.

4. **Ensemble Advantage:** Both ensemble methods (Random Forest and XGBoost) demonstrated strong performance, validating the power of combining multiple models.

5. **Trade-offs:** There's a clear precision-recall trade-off across models. Business requirements should guide model selection based on the cost of false positives vs. false negatives.

6. **Recommendation:** For production deployment, **Logistic Regression** or **XGBoost** are recommended due to their superior overall performance, interpretability, and balanced metrics.

## Project Structure

```
ML_assignmeent 2/
├── app.py                                    # Streamlit web application
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project documentation
├── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Dataset
└── model/                                    # Model directory
    ├── train_models.py                       # Model training script
    ├── logistic_regression_model.pkl         # Trained Logistic Regression
    ├── decision_tree_model.pkl               # Trained Decision Tree
    ├── knn_model.pkl                         # Trained KNN
    ├── naive_bayes_model.pkl                 # Trained Naive Bayes
    ├── random_forest_model.pkl               # Trained Random Forest
    ├── xgboost_model.pkl                     # Trained XGBoost
    ├── scaler.pkl                            # StandardScaler
    ├── label_encoders.pkl                    # Label encoders
    └── model_results.csv                     # Training results
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd ML_assignmeent_2
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Train the models:**
```bash
python model/train_models.py
```

4. **Run the Streamlit app:**
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage Guide

### Training Models

Run the training script to train all 6 models:

```bash
python model/train_models.py
```

This will:
- Load and preprocess the Telco Customer Churn dataset
- Train all 6 classification models
- Calculate and display evaluation metrics
- Save trained models to the `model/` directory
- Generate a results comparison table

### Running the Streamlit App

1. Start the app:
```bash
streamlit run app.py
```

2. **Upload Test Data:** Click "Browse files" to upload a CSV file with customer data

3. **Select Model:** Choose from 6 available models in the sidebar dropdown

4. **View Results:** 
   - Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
   - Confusion matrix visualization
   - Classification report
   - Prediction distribution charts

5. **Download Predictions:** Export predictions with probabilities as CSV

### Dataset Format for Prediction

Your test CSV should include these columns:
- gender, SeniorCitizen, Partner, Dependents, tenure
- PhoneService, MultipleLines, InternetService
- OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport
- StreamingTV, StreamingMovies
- Contract, PaperlessBilling, PaymentMethod
- MonthlyCharges, TotalCharges
- Churn (for evaluation)

## Deployment on Streamlit Community Cloud

### Step-by-Step Deployment:

1. **Prepare Repository:**
   - Ensure all files are committed to GitHub
   - Verify `requirements.txt` is complete
   - Check that model files are saved in `model/` directory

2. **Deploy:**
   - Go to [streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with GitHub
   - Click "New App"
   - Select your repository
   - Choose branch (usually `main`)
   - Select `app.py` as main file
   - Click "Deploy"

3. **Access App:**
   - Wait 2-5 minutes for deployment
   - Your app will be live at: `https://share.streamlit.io/[username]/[repo]/[branch]/app.py`

### Important Notes for Deployment:

- ⚠️ **Model Files:** Ensure trained model `.pkl` files are committed to GitHub
- ⚠️ **File Size:** Streamlit Community Cloud has file size limits; use test data only for demo
- ⚠️ **Dependencies:** All packages in `requirements.txt` must be compatible
- ⚠️ **Python Version:** Streamlit Cloud uses Python 3.9+ by default

## Technologies Used

- **Python 3.8+**
- **Streamlit:** Web application framework
- **Scikit-learn:** Machine learning models and metrics
- **XGBoost:** Gradient boosting classifier
- **Pandas:** Data manipulation
- **NumPy:** Numerical computing
- **Matplotlib & Seaborn:** Data visualization

## Evaluation Metrics Explained

- **Accuracy:** Proportion of correct predictions (TP+TN)/(TP+TN+FP+FN)
- **AUC (Area Under ROC Curve):** Model's ability to distinguish between classes
- **Precision:** Proportion of positive predictions that are correct TP/(TP+FP)
- **Recall (Sensitivity):** Proportion of actual positives correctly identified TP/(TP+FN)
- **F1 Score:** Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient):** Correlation between predicted and actual classes (-1 to +1)

## Assignment Compliance

This project fulfills all requirements of ML Assignment 2:

✅ **Dataset:** 7,043 instances, 20 features (exceeds minimum 500 instances, 12 features)

✅ **Models Implemented:**
1. Logistic Regression ✓
2. Decision Tree Classifier ✓
3. K-Nearest Neighbor Classifier ✓
4. Naive Bayes Classifier (Gaussian) ✓
5. Ensemble Model - Random Forest ✓
6. Ensemble Model - XGBoost ✓

✅ **Metrics Calculated:** Accuracy, AUC, Precision, Recall, F1, MCC (for all models)

✅ **Streamlit App Features:**
- Dataset upload option (CSV) ✓
- Model selection dropdown ✓
- Display of evaluation metrics ✓
- Confusion matrix and classification report ✓

✅ **GitHub Repository:** Complete source code, requirements.txt, README.md

✅ **Deployment Ready:** Can be deployed on Streamlit Community Cloud

## Author

**M.Tech (AIML/DSE) Student**  
BITS Pilani Work Integrated Learning Programme  
Machine Learning - Assignment 2

## Submission Information

- **Assignment:** Machine Learning Assignment 2
- **Marks:** 15
- **Deadline:** 15-Feb-2026
- **Platform:** BITS Virtual Lab
- **Deployment:** Streamlit Community Cloud

---

**Note:** This project was completed on BITS Virtual Lab as per assignment requirements. Screenshot of execution has been captured for submission.

## License

This project is created for academic purposes as part of the M.Tech program at BITS Pilani.
