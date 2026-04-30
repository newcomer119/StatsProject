# Credit Card Fraud Detection (Streamlit + Scikit-learn)

This project demonstrates a complete supervised machine learning workflow for fraud detection:

1. Train a fraud classifier on synthetic data (`train_model.py`).
2. Save trained artifacts (`model/` folder).
3. Run a real-time prediction dashboard (`app.py`) using Streamlit.

## Algorithms Used

- Logistic Regression (baseline model)
- Random Forest Classifier (main selected model)
- StandardScaler (feature preprocessing)
- Stratified 5-Fold Cross-Validation (model stability check)

## End-to-End Workflow

1. **Data generation**  
   Create synthetic legitimate and fraudulent transactions with class imbalance.

2. **Preprocessing**  
   Split train/test and scale features with `StandardScaler`.

3. **Baseline training**  
   Train Logistic Regression for comparison.

4. **Main model training**  
   Train Random Forest (with `class_weight="balanced"`).

5. **Evaluation**  
   Compute ROC-AUC, F1, Precision, Recall, confusion matrix, and CV AUC.

6. **Save artifacts**  
   Save model (`fraud_model.pkl`), scaler (`scaler.pkl`), and metadata (`meta.json`).

7. **Deployment UI**  
   Streamlit app loads artifacts and predicts fraud probability from user inputs.

## Project Structure

```text
StatsProject/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── FLOWCHART.md
└── model/
    ├── fraud_model.pkl
    ├── scaler.pkl
    └── meta.json
```

## Setup and Run

Run from project root:

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

App URL: `http://localhost:8501`

## Prediction Logic in App

When the user clicks **CLASSIFY TRANSACTION**:

1. Collect 10 input features (`V1, V3, V4, V7, V10, V12, V14, V17, V28, Amount`).
2. Scale input with saved `scaler.pkl`.
3. Predict class probabilities using `fraud_model.pkl`.
4. Use `P(fraud) > 0.5` as decision threshold.
5. Show class result, risk tier, probability bar, and recommendation.

## Evaluation Metrics Used

- ROC-AUC
- F1-score (fraud class)
- Precision (fraud class)
- Recall (fraud class)
- Accuracy
- Confusion Matrix
- 5-Fold CV Mean/Std AUC

## Important Note for Viva/Presentation

The implementation currently uses:

- Synthetic data generation (not direct Kaggle file ingestion)
- Logistic Regression + Random Forest

Some dashboard text references advanced items (such as SMOTE/XGBoost/FastAPI), but those are presentation labels and are not fully implemented in the current training script.

