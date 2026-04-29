# Credit Card Fraud Detection - Flowchart

```mermaid
flowchart TD
    A[Start Project] --> B[Install Dependencies]
    B --> C[Run train_model.py]

    C --> D[Generate Synthetic Dataset]
    D --> E[Split Data Train/Test]
    E --> F[Scale Features with StandardScaler]
    F --> G[Train Logistic Regression Baseline]
    G --> H[Train Random Forest Main Model]
    H --> I[Evaluate Metrics and 5-Fold CV]
    I --> J[Save Artifacts]

    J --> J1[model/fraud_model.pkl]
    J --> J2[model/scaler.pkl]
    J --> J3[model/meta.json]

    J1 --> K[Run streamlit run app.py]
    J2 --> K
    J3 --> K

    K --> L[Load Model + Scaler + Metadata]
    L --> M[User Inputs Transaction Features]
    M --> N[Scale Input Features]
    N --> O[Predict Fraud Probability]
    O --> P{Probability > 0.5?}

    P -- Yes --> Q[Classify as Fraud]
    P -- No --> R[Classify as Legit]

    Q --> S[Assign Risk Tier + Show Alert]
    R --> S
    S --> T[Display Performance, Importance, Pipeline Tabs]
    T --> U[End]
```

## Short Explanation

1. We first train the model offline using `train_model.py`.
2. The script creates synthetic imbalanced fraud data and preprocesses it.
3. It trains two algorithms: Logistic Regression (baseline) and Random Forest (final model).
4. It evaluates performance using ROC-AUC, F1, Recall, Precision, confusion matrix, and 5-fold cross-validation.
5. It saves the trained model, scaler, and metadata in the `model/` folder.
6. The Streamlit app (`app.py`) loads these files and performs real-time fraud prediction from user input.
