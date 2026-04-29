# 🛡️ Credit Card Fraud Detection — Streamlit Demo

**Capstone Project | Mitarth Pandey & Devansh Bharat Lalwani**

---

## ⚡ Quick Setup (5 minutes)

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model (run once)
```bash
python train_model.py
```
You'll see training metrics printed in your terminal.

### Step 3 — Launch the Streamlit app
```bash
streamlit run app.py
```

Your browser will open automatically at **http://localhost:8501**

---

## 🎯 Demo Script (for presentation tomorrow)

### Show 1 — Legitimate transaction
1. In the sidebar, select **"🟢 Normal Purchase ($45)"**
2. Click **CLASSIFY TRANSACTION**
3. Show the green ✅ result, probability bar, and feature contributions

### Show 2 — Fraud detected  
1. In the sidebar, select **"🔴 Suspicious Pattern"**
2. Click **CLASSIFY TRANSACTION**
3. Show the red ⚠️ FRAUD DETECTED alert with animated pulse border

### Show 3 — Borderline case
1. Select **"🟡 Borderline Case"**
2. Show Medium Risk tier

### Show 4 — Model Performance tab
1. Click **📈 Model Performance** tab
2. Show Confusion Matrix, model comparison table, Feature Importance chart

### Show 5 — Explain the ML pipeline
1. Click **📚 How It Works** tab
2. Walk through the 6-step pipeline

---

## 🧠 Key Points to Mention in Presentation

- **Binary Classification** problem: Class 0 (Legit) vs Class 1 (Fraud)
- **Class Imbalance**: Only 0.17% fraud — handled with `class_weight='balanced'`
- **Features**: V1–V28 are PCA-transformed (anonymized), plus Amount
- **Why Random Forest?**: Ensemble method, handles non-linearity, gives feature importance
- **Validation**: ROC-AUC > 0.95, 5-Fold Cross-Validation, Confusion Matrix
- **Real-time**: Predictions return in milliseconds

---

## 📁 Project Structure
```
fraud_detection/
├── app.py              ← Streamlit dashboard
├── train_model.py      ← Model training script
├── requirements.txt    ← Dependencies
├── README.md           ← This file
└── model/              ← Created after training
    ├── fraud_model.pkl
    ├── scaler.pkl
    └── meta.json
```

---

*Built for Capstone Presentation | Machine Learning — Supervised Classification*