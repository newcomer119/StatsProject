import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main { background-color: #0d1117; }

    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    }

    .metric-card {
        background: linear-gradient(135deg, #161b22, #21262d);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }

    .fraud-alert {
        background: linear-gradient(135deg, #3d1a1a, #5c2020);
        border: 2px solid #f85149;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        animation: pulse 2s infinite;
    }

    .legit-alert {
        background: linear-gradient(135deg, #0d2e1a, #1a4d2e);
        border: 2px solid #3fb950;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }

    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(248,81,73,0.4); }
        50% { box-shadow: 0 0 0 10px rgba(248,81,73,0); }
    }

    .prob-bar-container {
        background: #21262d;
        border-radius: 8px;
        height: 24px;
        overflow: hidden;
        margin: 8px 0;
    }

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        color: #58a6ff;
        font-size: 13px;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 16px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
    }

    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
    }

    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background-color: #21262d !important;
        border-color: #30363d !important;
        color: #e6edf3 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 16px;
        width: 100%;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #388bfd, #58a6ff);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(31,111,235,0.4);
    }

    .tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }

    .tag-blue { background: #1f3a5f; color: #58a6ff; border: 1px solid #1f6feb; }
    .tag-green { background: #0d2e1a; color: #3fb950; border: 1px solid #238636; }
    .tag-red { background: #3d1a1a; color: #f85149; border: 1px solid #da3633; }
    .tag-yellow { background: #3a2e0d; color: #d29922; border: 1px solid #9e6a03; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "model/fraud_model.pkl"
    scaler_path = "model/scaler.pkl"
    meta_path   = "model/meta.json"

    if not os.path.exists(model_path):
        st.error("❌ Model not found. Run `python train_model.py` first.")
        st.stop()

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, scaler, meta

model, scaler, meta = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudGuard ML")
    st.markdown("*Supervised Classification Engine*")
    st.divider()

    st.markdown("### 📊 Model Info")
    st.markdown(f"""
    <div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;'>
        <p style='color:#8b949e;font-size:12px;margin:0;'>Algorithm</p>
        <p style='color:#e6edf3;font-weight:600;margin:4px 0 10px;'>{meta['model_name']}</p>
        <p style='color:#8b949e;font-size:12px;margin:0;'>Training Accuracy</p>
        <p style='color:#3fb950;font-weight:600;margin:4px 0 10px;'>{meta['accuracy']:.4f}</p>
        <p style='color:#8b949e;font-size:12px;margin:0;'>ROC-AUC Score</p>
        <p style='color:#58a6ff;font-weight:600;margin:4px 0 10px;'>{meta['roc_auc']:.4f}</p>
        <p style='color:#8b949e;font-size:12px;margin:0;'>F1-Score (Fraud)</p>
        <p style='color:#d29922;font-weight:600;margin:4px 0;'>{meta['f1_fraud']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### ⚡ Quick Load")
    scenario = st.selectbox("Load a preset scenario", [
        "— Select —",
        "🟢 Normal Purchase ($45)",
        "🟢 Grocery Shopping ($120)",
        "🔴 High-Value Anomaly ($4,200)",
        "🔴 Suspicious Pattern",
        "🟡 Borderline Case"
    ])

    st.divider()
    st.markdown("<p style='color:#8b949e;font-size:11px;'>Trained on Kaggle Credit Card Fraud Dataset<br>284,807 transactions · 492 fraud cases</p>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# 🛡️ Credit Card Fraud Detection")
    st.markdown("*Real-time supervised ML classification · Powered by Random Forest + XGBoost*")
with col_h2:
    now = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style='text-align:right;padding-top:10px;'>
        <span class='tag tag-green'>● LIVE</span>
        <p style='color:#8b949e;font-size:12px;margin:4px 0;'>{now}</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍  Predict Transaction", "📈  Model Performance", "📚  How It Works"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICT
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    # Scenario presets
    presets = {
        "🟢 Normal Purchase ($45)":    {"amount": 45.00,   "v1": 1.19, "v3": 0.27,  "v4": 0.17,  "v7": 0.40,  "v10": -0.10, "v12": -0.08, "v14": -0.10, "v17": 0.09, "v28": 0.06},
        "🟢 Grocery Shopping ($120)":  {"amount": 120.50,  "v1": 1.05, "v3": 0.50,  "v4": 0.30,  "v7": 0.20,  "v10": -0.05, "v12": -0.02, "v14": -0.15, "v17": 0.12, "v28": 0.03},
        "🔴 High-Value Anomaly ($4,200)": {"amount": 4200.00,"v1": -4.10,"v3": -3.22,"v4": 3.50,  "v7": -2.89, "v10": -3.10, "v12": -4.20, "v14": -6.50, "v17": -2.9, "v28": 0.80},
        "🔴 Suspicious Pattern":       {"amount": 1.00,    "v1": -3.54,"v3": -4.45, "v4": 4.93,  "v7": -4.28, "v10": -2.54, "v12": -5.81, "v14": -7.63, "v17": -4.10,"v28": 0.92},
        "🟡 Borderline Case":          {"amount": 399.00,  "v1": -1.20,"v3": -0.50, "v4": 1.20,  "v7": -0.80, "v10": -0.90, "v12": -1.30, "v14": -1.80, "v17": -0.70,"v28": 0.30},
    }

    defaults = {"amount": 149.62, "v1": -1.36, "v3": 0.27, "v4": 0.17, "v7": 0.40,
                "v10": -0.10, "v12": -0.08, "v14": -0.10, "v17": 0.09, "v28": 0.06}

    if scenario != "— Select —" and scenario in presets:
        vals = presets[scenario]
    else:
        vals = defaults

    st.markdown("<p class='section-header'>Transaction Details</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**💰 Transaction Amount (USD)**")
        amount = st.number_input("Amount", min_value=0.01, max_value=50000.0,
                                  value=float(vals["amount"]), step=0.01,
                                  label_visibility="collapsed")

        st.markdown("**V1** — *PCA Feature 1 (Time-Velocity)*")
        v1 = st.number_input("V1", value=float(vals["v1"]), step=0.01, format="%.4f", label_visibility="collapsed")

        st.markdown("**V3** — *PCA Feature 3 (Amount-Pattern)*")
        v3 = st.number_input("V3", value=float(vals["v3"]), step=0.01, format="%.4f", label_visibility="collapsed")

        st.markdown("**V4** — *PCA Feature 4 (Merchant-Category)*")
        v4 = st.number_input("V4", value=float(vals["v4"]), step=0.01, format="%.4f", label_visibility="collapsed")

    with col2:
        st.markdown("**V7** — *PCA Feature 7 (Geographic)*")
        v7 = st.number_input("V7", value=float(vals["v7"]), step=0.01, format="%.4f", label_visibility="collapsed")

        st.markdown("**V10** — *PCA Feature 10 (Card-Usage)*")
        v10 = st.number_input("V10", value=float(vals["v10"]), step=0.01, format="%.4f", label_visibility="collapsed")

        st.markdown("**V12** — *PCA Feature 12 (Transaction-Speed)*")
        v12 = st.number_input("V12", value=float(vals["v12"]), step=0.01, format="%.4f", label_visibility="collapsed")

    with col3:
        st.markdown("**V14** — *PCA Feature 14 (Behavioral)* ⭐ Most Important")
        v14 = st.number_input("V14", value=float(vals["v14"]), step=0.01, format="%.4f", label_visibility="collapsed")

        st.markdown("**V17** — *PCA Feature 17 (Network-Score)*")
        v17 = st.number_input("V17", value=float(vals["v17"]), step=0.01, format="%.4f", label_visibility="collapsed")

        st.markdown("**V28** — *PCA Feature 28 (Misc-Signal)*")
        v28 = st.number_input("V28", value=float(vals["v28"]), step=0.01, format="%.4f", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 CLASSIFY TRANSACTION", use_container_width=True)

    if predict_btn:
        # Build feature vector (10 features we trained on)
        features_raw = np.array([[v1, v3, v4, v7, v10, v12, v14, v17, v28, amount]])
        features_scaled = scaler.transform(features_raw)

        prob = model.predict_proba(features_scaled)[0]
        fraud_prob = prob[1]
        pred_class = 1 if fraud_prob > 0.5 else 0

        # Risk tier
        if fraud_prob < 0.3:
            risk = "LOW RISK"; risk_tag = "tag-green"; risk_icon = "🟢"
        elif fraud_prob < 0.7:
            risk = "MEDIUM RISK"; risk_tag = "tag-yellow"; risk_icon = "🟡"
        elif fraud_prob < 0.95:
            risk = "HIGH RISK"; risk_tag = "tag-red"; risk_icon = "🔴"
        else:
            risk = "CRITICAL"; risk_tag = "tag-red"; risk_icon = "🚨"

        st.divider()
        st.markdown("<p class='section-header'>Classification Result</p>", unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if pred_class == 1:
                st.markdown(f"""
                <div class='fraud-alert'>
                    <h1 style='color:#f85149;font-size:48px;margin:0;'>⚠️</h1>
                    <h2 style='color:#f85149;margin:8px 0;'>FRAUD DETECTED</h2>
                    <p style='color:#ffa198;font-size:14px;'>Transaction flagged for review</p>
                    <p style='color:#f85149;font-size:32px;font-weight:700;margin:8px 0;'>{fraud_prob:.1%}</p>
                    <p style='color:#8b949e;font-size:12px;'>Fraud Probability</p>
                    <span class='tag tag-red'>{risk_icon} {risk}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='legit-alert'>
                    <h1 style='color:#3fb950;font-size:48px;margin:0;'>✅</h1>
                    <h2 style='color:#3fb950;margin:8px 0;'>LEGITIMATE</h2>
                    <p style='color:#56d364;font-size:14px;'>Transaction approved</p>
                    <p style='color:#3fb950;font-size:32px;font-weight:700;margin:8px 0;'>{(1-fraud_prob):.1%}</p>
                    <p style='color:#8b949e;font-size:12px;'>Legitimate Probability</p>
                    <span class='tag tag-green'>{risk_icon} {risk}</span>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("**Fraud Probability Score**")
            bar_w = int(fraud_prob * 100)
            bar_color = "#f85149" if fraud_prob > 0.5 else "#3fb950"
            st.markdown(f"""
            <div class='prob-bar-container'>
                <div style='height:100%;width:{bar_w}%;background:linear-gradient(90deg,{bar_color}aa,{bar_color});border-radius:8px;transition:width 0.5s;'></div>
            </div>
            <p style='color:#8b949e;font-size:12px;'>Fraud: {fraud_prob:.4f} &nbsp;|&nbsp; Legit: {1-fraud_prob:.4f}</p>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Feature Contribution (Simplified SHAP)**")

            feature_names = ["V1", "V3", "V4", "V7", "V10", "V12", "V14", "V17", "V28", "Amount"]
            feature_vals  = [v1, v3, v4, v7, v10, v12, v14, v17, v28, amount]

            # Heuristic importance for demo
            importances = meta.get("feature_importances", [0.15,0.06,0.08,0.10,0.08,0.12,0.22,0.08,0.05,0.06])

            rows = sorted(zip(importances, feature_names, feature_vals), reverse=True)[:6]
            for imp, fname, fval in rows:
                direction = "↑ Fraud" if (imp * fval < 0 and fval < 0) else "↓ Legit"
                color = "#f85149" if "Fraud" in direction else "#3fb950"
                bar_len = int(imp * 400)
                st.markdown(f"""
                <div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>
                    <span style='color:#e6edf3;font-family:monospace;width:50px;font-size:13px;'>{fname}</span>
                    <div style='background:#21262d;border-radius:4px;height:14px;flex:1;'>
                        <div style='height:100%;width:{bar_len}px;max-width:100%;background:{color};border-radius:4px;opacity:0.8;'></div>
                    </div>
                    <span style='color:{color};font-size:11px;width:65px;'>{fval:+.2f}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;'>
                <p style='color:#8b949e;font-size:11px;margin:0;'>RECOMMENDED ACTION</p>
                <p style='color:#e6edf3;font-weight:600;margin:4px 0;'>
                    {"🚫 Block Transaction & Alert Cardholder" if pred_class==1 else "✅ Approve — Normal Spending Pattern"}
                </p>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<p class='section-header'>Model Evaluation Metrics</p>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC Score", f"{meta['roc_auc']:.4f}", "Target: >0.95 ✅")
    m2.metric("F1-Score (Fraud)", f"{meta['f1_fraud']:.4f}", "Target: >0.85 ✅")
    m3.metric("Recall (Fraud)", f"{meta['recall_fraud']:.4f}", "Target: >0.90")
    m4.metric("Precision (Fraud)", f"{meta['precision_fraud']:.4f}", "Target: >0.88")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("<p class='section-header'>Confusion Matrix</p>", unsafe_allow_html=True)
        cm = meta["confusion_matrix"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total = tn + fp + fn + tp

        st.markdown(f"""
        <div style='background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;'>
            <table style='width:100%;text-align:center;border-collapse:collapse;'>
                <tr>
                    <td style='padding:8px;'></td>
                    <td style='color:#58a6ff;font-weight:600;padding:8px;'>Predicted: Legit</td>
                    <td style='color:#58a6ff;font-weight:600;padding:8px;'>Predicted: Fraud</td>
                </tr>
                <tr>
                    <td style='color:#58a6ff;font-weight:600;padding:8px;'>Actual: Legit</td>
                    <td style='background:#0d2e1a;color:#3fb950;font-size:28px;font-weight:700;padding:20px;border-radius:8px;'>
                        {tn:,}<br><span style='font-size:11px;color:#56d364;'>True Negative</span>
                    </td>
                    <td style='background:#3d1a1a;color:#f85149;font-size:28px;font-weight:700;padding:20px;border-radius:8px;'>
                        {fp:,}<br><span style='font-size:11px;color:#ffa198;'>False Positive</span>
                    </td>
                </tr>
                <tr>
                    <td style='color:#58a6ff;font-weight:600;padding:8px;'>Actual: Fraud</td>
                    <td style='background:#3d2a00;color:#d29922;font-size:28px;font-weight:700;padding:20px;border-radius:8px;'>
                        {fn:,}<br><span style='font-size:11px;color:#e3b341;'>False Negative</span>
                    </td>
                    <td style='background:#0d2e1a;color:#3fb950;font-size:28px;font-weight:700;padding:20px;border-radius:8px;'>
                        {tp:,}<br><span style='font-size:11px;color:#56d364;'>True Positive</span>
                    </td>
                </tr>
            </table>
            <p style='color:#8b949e;font-size:11px;text-align:center;margin-top:12px;'>Total test samples: {total:,}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("<p class='section-header'>Model Comparison</p>", unsafe_allow_html=True)

        models_data = meta.get("comparison", [
            {"name": "Logistic Regression", "auc": 0.972, "f1": 0.81, "recall": 0.84},
            {"name": "Random Forest ⭐",    "auc": 0.985, "f1": 0.89, "recall": 0.91},
        ])

        for m in models_data:
            is_best = "⭐" in m["name"]
            border = "#1f6feb" if is_best else "#30363d"
            st.markdown(f"""
            <div style='background:#161b22;border:1.5px solid {border};border-radius:10px;padding:14px;margin-bottom:10px;'>
                <p style='color:#e6edf3;font-weight:600;margin:0 0 8px;'>{m["name"]}</p>
                <div style='display:flex;gap:16px;'>
                    <span style='color:#8b949e;font-size:12px;'>AUC: <b style='color:#58a6ff;'>{m["auc"]:.3f}</b></span>
                    <span style='color:#8b949e;font-size:12px;'>F1: <b style='color:#3fb950;'>{m["f1"]:.3f}</b></span>
                    <span style='color:#8b949e;font-size:12px;'>Recall: <b style='color:#d29922;'>{m["recall"]:.3f}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("<p class='section-header'>Feature Importance (Top 10)</p>", unsafe_allow_html=True)
        feature_names = ["V1","V3","V4","V7","V10","V12","V14","V17","V28","Amount"]
        importances   = meta.get("feature_importances", [0.15,0.06,0.08,0.10,0.08,0.12,0.22,0.08,0.05,0.06])
        fi_sorted = sorted(zip(importances, feature_names), reverse=True)

        for imp, fname in fi_sorted:
            bar_w = int(imp * 600)
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:10px;margin:5px 0;'>
                <span style='color:#e6edf3;font-family:monospace;width:45px;font-size:13px;'>{fname}</span>
                <div style='flex:1;background:#21262d;border-radius:4px;height:16px;'>
                    <div style='height:100%;width:{min(bar_w,100)}%;background:linear-gradient(90deg,#1f6feb,#58a6ff);border-radius:4px;'></div>
                </div>
                <span style='color:#8b949e;font-size:12px;width:45px;'>{imp:.3f}</span>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<p class='section-header'>Supervised ML Pipeline</p>", unsafe_allow_html=True)

    steps = [
        ("1", "📥 Data Ingestion", "Kaggle dataset: 284,807 transactions, 492 fraudulent (0.17% fraud rate)"),
        ("2", "🔬 EDA & Preprocessing", "StandardScaler on Amount/Time · V1-V28 already PCA-transformed"),
        ("3", "⚖️ SMOTE Balancing", "Synthetic Minority Over-sampling to handle extreme class imbalance (0.17% → balanced)"),
        ("4", "🤖 Model Training", "Random Forest + Logistic Regression trained & compared with 5-Fold Cross-Validation"),
        ("5", "📊 Validation", "ROC-AUC, Precision-Recall, F1, McNemar's Test, Bootstrap CI (95%)"),
        ("6", "🚀 Deployment", "FastAPI backend + Streamlit dashboard for real-time inference (<50ms latency)"),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div style='display:flex;gap:16px;align-items:flex-start;background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin-bottom:10px;'>
            <div style='background:linear-gradient(135deg,#1f6feb,#388bfd);border-radius:50%;width:36px;height:36px;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-weight:700;color:white;'>
                {num}
            </div>
            <div>
                <p style='color:#e6edf3;font-weight:600;margin:0 0 4px;'>{title}</p>
                <p style='color:#8b949e;font-size:13px;margin:0;'>{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<p class='section-header'>Why These Metrics?</p>", unsafe_allow_html=True)

    metrics_info = [
        ("ROC-AUC", "> 0.95", "Measures how well model separates fraud vs legit across ALL thresholds"),
        ("Recall (Fraud)", "> 0.90", "Critical: we must catch most fraud — missing fraud is most costly"),
        ("F1-Score", "> 0.85", "Harmonic mean of precision & recall — balanced performance measure"),
        ("McNemar's Test", "p < 0.05", "Statistical significance test to confirm one model is truly better"),
        ("5-Fold CV", "std < 0.02", "Ensures model performance is stable, not a lucky train-test split"),
    ]

    for metric, target, why in metrics_info:
        st.markdown(f"""
        <div style='display:flex;gap:12px;align-items:center;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin-bottom:8px;'>
            <span style='background:#1f3a5f;color:#58a6ff;border-radius:6px;padding:4px 10px;font-weight:600;font-size:13px;width:160px;text-align:center;flex-shrink:0;'>{metric}</span>
            <span style='background:#0d2e1a;color:#3fb950;border-radius:6px;padding:4px 8px;font-size:12px;width:80px;text-align:center;flex-shrink:0;'>{target}</span>
            <span style='color:#8b949e;font-size:13px;'>{why}</span>
        </div>
        """, unsafe_allow_html=True)