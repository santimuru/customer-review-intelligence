"""
app.py — Streamlit app: Customer Review Intelligence
NLP & LLM Churn Detection — Portfolio Project
"""

import os
import sys
import warnings
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # carga .env desde raíz del proyecto
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import re

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
sys.path.insert(0, os.path.join(ROOT, "src"))

# ── Colour palette ─────────────────────────────────────────────────────────
C_PRIMARY  = "#635BFF"
C_DANGER   = "#FF4B4B"
C_SAFE     = "#21C55D"
C_WARNING  = "#F59E0B"
C_INFO     = "#06B6D4"
COLORS     = [C_PRIMARY, C_DANGER, C_SAFE, C_WARNING, C_INFO]

PLOTLY_LAYOUT = dict(
    plot_bgcolor  = "white",
    paper_bgcolor = "white",
    font          = dict(color="#1E293B"),
)

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title = "Review Intelligence",
    page_icon  = "📝",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
    /* smooth tab underlines */
    .stTabs [data-baseweb="tab"] { font-size: 0.92rem; }
    /* reduce sidebar padding */
    [data-testid="stSidebar"] { padding-top: 1rem; }
    /* metric label */
    [data-testid="stMetricLabel"] p { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    meta_path = os.path.join(MODELS_DIR, "meta.pkl")
    if not os.path.exists(meta_path):
        return None, None, None, None

    meta          = joblib.load(meta_path)
    tfidf_model   = joblib.load(os.path.join(MODELS_DIR, "tfidf_model.pkl"))
    feat_model    = joblib.load(os.path.join(MODELS_DIR, "features_model.pkl"))
    emb_model     = joblib.load(os.path.join(MODELS_DIR, "embeddings_model.pkl"))
    return meta, tfidf_model, feat_model, emb_model


@st.cache_resource(show_spinner="Loading sentence encoder…")
def load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def get_vader_sia():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


# ══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTOR (mirrors train.py)
# ══════════════════════════════════════════════════════════════════════════

def extract_features_single(text):
    sia    = get_vader_sia()
    scores = sia.polarity_scores(text)
    words  = text.split()
    return {
        "vader_neg":         scores["neg"],
        "vader_neu":         scores["neu"],
        "vader_pos":         scores["pos"],
        "vader_compound":    scores["compound"],
        "text_length":       len(text),
        "word_count":        len(words),
        "avg_word_length":   float(np.mean([len(w) for w in words])) if words else 0.0,
        "exclamation_count": text.count("!"),
        "question_count":    text.count("?"),
        "caps_ratio":        sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "has_never":         int("never"      in text.lower()),
        "has_worst":         int("worst"      in text.lower()),
        "has_terrible":      int("terrible"   in text.lower()),
        "has_amazing":       int("amazing"    in text.lower()),
        "has_excellent":     int("excellent"  in text.lower()),
        "has_love":          int("love"       in text.lower()),
        "has_waste":         int("waste"      in text.lower()),
        "has_horrible":      int("horrible"   in text.lower()),
        "has_recommend":     int("recommend"  in text.lower()),
        "has_disappoint":    int("disappoint" in text.lower()),
        "sentence_count":    len(re.split(r'[.!?]+', text)),
        "ellipsis_count":    text.count("..."),
    }


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

def render_sidebar(meta):
    with st.sidebar:
        st.markdown("""
        <h2 style="margin-bottom:0; color:inherit;">📝 Review Intelligence</h2>
        <p style="color:inherit; opacity:0.7; margin-top:2px; font-size:0.85rem;">
        NLP & LLM Churn Detection
        </p>
        """, unsafe_allow_html=True)

        st.divider()

        page = st.radio(
            "Navigation",
            ["Overview", "Model Comparison", "Text Analyzer", "LLM Features", "Business Insights"],
            label_visibility="collapsed",
        )

        st.divider()

        if meta:
            st.markdown("**Dataset**")
            st.caption(f"Train: {meta['n_train']:,} reviews")
            st.caption(f"Test:  {meta['n_test']:,} reviews")
            st.caption(f"Churn rate: {meta['churn_rate']:.1%}")

        st.divider()
        st.caption("Portfolio project — Yelp Review Intelligence")

    return page


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════

def section_overview(meta):
    st.markdown("""
    <h1 style="color:inherit;">Can text alone predict churn risk?</h1>
    <p style="color:inherit; font-size:1.1rem; opacity:0.8;">
    4 progressive NLP methods — from TF-IDF to Zero-Shot LLMs — on 650K Yelp reviews
    </p>
    """, unsafe_allow_html=True)

    best_auc = max(m["roc_auc"] for m in meta["models"].values())
    best_name = max(meta["models"], key=lambda k: meta["models"][k]["roc_auc"])

    # ── KPIs ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Samples", f"{meta['n_train']:,}")
    c2.metric("Test Samples",     f"{meta['n_test']:,}")
    c3.metric("Churn Rate",       f"{meta['churn_rate']:.1%}")
    c4.metric("Best AUC-ROC",     f"{best_auc:.4f}", help=f"Model: {best_name}")

    st.divider()

    # ── Problem + Results ─────────────────────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("""
        <div style="background:rgba(99,91,255,0.08); border-left:4px solid #635BFF;
                    padding:1.2rem 1.4rem; border-radius:0 8px 8px 0;">
        <h3 style="color:inherit; margin-top:0;">The Problem</h3>
        <p style="color:inherit;">
        By the time a customer leaves a 1-star review, they've <strong>already churned</strong>.
        Traditional star-rating prediction is reactive — it describes what happened, not what
        <em>will</em> happen.
        </p>
        <p style="color:inherit;">
        The real question is: <strong>can we detect churn signals in language itself</strong> —
        the choice of words, tone, emotion — before an explicit rating is given?
        </p>
        <p style="color:inherit; margin-bottom:0;">
        This project tests four progressively sophisticated NLP approaches on Yelp reviews,
        treating 1–2 star reviews as churn signals and 4–5 star reviews as retention signals.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("**Key Results**")
        rows = []
        for name, m in meta["models"].items():
            star = " ★" if name == best_name else ""
            rows.append({
                "Model": name + star,
                "AUC-ROC": f"{m['roc_auc']:.4f}",
                "F1": f"{m['f1']:.4f}",
                "Time": f"{m['training_time']:.0f}s",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()

    # ── Methodology cards ─────────────────────────────────────────
    st.markdown('<h3 style="color:inherit;">Methodology Progression</h3>', unsafe_allow_html=True)

    cards = [
        ("🔢 TF-IDF Baseline",
         "Count-based word frequencies + Logistic Regression. Fast, interpretable, surprisingly strong.",
         C_PRIMARY),
        ("⚙️ Feature Engineering",
         "22 hand-crafted signals: VADER sentiment, keyword flags, stylistic patterns + XGBoost.",
         C_WARNING),
        ("🧠 Semantic Embeddings",
         "all-MiniLM-L6-v2 turns each review into a 384-dim dense vector capturing meaning, not just words.",
         C_INFO),
        ("🤖 Zero-Shot LLM",
         "No training at all — raw language model reasoning. Shows what off-the-shelf LLMs can do.",
         C_SAFE),
    ]

    cols = st.columns(4)
    for col, (title, desc, color) in zip(cols, cards):
        col.markdown(f"""
        <div style="background:rgba(0,0,0,0.04); border-top:4px solid {color};
                    padding:1rem; border-radius:0 0 8px 8px; height:160px;">
        <h4 style="color:inherit; margin:0 0 0.5rem 0; font-size:0.95rem;">{title}</h4>
        <p style="color:inherit; font-size:0.85rem; opacity:0.85; margin:0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Key findings ──────────────────────────────────────────────
    st.markdown('<h3 style="color:inherit;">Key Findings</h3>', unsafe_allow_html=True)

    auc_vals = {k: v["roc_auc"] for k, v in meta["models"].items()}
    delta_emb_vs_tfidf = auc_vals["Sentence Embeddings + XGBoost"] - auc_vals["TF-IDF + LR"]
    delta_zs_vs_feats  = auc_vals["Zero-Shot LLM"] - auc_vals["Engineered Features + XGBoost"]

    findings = [
        f"**Best model ({best_name})** achieves AUC-ROC of **{best_auc:.4f}** — strong predictive signal from text alone.",
        f"Sentence embeddings {'beat' if delta_emb_vs_tfidf > 0 else 'trail'} TF-IDF baseline by **{abs(delta_emb_vs_tfidf):.4f} AUC** — semantic meaning adds measurable value.",
        f"Zero-shot LLM {'outperforms' if delta_zs_vs_feats > 0 else 'underperforms vs.'} engineered features by **{abs(delta_zs_vs_feats):.4f} AUC** with zero training data.",
        "VADER compound score is the single strongest engineered feature (per SHAP), confirming sentiment polarity is the dominant signal.",
        f"Churn language is specific: words like *{', '.join(meta['top_churn_words'][:4])}* are highly predictive.",
    ]

    for f in findings:
        st.markdown(f"- {f}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════

def section_model_comparison(meta):
    st.markdown('<h2 style="color:inherit;">Model Comparison</h2>', unsafe_allow_html=True)

    model_names = list(meta["models"].keys())
    best_name   = max(meta["models"], key=lambda k: meta["models"][k]["roc_auc"])

    tabs = st.tabs(["📈 ROC Curves", "📉 Precision-Recall", "📊 Metrics Table", "🔲 Confusion Matrices"])

    # ── ROC ────────────────────────────────────────────────────────
    with tabs[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", line=dict(dash="dash", color="#CBD5E1", width=1),
            name="Random", showlegend=True,
        ))
        for i, (name, m) in enumerate(meta["models"].items()):
            roc = m["roc_curve"]
            fig.add_trace(go.Scatter(
                x=roc["fpr"], y=roc["tpr"],
                mode="lines",
                name=f"{name} (AUC={m['roc_auc']:.4f})",
                line=dict(
                    color=COLORS[i % len(COLORS)],
                    width=4 if name == best_name else 1.8,
                ),
            ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="ROC Curve — All Models",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.6, y=0.1),
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── PR ─────────────────────────────────────────────────────────
    with tabs[1]:
        fig = go.Figure()
        churn_rate = meta["churn_rate"]
        fig.add_hline(
            y=churn_rate, line_dash="dash", line_color="#CBD5E1",
            annotation_text=f"Baseline ({churn_rate:.2f})",
        )
        for i, (name, m) in enumerate(meta["models"].items()):
            pr = m["pr_curve"]
            fig.add_trace(go.Scatter(
                x=pr["recall"], y=pr["precision"],
                mode="lines",
                name=f"{name} (AP={m['avg_precision']:.4f})",
                line=dict(
                    color=COLORS[i % len(COLORS)],
                    width=4 if name == best_name else 1.8,
                ),
            ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Precision-Recall Curve — All Models",
            xaxis_title="Recall",
            yaxis_title="Precision",
            legend=dict(x=0.0, y=0.1),
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Metrics table ──────────────────────────────────────────────
    with tabs[2]:
        rows = []
        for name, m in meta["models"].items():
            rows.append({
                "Model":         ("★ " if name == best_name else "  ") + name,
                "Accuracy":      round(m["accuracy"],      4),
                "Precision":     round(m["precision"],     4),
                "Recall":        round(m["recall"],        4),
                "F1":            round(m["f1"],            4),
                "AUC-ROC":       round(m["roc_auc"],       4),
                "Avg Precision": round(m["avg_precision"], 4),
                "Train Time (s)": m["training_time"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.markdown("**AUC-ROC Comparison**")
        fig2 = go.Figure(go.Bar(
            x=[r["AUC-ROC"] for r in rows],
            y=[r["Model"]   for r in rows],
            orientation="h",
            marker_color=[
                C_DANGER if best_name in r["Model"] else C_PRIMARY
                for r in rows
            ],
            text=[str(r["AUC-ROC"]) for r in rows],
            textposition="outside",
        ))
        fig2.update_layout(
            **PLOTLY_LAYOUT,
            xaxis=dict(range=[0.5, 1.05]),
            height=280,
            margin=dict(l=0, r=0, t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Confusion Matrices ─────────────────────────────────────────
    with tabs[3]:
        n_models = len(model_names)
        fig = make_subplots(
            rows=1, cols=n_models,
            subplot_titles=model_names,
        )
        for i, (name, m) in enumerate(meta["models"].items()):
            cm = np.array(m["confusion_matrix"])
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=["Pred Retain", "Pred Churn"],
                    y=["Actual Retain", "Actual Churn"],
                    colorscale=[[0, "white"], [1, C_PRIMARY]],
                    showscale=(i == 0),
                    text=cm,
                    texttemplate="%{text}",
                    textfont=dict(size=14, color="#1E293B"),
                ),
                row=1, col=i + 1,
            )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            margin=dict(t=50),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: TEXT ANALYZER
# ══════════════════════════════════════════════════════════════════════════

def section_text_analyzer(meta, tfidf_model, feat_model, emb_model):
    st.markdown('<h2 style="color:inherit;">⭐ Text Analyzer</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:inherit; opacity:0.8;">Score any review in real-time across 3 models '
        '(Zero-Shot excluded — too slow for live inference).</p>',
        unsafe_allow_html=True,
    )

    # ── Example selector ──────────────────────────────────────────
    st.markdown("**Quick-load an example:**")
    example_type = st.selectbox(
        "Example type",
        ["— choose —", "Churn signal example", "Retention signal example"],
        label_visibility="collapsed",
    )

    example_text = ""
    if example_type == "Churn signal example" and meta["sample_reviews"]["churn"]:
        example_text = meta["sample_reviews"]["churn"][0]
    elif example_type == "Retention signal example" and meta["sample_reviews"]["retain"]:
        example_text = meta["sample_reviews"]["retain"][0]

    text_input = st.text_area(
        "Enter a customer review:",
        value=example_text,
        height=130,
        placeholder="Type or paste any review here…",
    )

    run = st.button("Analyze", type="primary")

    if run and text_input.strip():
        with st.spinner("Scoring…"):
            _run_analysis(text_input.strip(), meta, tfidf_model, feat_model, emb_model)
    elif run:
        st.warning("Please enter a review first.")


def _run_analysis(text, meta, tfidf_model, feat_model, emb_model):
    import xgboost as xgb

    # ── Scores ────────────────────────────────────────────────────
    prob_tfidf = float(tfidf_model.predict_proba([text])[0, 1])

    feats_dict = extract_features_single(text)
    feats_df   = pd.DataFrame([feats_dict])
    prob_feat  = float(feat_model.predict_proba(feats_df.values)[0, 1])

    encoder    = load_encoder()
    emb        = encoder.encode([text], batch_size=1, show_progress_bar=False, convert_to_numpy=True)
    prob_emb   = float(emb_model.predict_proba(emb)[0, 1])

    # ── 3-column model scores ─────────────────────────────────────
    model_results = [
        ("TF-IDF + LR",          prob_tfidf, C_PRIMARY),
        ("Engineered Features",  prob_feat,  C_WARNING),
        ("Sentence Embeddings",  prob_emb,   C_INFO),
    ]

    st.markdown("---")
    st.markdown('<h4 style="color:inherit;">Model Scores</h4>', unsafe_allow_html=True)

    cols = st.columns(3)
    for col, (mname, prob, color) in zip(cols, model_results):
        risk_label = "HIGH RISK" if prob >= 0.6 else ("MEDIUM" if prob >= 0.4 else "LOW RISK")
        risk_color = C_DANGER if prob >= 0.6 else (C_WARNING if prob >= 0.4 else C_SAFE)
        col.markdown(f"""
        <div style="background:rgba(0,0,0,0.04); border-radius:10px; padding:1rem; text-align:center; border:1px solid rgba(0,0,0,0.08);">
          <div style="color:inherit; font-size:0.85rem; opacity:0.7; margin-bottom:0.3rem;">{mname}</div>
          <div style="font-size:2.2rem; font-weight:700; color:{color};">{prob:.1%}</div>
          <div style="display:inline-block; background:{risk_color}22; color:{risk_color};
                      border:1px solid {risk_color}55; border-radius:20px;
                      padding:2px 12px; font-size:0.8rem; font-weight:600; margin-top:0.3rem;">
            {risk_label}
          </div>
          <div style="color:inherit; opacity:0.6; font-size:0.78rem; margin-top:0.5rem;">churn probability</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── SHAP for Model 2 ──────────────────────────────────────────
    st.markdown('<h4 style="color:inherit;">Feature Explanation (Model 2 — Engineered Features)</h4>',
                unsafe_allow_html=True)

    try:
        booster = feat_model.get_booster()
        feat_names = list(feats_dict.keys())
        dmat = xgb.DMatrix(feats_df.values, feature_names=feat_names)
        contribs  = booster.predict(dmat, pred_contribs=True)
        shap_vals = contribs[0, :-1]   # single sample, drop bias

        shap_df = pd.DataFrame({"feature": feat_names, "shap": shap_vals})
        shap_df = shap_df.reindex(shap_df["shap"].abs().sort_values(ascending=True).index)

        colors_bar = [C_DANGER if v > 0 else C_SAFE for v in shap_df["shap"]]

        fig = go.Figure(go.Bar(
            x=shap_df["shap"],
            y=shap_df["feature"],
            orientation="h",
            marker_color=colors_bar,
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="SHAP values — positive = pushes toward churn",
            height=420,
            xaxis_title="SHAP value",
            margin=dict(l=0, r=0, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"SHAP explanation unavailable: {e}")

    # ── VADER breakdown ───────────────────────────────────────────
    st.markdown('<h4 style="color:inherit;">VADER Sentiment Breakdown</h4>', unsafe_allow_html=True)

    vader_data = {
        "neg": feats_dict["vader_neg"],
        "neu": feats_dict["vader_neu"],
        "pos": feats_dict["vader_pos"],
        "compound": (feats_dict["vader_compound"] + 1) / 2,   # normalize to [0,1] for display
    }
    vader_colors = [C_DANGER, "#94A3B8", C_SAFE, C_PRIMARY]
    fig_v = go.Figure(go.Bar(
        x=list(vader_data.values()),
        y=[k.upper() for k in vader_data],
        orientation="h",
        marker_color=vader_colors,
        text=[f"{v:.3f}" for v in vader_data.values()],
        textposition="outside",
    ))
    fig_v.update_layout(
        **PLOTLY_LAYOUT,
        height=200,
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(range=[0, 1.15]),
    )
    st.plotly_chart(fig_v, use_container_width=True)
    st.caption("Compound normalized to [0,1] for display; raw range is [-1,+1]")

    # ── Keyword signals ───────────────────────────────────────────
    st.markdown('<h4 style="color:inherit;">Keyword Signals Detected</h4>', unsafe_allow_html=True)

    churn_kw   = ["never", "worst", "terrible", "waste", "horrible", "disappoint"]
    retain_kw  = ["amazing", "excellent", "love", "recommend"]
    found_churn  = [w for w in churn_kw  if w in text.lower()]
    found_retain = [w for w in retain_kw if w in text.lower()]

    kc1, kc2 = st.columns(2)
    with kc1:
        if found_churn:
            badges = "".join(
                f'<span style="background:rgba(255,75,75,0.15); color:{C_DANGER}; '
                f'border:1px solid {C_DANGER}55; border-radius:12px; '
                f'padding:2px 10px; margin:3px; display:inline-block; font-size:0.85rem;">'
                f'{w}</span>'
                for w in found_churn
            )
            st.markdown(f'<p style="color:inherit;">🔴 Churn signals: {badges}</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:inherit; opacity:0.6;">🔴 No churn keywords found</p>',
                        unsafe_allow_html=True)
    with kc2:
        if found_retain:
            badges = "".join(
                f'<span style="background:rgba(33,197,93,0.15); color:{C_SAFE}; '
                f'border:1px solid {C_SAFE}55; border-radius:12px; '
                f'padding:2px 10px; margin:3px; display:inline-block; font-size:0.85rem;">'
                f'{w}</span>'
                for w in found_retain
            )
            st.markdown(f'<p style="color:inherit;">🟢 Retention signals: {badges}</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:inherit; opacity:0.6;">🟢 No retention keywords found</p>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: LLM FEATURES (GEMINI)
# ══════════════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """Analyze this customer review and extract structured information.
Return ONLY a valid JSON object with these fields:
{{
  "complaint_category": "price|quality|service|delivery|other|none",
  "churn_intent": "explicit|implicit|none",
  "urgency": "high|medium|low",
  "emotion": "angry|frustrated|disappointed|neutral|happy|delighted",
  "key_phrases": ["phrase1", "phrase2"],
  "positive_aspects": ["aspect1"],
  "churn_risk_score": 0
}}

Review: {text}"""

# Static demo data shown when no API key is provided
DEMO_GEMINI = {
    "review": (
        "I've been a customer for 5 years and this is how they treat loyal customers? "
        "The service was absolutely appalling, no one could answer my questions, "
        "and I waited 45 minutes on hold. Never coming back."
    ),
    "result": {
        "complaint_category": "service",
        "churn_intent": "explicit",
        "urgency": "high",
        "emotion": "angry",
        "key_phrases": ["absolutely appalling", "waited 45 minutes", "no one could answer"],
        "positive_aspects": [],
        "churn_risk_score": 9,
    },
    "ml_score": 0.87,
}

FIELD_ICONS = {
    "complaint_category": "🏷️",
    "churn_intent":       "⚠️",
    "urgency":            "🔥",
    "emotion":            "😤",
    "key_phrases":        "💬",
    "positive_aspects":   "✅",
    "churn_risk_score":   "📊",
}

RISK_COLORS = {
    "explicit": C_DANGER,
    "implicit": C_WARNING,
    "none":     C_SAFE,
}


def render_gemini_result(result: dict, ml_score: float):
    st.markdown('<h4 style="color:inherit;">Extracted Features</h4>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    simple_fields = [
        ("complaint_category", result.get("complaint_category", "—")),
        ("churn_intent",       result.get("churn_intent", "—")),
        ("urgency",            result.get("urgency", "—")),
        ("emotion",            result.get("emotion", "—")),
    ]

    list_fields = [
        ("key_phrases",      result.get("key_phrases",     [])),
        ("positive_aspects", result.get("positive_aspects", [])),
    ]

    with col1:
        for key, val in simple_fields:
            color = RISK_COLORS.get(val, C_PRIMARY)
            st.markdown(f"""
            <div style="background:rgba(0,0,0,0.04); border-radius:8px;
                        padding:0.6rem 1rem; margin-bottom:0.5rem;
                        border-left:3px solid {color};">
              <span style="color:inherit; opacity:0.6; font-size:0.78rem;">
                {FIELD_ICONS.get(key,'')} {key.replace('_',' ').upper()}
              </span><br>
              <strong style="color:inherit;">{val}</strong>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        for key, items in list_fields:
            pills = "".join(
                f'<span style="background:rgba(99,91,255,0.12); color:inherit; '
                f'border-radius:12px; padding:2px 9px; margin:2px; '
                f'display:inline-block; font-size:0.82rem;">{item}</span>'
                for item in (items or ["none"])
            )
            st.markdown(f"""
            <div style="background:rgba(0,0,0,0.04); border-radius:8px;
                        padding:0.6rem 1rem; margin-bottom:0.5rem;">
              <span style="color:inherit; opacity:0.6; font-size:0.78rem;">
                {FIELD_ICONS.get(key,'')} {key.replace('_',' ').upper()}
              </span><br>
              {pills}
            </div>
            """, unsafe_allow_html=True)

    # Score comparison
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<h4 style="color:inherit;">Score Comparison</h4>', unsafe_allow_html=True)

    gemini_score = result.get("churn_risk_score", 5)
    gemini_norm  = gemini_score / 10.0

    sc1, sc2 = st.columns(2)
    with sc1:
        sc_color = C_DANGER if ml_score >= 0.6 else (C_WARNING if ml_score >= 0.4 else C_SAFE)
        st.markdown(f"""
        <div style="text-align:center; background:rgba(0,0,0,0.04);
                    border-radius:10px; padding:1rem;">
          <div style="color:inherit; opacity:0.65; font-size:0.85rem;">ML Model Score</div>
          <div style="font-size:2rem; font-weight:700; color:{sc_color};">{ml_score:.1%}</div>
          <div style="color:inherit; opacity:0.55; font-size:0.78rem;">TF-IDF + LR</div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        g_color = C_DANGER if gemini_norm >= 0.6 else (C_WARNING if gemini_norm >= 0.4 else C_SAFE)
        st.markdown(f"""
        <div style="text-align:center; background:rgba(0,0,0,0.04);
                    border-radius:10px; padding:1rem;">
          <div style="color:inherit; opacity:0.65; font-size:0.85rem;">Gemini Risk Score</div>
          <div style="font-size:2rem; font-weight:700; color:{g_color};">{gemini_score}/10</div>
          <div style="color:inherit; opacity:0.55; font-size:0.78rem;">Zero-shot LLM</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(6,182,212,0.08); border-left:3px solid #06B6D4;
                padding:0.8rem 1rem; border-radius:0 8px 8px 0; margin-top:1rem;">
    <p style="color:inherit; margin:0; font-size:0.88rem;">
    <strong>Why this matters:</strong> The ML model sees statistical patterns in word frequencies and embeddings.
    Gemini understands <em>semantics</em> — intent, complaint category, emotional state — that bag-of-words
    models completely miss. Together they provide complementary signals for production churn systems.
    </p>
    </div>
    """, unsafe_allow_html=True)


def section_llm_features(meta, tfidf_model):
    st.markdown('<h2 style="color:inherit;">LLM Features — Gemini</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:inherit; opacity:0.8;">Structured feature extraction with Gemini 1.5 Flash '
        '— semantic understanding beyond what traditional models capture.</p>',
        unsafe_allow_html=True,
    )

    import os
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIzaSy… (o seteá GEMINI_API_KEY como variable de entorno)",
            help="Get a free key at https://aistudio.google.com",
        )

    if api_key:
        import google.generativeai as genai
        import json

        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")

        review_text = st.text_area(
            "Review to analyze:",
            height=110,
            placeholder="Paste any customer review…",
        )

        if st.button("Extract with Gemini", type="primary"):
            if not review_text.strip():
                st.warning("Please enter a review.")
            else:
                with st.spinner("Calling Gemini…"):
                    try:
                        prompt   = EXTRACTION_PROMPT.format(text=review_text.strip())
                        response = gemini_model.generate_content(prompt)
                        raw      = response.text.strip()

                        # Strip markdown code fences if present
                        raw = re.sub(r"^```(?:json)?\s*", "", raw)
                        raw = re.sub(r"\s*```$", "", raw)

                        result   = json.loads(raw)
                        ml_score = float(tfidf_model.predict_proba([review_text.strip()])[0, 1])

                        render_gemini_result(result, ml_score)
                    except json.JSONDecodeError:
                        st.error("Gemini returned invalid JSON. Try again.")
                        st.code(raw)
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        # Static demo
        st.markdown("""
        <div style="background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.4);
                    border-radius:8px; padding:0.8rem 1rem; margin-bottom:1rem;">
        <p style="color:inherit; margin:0; font-size:0.88rem;">
        <strong>Demo mode</strong> — No API key provided. Showing pre-computed example.
        Enter your free Gemini API key above to try live extraction.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:rgba(0,0,0,0.04); border-radius:8px; padding:1rem; margin-bottom:1rem;">
        <strong style="color:inherit; opacity:0.6; font-size:0.8rem;">EXAMPLE REVIEW</strong><br>
        <p style="color:inherit; font-style:italic; margin:0.4rem 0 0 0;">"{DEMO_GEMINI['review']}"</p>
        </div>
        """, unsafe_allow_html=True)

        render_gemini_result(DEMO_GEMINI["result"], DEMO_GEMINI["ml_score"])


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════

def section_business_insights(meta):
    st.markdown('<h2 style="color:inherit;">Business Insights</h2>', unsafe_allow_html=True)

    tabs = st.tabs(["🔤 Word Analysis", "🧩 What Drives Churn?", "📊 Sentiment Distribution"])

    # ── Word Analysis ─────────────────────────────────────────────
    with tabs[0]:
        st.markdown('<p style="color:inherit; opacity:0.8;">Top 20 most predictive words '
                    'from TF-IDF Logistic Regression coefficients.</p>',
                    unsafe_allow_html=True)

        wc1, wc2 = st.columns(2)

        with wc1:
            churn_w = meta["top_churn_words"]
            fig = go.Figure(go.Bar(
                x=list(range(len(churn_w), 0, -1)),
                y=churn_w,
                orientation="h",
                marker_color=C_DANGER,
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title="Top Churn Signal Words",
                xaxis_title="Rank (relative)",
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)

        with wc2:
            retain_w = meta["top_retain_words"]
            fig2 = go.Figure(go.Bar(
                x=list(range(len(retain_w), 0, -1)),
                y=retain_w,
                orientation="h",
                marker_color=C_SAFE,
            ))
            fig2.update_layout(
                **PLOTLY_LAYOUT,
                title="Top Retention Signal Words",
                xaxis_title="Rank (relative)",
                height=520,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── SHAP ──────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown(
            '<p style="color:inherit; opacity:0.8;">SHAP values for Model 2 '
            '(Engineered Features + XGBoost) on 200 test samples. '
            'Positive SHAP = pushes prediction toward churn.</p>',
            unsafe_allow_html=True,
        )

        shap_vals  = np.array(meta["shap_values"])
        feat_names = meta["shap_feature_names"]
        X_sample   = meta["X_test_features_sample"]

        if isinstance(X_sample, pd.DataFrame):
            X_arr = X_sample.values
        else:
            X_arr = np.array(X_sample)

        # Sort features by mean |SHAP|
        mean_abs = np.abs(shap_vals).mean(axis=0)
        order    = np.argsort(mean_abs)

        sorted_names = [feat_names[i] for i in order]
        sorted_shap  = shap_vals[:, order]
        sorted_X     = X_arr[:, order]

        # Normalise feature values for colouring
        X_norm = (sorted_X - sorted_X.min(axis=0)) / (sorted_X.ptp(axis=0) + 1e-9)

        rng_j = np.random.default_rng(SEED if 'SEED' in dir() else 42)
        jitter = rng_j.uniform(-0.35, 0.35, size=sorted_shap.shape)

        fig = go.Figure()
        for j, fname in enumerate(sorted_names):
            fig.add_trace(go.Scatter(
                x=sorted_shap[:, j],
                y=np.full(sorted_shap.shape[0], j) + jitter[:, j],
                mode="markers",
                name=fname,
                showlegend=False,
                marker=dict(
                    size=4,
                    color=X_norm[:, j],
                    colorscale=[[0, C_SAFE], [0.5, "#F59E0B"], [1, C_DANGER]],
                    opacity=0.6,
                    colorbar=dict(
                        title="Feature<br>value",
                        thickness=10,
                        len=0.7,
                    ) if j == len(sorted_names) - 1 else None,
                ),
            ))

        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="SHAP Beeswarm — Engineered Features",
            xaxis_title="SHAP value (impact on churn probability)",
            yaxis=dict(
                tickvals=list(range(len(sorted_names))),
                ticktext=sorted_names,
            ),
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── VADER Distribution ────────────────────────────────────────
    with tabs[2]:
        st.markdown(
            '<p style="color:inherit; opacity:0.8;">Distribution of VADER compound scores '
            'for churn vs. retained customers (training set). '
            'Where the distributions diverge is the signal the model exploits.</p>',
            unsafe_allow_html=True,
        )

        vader_stats = meta.get("vader_stats", {})
        churn_vals  = vader_stats.get("churn",  [])
        retain_vals = vader_stats.get("retain", [])

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=churn_vals,
            name="Churn (at-risk)",
            marker_color=C_DANGER,
            opacity=0.7,
            nbinsx=50,
            histnorm="probability",
        ))
        fig.add_trace(go.Histogram(
            x=retain_vals,
            name="Retained",
            marker_color=C_SAFE,
            opacity=0.7,
            nbinsx=50,
            histnorm="probability",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            barmode="overlay",
            title="VADER Compound Score Distribution",
            xaxis_title="Compound Score (−1 = most negative, +1 = most positive)",
            yaxis_title="Probability",
            height=420,
            legend=dict(x=0.02, y=0.98),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div style="background:rgba(99,91,255,0.08); border-left:3px solid #635BFF;
                    padding:0.8rem 1rem; border-radius:0 8px 8px 0;">
        <strong style="color:inherit;">Interpretation:</strong>
        <p style="color:inherit; margin:0.4rem 0 0 0; font-size:0.9rem;">
        Churn-risk reviews cluster at strongly negative compound scores (−1 to −0.5),
        while retained customers dominate the positive range (+0.5 to +1).
        The overlap region (−0.3 to +0.3) is where models must rely on
        subtler linguistic signals beyond raw sentiment polarity.
        </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# MISSING MODELS PAGE
# ══════════════════════════════════════════════════════════════════════════

def page_no_models():
    st.error("Model artifacts not found in `models/` directory.")
    st.markdown("""
    **To train all models, run from the project root:**
    ```bash
    python src/train.py
    ```
    This will:
    1. Download the Yelp Review Full dataset (~700 MB, first run only)
    2. Train 4 NLP models (TF-IDF + LR, XGBoost ×2, Zero-Shot)
    3. Save artifacts to `models/`

    Estimated time: 20–60 minutes depending on hardware (GPU optional).
    """)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    meta, tfidf_model, feat_model, emb_model = load_artifacts()

    page = render_sidebar(meta)

    if meta is None:
        page_no_models()
        return

    if page == "Overview":
        section_overview(meta)
    elif page == "Model Comparison":
        section_model_comparison(meta)
    elif page == "Text Analyzer":
        section_text_analyzer(meta, tfidf_model, feat_model, emb_model)
    elif page == "LLM Features":
        section_llm_features(meta, tfidf_model)
    elif page == "Business Insights":
        section_business_insights(meta)


if __name__ == "__main__":
    main()
