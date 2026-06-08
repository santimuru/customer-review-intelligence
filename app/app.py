"""
app.py  Customer Review Intelligence
NLP churn detection from raw review text. Portfolio project.

Design system: Swiss / International Typographic.
Warm paper ground, ink + one saturated red, Archivo grotesque, strict grid,
flat 1px-bordered blocks, zero rounding, no decorative chrome.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import re

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
sys.path.insert(0, os.path.join(ROOT, "src"))

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────
PAPER   = "#F2EFE8"   # warm off-white ground
PAPER_2 = "#E7E2D6"   # slightly deeper panel ground
INK     = "#1A1714"   # warm near-black
INK_60  = "#6B655C"   # muted ink for secondary text
RULE    = "#1A1714"   # hairline / border colour
RED     = "#E5342A"   # the one saturated accent
GREY    = "#9A938633"  # translucent block fill

# Chart palette: ink, red, two warm greys. No rainbow.
SERIES  = [INK, RED, "#8A8478", "#BFB8A8", "#56524B"]

# Axis styling lives in a template so per-chart xaxis=/yaxis= overrides never collide.
PLOTLY_TEMPLATE = go.layout.Template(layout=dict(
    xaxis = dict(gridcolor="#D8D2C4", zerolinecolor="#C2BBAC", linecolor=INK),
    yaxis = dict(gridcolor="#D8D2C4", zerolinecolor="#C2BBAC", linecolor=INK),
))

PLOTLY_LAYOUT = dict(
    plot_bgcolor  = PAPER,
    paper_bgcolor = PAPER,
    font          = dict(color=INK, family="Archivo, sans-serif", size=13),
    colorway      = SERIES,
    template      = PLOTLY_TEMPLATE,
)

NAV = ["Overview", "Models", "Analyze", "Insights"]


def risk_label(prob: float) -> tuple[str, str]:
    """Return (label, colour) for a churn probability in [0, 1]. Ink/red system."""
    if prob >= 0.6:
        return "HIGH RISK", RED
    if prob >= 0.4:
        return "ELEVATED", INK
    return "LOW RISK", INK_60


# ─────────────────────────────────────────────────────────────────────────────
# Page config + global stylesheet
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Review Intelligence",
    page_icon  = "▪",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800;900&family=Archivo+Expanded:wght@600;700;800;900&display=swap');

:root {{
  --ink: {INK}; --red: {RED}; --paper: {PAPER}; --ink60: {INK_60};
}}

html, body, [class*="css"], .stMarkdown, p, li, span, div {{
  font-family: 'Archivo', system-ui, sans-serif;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}

/* kill all rounding, the brutalist baseline */
.stApp *, [data-testid] * {{ border-radius: 0 !important; }}

/* no sidebar in this design */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}

/* content column: editorial width, generous top margin */
.block-container {{ max-width: 1180px; padding-top: 1.2rem; padding-bottom: 4rem; }}

/* Headings: Archivo Expanded, tight, uppercase display */
h1 {{
  font-family: 'Archivo Expanded','Archivo',sans-serif;
  font-weight: 900; letter-spacing: -0.02em; line-height: 0.98;
  font-size: clamp(2.4rem, 5vw, 4.1rem); color: {INK}; margin: 0.2rem 0 0.6rem 0;
}}
h2 {{
  font-family: 'Archivo Expanded','Archivo',sans-serif;
  font-weight: 800; letter-spacing: -0.01em; text-transform: uppercase;
  font-size: 1.7rem; color: {INK}; margin: 0.4rem 0 0.8rem 0;
}}
h3 {{ font-weight: 800; text-transform: uppercase; letter-spacing: 0.01em;
      font-size: 1.05rem; color: {INK}; }}
h4 {{ font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em;
      font-size: 0.82rem; color: {INK}; }}

/* the masthead rule */
.masthead {{
  display:flex; align-items:baseline; justify-content:space-between;
  border-bottom: 3px solid {INK}; padding-bottom: 0.5rem; margin-bottom: 0.4rem;
}}
.masthead .brand {{ font-family:'Archivo Expanded','Archivo',sans-serif;
  font-weight:900; text-transform:uppercase; letter-spacing:0.06em; font-size:0.95rem; }}
.masthead .meta {{ font-size:0.72rem; text-transform:uppercase;
  letter-spacing:0.14em; color:{INK_60}; }}

/* top navigation as numbered Swiss tabs (styled radio) */
div[data-testid="stRadio"] > label {{ display:none; }}
div[data-testid="stRadio"] [role="radiogroup"] {{
  display:flex; gap:0; border-bottom:1px solid {INK}; margin-bottom:1.8rem; flex-wrap:wrap;
}}
div[data-testid="stRadio"] [role="radiogroup"] label {{
  margin:0; padding:0.55rem 1.1rem 0.55rem 0; cursor:pointer;
  border-right:1px solid #CFC8B8;
}}
div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child {{ display:none; }}
div[data-testid="stRadio"] [role="radiogroup"] label > div:last-child p {{
  font-weight:700; text-transform:uppercase; letter-spacing:0.04em;
  font-size:0.78rem; color:{INK_60}; padding:0 1.1rem; margin:0;
}}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) > div:last-child p {{
  color:{INK}; box-shadow: inset 0 -3px 0 0 {RED};
}}

/* flat bordered building blocks */
.blk {{ border:1px solid {INK}; background:{PAPER}; padding:1.1rem 1.25rem; }}
.blk-fill {{ background:{PAPER_2}; }}
.idx {{ font-family:'Archivo Expanded',sans-serif; font-weight:900;
  font-size:0.8rem; color:{RED}; letter-spacing:0.05em; }}
.lede {{ font-size:1.15rem; line-height:1.5; color:{INK}; max-width:64ch;
  font-weight:400; }}
.lede b {{ font-weight:800; }}

/* KPI grid: hard-ruled cells, big ink numerals */
.kpis {{ display:grid; grid-template-columns:repeat(4,1fr); border:1px solid {INK};
  border-right:none; margin:1.4rem 0; }}
.kpi {{ border-right:1px solid {INK}; padding:1rem 1.1rem; }}
.kpi .v {{ font-family:'Archivo Expanded',sans-serif; font-weight:900;
  font-size:2.3rem; line-height:1; color:{INK}; }}
.kpi .v.accent {{ color:{RED}; }}
.kpi .k {{ font-size:0.68rem; text-transform:uppercase; letter-spacing:0.12em;
  color:{INK_60}; margin-top:0.5rem; font-weight:600; }}

/* methodology steps: numbered, no cards */
.steps {{ display:grid; grid-template-columns:repeat(4,1fr); border:1px solid {INK};
  border-right:none; }}
.step {{ border-right:1px solid {INK}; padding:1rem 1.1rem; min-height:170px; }}
.step .n {{ font-family:'Archivo Expanded',sans-serif; font-weight:900; font-size:1.5rem;
  color:{RED}; }}
.step .t {{ font-weight:800; text-transform:uppercase; font-size:0.82rem;
  letter-spacing:0.02em; margin:0.4rem 0; }}
.step .d {{ font-size:0.82rem; color:{INK_60}; line-height:1.45; }}

/* pills / tags */
.tag {{ display:inline-block; border:1px solid {INK}; padding:2px 9px; margin:3px 4px 3px 0;
  font-size:0.78rem; font-weight:600; }}
.tag.red {{ border-color:{RED}; color:{RED}; }}

/* score block */
.score {{ border:1px solid {INK}; padding:1rem; }}
.score .v {{ font-family:'Archivo Expanded',sans-serif; font-weight:900; font-size:2.1rem;
  line-height:1; }}
.score .m {{ font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em;
  color:{INK_60}; font-weight:600; }}

/* buttons: square, ink, red on hover */
.stButton > button {{
  border:1px solid {INK}; background:{INK}; color:{PAPER};
  font-weight:800; text-transform:uppercase; letter-spacing:0.06em; font-size:0.8rem;
  padding:0.5rem 1.3rem;
}}
.stButton > button:hover {{ background:{RED}; border-color:{RED}; color:{PAPER}; }}

/* tabs underline */
.stTabs [data-baseweb="tab-list"] {{ border-bottom:1px solid {INK}; gap:0.4rem; }}
.stTabs [data-baseweb="tab"] {{ font-weight:700; text-transform:uppercase;
  letter-spacing:0.03em; font-size:0.78rem; padding:0.5rem 1.1rem; }}
.stTabs [aria-selected="true"] {{ box-shadow: inset 0 -3px 0 0 {RED}; }}

hr {{ border:none; border-top:1px solid {INK}; margin:1.6rem 0; }}

/* dataframe: square, hairline */
[data-testid="stDataFrame"] {{ border:1px solid {INK}; }}

/* text inputs */
textarea, input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
  border:1px solid {INK} !important; background:{PAPER} !important; color:{INK} !important;
}}
::placeholder {{ color:{INK_60} !important; }}
</style>
""", unsafe_allow_html=True)


def rule(text_left: str, text_right: str = ""):
    """A Swiss section rule: bold left label, hairline, optional right meta."""
    st.markdown(
        f"""<div style="display:flex; align-items:baseline; justify-content:space-between;
        border-top:1px solid {INK}; padding-top:0.5rem; margin:2rem 0 1rem 0;">
        <span style="font-family:'Archivo Expanded',sans-serif; font-weight:900;
        text-transform:uppercase; letter-spacing:0.02em; font-size:1.0rem;">{text_left}</span>
        <span style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em;
        color:{INK_60};">{text_right}</span></div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders  (logic unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models")
def load_artifacts():
    meta_path = os.path.join(MODELS_DIR, "meta.pkl")
    if not os.path.exists(meta_path):
        return None, None, None, None
    meta        = joblib.load(meta_path)
    tfidf_model = joblib.load(os.path.join(MODELS_DIR, "tfidf_model.pkl"))
    feat_model  = joblib.load(os.path.join(MODELS_DIR, "features_model.pkl"))
    emb_model   = joblib.load(os.path.join(MODELS_DIR, "embeddings_model.pkl"))
    return meta, tfidf_model, feat_model, emb_model


@st.cache_resource(show_spinner="Loading sentence encoder")
def load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def get_vader_sia():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


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


# ─────────────────────────────────────────────────────────────────────────────
# Masthead + navigation
# ─────────────────────────────────────────────────────────────────────────────
def masthead(meta):
    sub = "NLP churn detection from raw review text"
    if meta:
        n_methods = len(meta["models"])
        sub = f"{meta['n_train'] + meta['n_test']:,} reviews · churn vs retain · {n_methods} NLP methods"
    st.markdown(
        f"""<div class="masthead">
        <span class="brand">Review Intelligence</span>
        <span class="meta">{sub}</span></div>""",
        unsafe_allow_html=True,
    )


def top_nav():
    labels = [f"{i+1:02d}  {n}" for i, n in enumerate(NAV)]
    choice = st.radio("nav", labels, horizontal=True, label_visibility="collapsed")
    return NAV[labels.index(choice)]


# ─────────────────────────────────────────────────────────────────────────────
# 01 · OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def section_overview(meta):
    n_total   = meta["n_train"] + meta["n_test"]
    best_auc  = max(m["roc_auc"] for m in meta["models"].values())
    best_name = max(meta["models"], key=lambda k: meta["models"][k]["roc_auc"])

    n_methods = len(meta["models"])
    n_word = {3: "Three", 4: "Four", 5: "Five", 6: "Six"}.get(n_methods, str(n_methods))
    n_word_lower = n_word.lower()

    st.markdown('<div class="idx">01 / OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown("# Can text alone<br>predict churn?", unsafe_allow_html=True)
    st.markdown(
        f"""<p class="lede">{n_word} NLP methods, from a TF-IDF baseline to a zero-shot LLM,
        trained on <b>{n_total:,}</b> reviews sampled from the 650K Yelp Review Full corpus.
        One-and-two-star reviews are treated as churn signal, four-and-five-star as retention.</p>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<div class="kpis">
        <div class="kpi"><div class="v">{meta['n_train']:,}</div><div class="k">Train reviews</div></div>
        <div class="kpi"><div class="v">{meta['n_test']:,}</div><div class="k">Test reviews</div></div>
        <div class="kpi"><div class="v">{meta['churn_rate']:.0%}</div><div class="k">Churn base rate</div></div>
        <div class="kpi"><div class="v accent">{best_auc:.3f}</div><div class="k">Best AUC-ROC</div></div>
        </div>""",
        unsafe_allow_html=True,
    )

    rule("The problem", "Why language, not rating")
    cL, cR = st.columns([3, 2], gap="large")
    with cL:
        st.markdown(
            f"""<p class="lede">By the time a customer leaves a one-star review, they have
            <b>already churned</b>. Rating prediction is reactive: it describes what happened,
            not what will happen.</p>
            <p class="lede">The real question is whether churn is legible in the language itself,
            in word choice, tone and emotion, before any explicit rating is given. This project
            answers it by escalating through {n_word_lower} progressively richer representations
            of the same text.</p>""",
            unsafe_allow_html=True,
        )
    with cR:
        rows = []
        for name, m in meta["models"].items():
            mark = "  ◆" if name == best_name else ""
            rows.append({
                "Model": name + mark,
                "AUC": f"{m['roc_auc']:.3f}",
                "F1": f"{m['f1']:.3f}",
                "Time": f"{m['training_time']:.0f}s",
            })
        st.markdown('<div class="idx" style="margin-bottom:0.4rem;">KEY RESULTS</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    rule("Methodology", f"{n_word} representations of text")
    steps = [
        ("TF-IDF baseline",
         "Word-frequency vectors into logistic regression. Fast, fully interpretable floor."),
        ("Engineered features",
         "22 hand-built signals: VADER sentiment, keyword flags, stylistic ratios, into XGBoost."),
        ("Sentence embeddings",
         "all-MiniLM-L6-v2 maps each review to a 384-dim dense vector of meaning, not words."),
        ("Zero-shot LLM",
         "No training at all. Raw language-model reasoning, as an off-the-shelf upper reference."),
    ]
    cells = "".join(
        f"""<div class="step"><div class="n">{i+1:02d}</div>
        <div class="t">{t}</div><div class="d">{d}</div></div>"""
        for i, (t, d) in enumerate(steps)
    )
    st.markdown(f'<div class="steps">{cells}</div>', unsafe_allow_html=True)

    rule("Findings", "Computed from the loaded artifact")
    auc_vals = {k: v["roc_auc"] for k, v in meta["models"].items()}
    d_emb = auc_vals["Sentence Embeddings + XGBoost"] - auc_vals["TF-IDF + LR"]
    d_zs  = auc_vals["Zero-Shot LLM"] - auc_vals["Engineered Features + XGBoost"]

    shap_feat_names = meta.get("shap_feature_names", [])
    shap_vals_arr   = np.array(meta.get("shap_values", []))
    top_shap_feat = None
    if len(shap_feat_names) and shap_vals_arr.ndim == 2:
        top_shap_feat = shap_feat_names[int(np.abs(shap_vals_arr).mean(axis=0).argmax())]

    findings = [
        f"Best model, **{best_name}**, reaches **{best_auc:.3f}** AUC-ROC. Strong predictive signal from text alone.",
        f"Sentence embeddings {'beat' if d_emb > 0 else 'trail'} the TF-IDF floor by **{abs(d_emb):.3f}** AUC. Semantic meaning adds measurable value.",
        f"The zero-shot LLM {'beats' if d_zs > 0 else 'trails'} engineered features by **{abs(d_zs):.3f}** AUC with no training data.",
        *([f"**{top_shap_feat}** is the highest-impact engineered feature by mean absolute SHAP."]
          if top_shap_feat else []),
        f"Churn vocabulary is specific: *{', '.join(meta['top_churn_words'][:4])}* rank among the most predictive tokens.",
    ]
    for f in findings:
        st.markdown(f"- {f}")


# ─────────────────────────────────────────────────────────────────────────────
# 02 · MODELS
# ─────────────────────────────────────────────────────────────────────────────
def section_model_comparison(meta):
    n_methods = len(meta["models"])
    n_word = {3: "Three", 4: "Four", 5: "Five", 6: "Six"}.get(n_methods, str(n_methods))

    st.markdown('<div class="idx">02 / MODELS</div>', unsafe_allow_html=True)
    st.markdown("## Model comparison")
    st.markdown(
        f'<p class="lede">{n_word} representations of the same reviews, ranked. The headline bar '
        f'reads top to bottom by AUC-ROC, the curves below show how each separates churn from '
        f'retention, and the table carries every metric.</p>', unsafe_allow_html=True)

    best_name = max(meta["models"], key=lambda k: meta["models"][k]["roc_auc"])

    # Order rows by AUC-ROC descending so the table and headline bar actually read as a ranking.
    ranked = sorted(meta["models"].items(), key=lambda kv: kv[1]["roc_auc"], reverse=True)

    rows = []
    for name, m in ranked:
        rows.append({
            "Model":          ("◆ " if name == best_name else "  ") + name,
            "Accuracy":       round(m["accuracy"], 4),
            "Precision":      round(m["precision"], 4),
            "Recall":         round(m["recall"], 4),
            "F1":             round(m["f1"], 4),
            "AUC-ROC":        round(m["roc_auc"], 4),
            "Avg Precision":  round(m["avg_precision"], 4),
            "Train Time (s)": m["training_time"],
        })

    # Headline ranking bar. Plotly reverses the y axis, so feed it worst-to-best
    # and the highest AUC lands on top.
    bar_rows = list(reversed(rows))
    fig_rank = go.Figure(go.Bar(
        x=[r["AUC-ROC"] for r in bar_rows], y=[r["Model"] for r in bar_rows], orientation="h",
        marker_color=[RED if "◆" in r["Model"] else INK for r in bar_rows],
        text=[f'{r["AUC-ROC"]:.3f}' for r in bar_rows], textposition="outside"))
    fig_rank.update_layout(**PLOTLY_LAYOUT, title="Ranked by AUC-ROC",
                           xaxis=dict(range=[0.5, 1.06]), height=300,
                           margin=dict(l=0, r=0, t=40, b=10))
    st.plotly_chart(fig_rank, use_container_width=True)

    # Curves side by side: ROC | Precision-Recall
    cL, cR = st.columns(2, gap="large")
    with cL:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(dash="dot", color=INK_60, width=1), name="Random"))
        for i, (name, m) in enumerate(meta["models"].items()):
            roc = m["roc_curve"]
            fig.add_trace(go.Scatter(
                x=roc["fpr"], y=roc["tpr"], mode="lines",
                name=f"{name} ({m['roc_auc']:.3f})",
                line=dict(color=SERIES[i % len(SERIES)], width=4 if name == best_name else 1.6)))
        fig.update_layout(**PLOTLY_LAYOUT, title="ROC curve",
                          xaxis_title="False positive rate", yaxis_title="True positive rate",
                          legend=dict(x=0.4, y=0.06, font=dict(size=10)), height=440)
        st.plotly_chart(fig, use_container_width=True)
    with cR:
        fig = go.Figure()
        cr = meta["churn_rate"]
        fig.add_hline(y=cr, line_dash="dot", line_color=INK_60,
                      annotation_text=f"Base rate {cr:.2f}")
        for i, (name, m) in enumerate(meta["models"].items()):
            pr = m["pr_curve"]
            fig.add_trace(go.Scatter(
                x=pr["recall"], y=pr["precision"], mode="lines",
                name=f"{name} ({m['avg_precision']:.3f})",
                line=dict(color=SERIES[i % len(SERIES)], width=4 if name == best_name else 1.6)))
        fig.update_layout(**PLOTLY_LAYOUT, title="Precision-recall",
                          xaxis_title="Recall", yaxis_title="Precision",
                          legend=dict(x=0.0, y=0.06, font=dict(size=10)), height=440)
        st.plotly_chart(fig, use_container_width=True)

    # Full metrics table + compact confusion matrices, under one rule
    rule("Every metric", "Test set, plus confusion at the chosen threshold")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    cm_fig = make_subplots(rows=1, cols=len(rows), horizontal_spacing=0.035,
                           subplot_titles=[n.replace(" + ", "+") for n in meta["models"]])
    for i, (name, m) in enumerate(meta["models"].items()):
        cm = np.array(m["confusion_matrix"])
        cm_fig.add_trace(go.Heatmap(
            z=cm, x=["Pred retain", "Pred churn"], y=["Retain", "Churn"],
            colorscale=[[0, PAPER], [1, RED]], showscale=False,
            text=cm, texttemplate="%{text}", textfont=dict(size=13, color=INK)),
            row=1, col=i + 1)
        cm_fig.update_yaxes(autorange="reversed", row=1, col=i + 1,
                            showticklabels=(i == 0))   # y labels only on the first matrix
        cm_fig.update_xaxes(tickfont=dict(size=10), row=1, col=i + 1)
    cm_fig.update_layout(**PLOTLY_LAYOUT, height=240, margin=dict(l=10, r=10, t=44, b=10))
    cm_fig.update_annotations(font_size=11)
    st.plotly_chart(cm_fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 03 · ANALYZER
# ─────────────────────────────────────────────────────────────────────────────
def section_analyze(meta, tfidf_model, feat_model, emb_model):
    st.markdown('<div class="idx">03 / ANALYZE</div>', unsafe_allow_html=True)
    st.markdown("## Score any review")
    st.markdown(
        f'<p class="lede">One review, every lens. Three trained models score it for churn risk, '
        f'SHAP and VADER explain why, and Gemini extracts structured fields from the same text.</p>',
        unsafe_allow_html=True)

    example_type = st.selectbox(
        "Quick-load an example",
        ["Blank", "Churn signal example", "Retention signal example"])

    example_text = ""
    if example_type == "Churn signal example" and meta["sample_reviews"]["churn"]:
        example_text = meta["sample_reviews"]["churn"][0]
    elif example_type == "Retention signal example" and meta["sample_reviews"]["retain"]:
        example_text = meta["sample_reviews"]["retain"][0]

    text_input = st.text_area("Customer review", value=example_text, height=130,
                              placeholder="Type or paste any review.")
    run = st.button("Analyze")

    if run and text_input.strip():
        with st.spinner("Scoring"):
            _run_analysis(text_input.strip(), meta, tfidf_model, feat_model, emb_model)
    elif run:
        st.warning("Enter a review first.")

    # Structured LLM extraction lives in the same section, on the same review text.
    gemini_block(text_input.strip(), tfidf_model)


def _run_analysis(text, meta, tfidf_model, feat_model, emb_model):
    import xgboost as xgb

    prob_tfidf = float(tfidf_model.predict_proba([text])[0, 1])
    feats_dict = extract_features_single(text)
    feats_df   = pd.DataFrame([feats_dict])
    prob_feat  = float(feat_model.predict_proba(feats_df.values)[0, 1])
    encoder    = load_encoder()
    emb        = encoder.encode([text], batch_size=1, show_progress_bar=False, convert_to_numpy=True)
    prob_emb   = float(emb_model.predict_proba(emb)[0, 1])

    model_results = [
        ("TF-IDF + LR", prob_tfidf),
        ("Engineered features", prob_feat),
        ("Sentence embeddings", prob_emb),
    ]

    rule("Model scores", "Churn probability")
    cols = st.columns(3, gap="medium")
    for col, (mname, prob) in zip(cols, model_results):
        rl, rc = risk_label(prob)
        col.markdown(
            f"""<div class="score">
            <div class="m">{mname}</div>
            <div class="v" style="color:{rc};">{prob:.0%}</div>
            <div style="margin-top:0.4rem;"><span class="tag" style="border-color:{rc};color:{rc};">{rl}</span></div>
            </div>""", unsafe_allow_html=True)

    rule("Feature explanation", "Engineered model, single review")
    try:
        booster = feat_model.get_booster()
        feat_names = list(feats_dict.keys())
        dmat = xgb.DMatrix(feats_df.values, feature_names=feat_names)
        contribs  = booster.predict(dmat, pred_contribs=True)
        shap_vals = contribs[0, :-1]
        shap_df = pd.DataFrame({"feature": feat_names, "shap": shap_vals})
        shap_df = shap_df.reindex(shap_df["shap"].abs().sort_values(ascending=True).index)
        fig = go.Figure(go.Bar(
            x=shap_df["shap"], y=shap_df["feature"], orientation="h",
            marker_color=[RED if v > 0 else INK for v in shap_df["shap"]]))
        fig.update_layout(**PLOTLY_LAYOUT, title="SHAP, positive pushes toward churn",
                          height=420, xaxis_title="SHAP value", margin=dict(l=0, r=0, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"SHAP explanation unavailable: {e}")

    rule("VADER sentiment", "Lexicon breakdown")
    vader_data = {
        "NEG": feats_dict["vader_neg"], "NEU": feats_dict["vader_neu"],
        "POS": feats_dict["vader_pos"], "COMPOUND": (feats_dict["vader_compound"] + 1) / 2,
    }
    fig_v = go.Figure(go.Bar(
        x=list(vader_data.values()), y=list(vader_data.keys()), orientation="h",
        marker_color=[RED, "#BFB8A8", INK, INK_60],
        text=[f"{v:.3f}" for v in vader_data.values()], textposition="outside"))
    fig_v.update_layout(**PLOTLY_LAYOUT, height=200, margin=dict(l=0, r=60, t=10, b=10),
                        xaxis=dict(range=[0, 1.15]))
    st.plotly_chart(fig_v, use_container_width=True)
    st.caption("Compound normalised to [0,1] for display. Raw range is minus one to plus one.")

    rule("Keyword signals", "Lexical flags found")
    churn_kw  = ["never", "worst", "terrible", "waste", "horrible", "disappoint"]
    retain_kw = ["amazing", "excellent", "love", "recommend"]
    found_churn  = [w for w in churn_kw  if w in text.lower()]
    found_retain = [w for w in retain_kw if w in text.lower()]
    kc1, kc2 = st.columns(2)
    with kc1:
        if found_churn:
            tags = "".join(f'<span class="tag red">{w}</span>' for w in found_churn)
            st.markdown(f"Churn signals: {tags}", unsafe_allow_html=True)
        else:
            st.markdown(f'<span style="color:{INK_60};">No churn keywords found.</span>',
                        unsafe_allow_html=True)
    with kc2:
        if found_retain:
            tags = "".join(f'<span class="tag">{w}</span>' for w in found_retain)
            st.markdown(f"Retention signals: {tags}", unsafe_allow_html=True)
        else:
            st.markdown(f'<span style="color:{INK_60};">No retention keywords found.</span>',
                        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 04 · LLM EXTRACT (Gemini)
# ─────────────────────────────────────────────────────────────────────────────
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

RISK_COLORS = {"explicit": RED, "implicit": INK, "none": INK_60}


def render_gemini_result(result: dict, ml_score: float):
    rule("Extracted features", "Structured output")
    col1, col2 = st.columns(2, gap="medium")

    simple_fields = [
        ("complaint_category", result.get("complaint_category", "none")),
        ("churn_intent",       result.get("churn_intent", "none")),
        ("urgency",            result.get("urgency", "low")),
        ("emotion",            result.get("emotion", "neutral")),
    ]
    list_fields = [
        ("key_phrases",      result.get("key_phrases", [])),
        ("positive_aspects", result.get("positive_aspects", [])),
    ]

    with col1:
        for key, val in simple_fields:
            color = RISK_COLORS.get(val, INK)
            st.markdown(
                f"""<div class="blk" style="margin-bottom:0.5rem;padding:0.6rem 0.9rem;">
                <span class="m" style="font-size:0.66rem;text-transform:uppercase;
                letter-spacing:0.1em;color:{INK_60};">{key.replace('_',' ').upper()}</span><br>
                <strong style="color:{color};font-size:1.05rem;">{val}</strong></div>""",
                unsafe_allow_html=True)
    with col2:
        for key, items in list_fields:
            pills = "".join(f'<span class="tag">{item}</span>' for item in (items or ["none"]))
            st.markdown(
                f"""<div class="blk" style="margin-bottom:0.5rem;padding:0.6rem 0.9rem;">
                <span class="m" style="font-size:0.66rem;text-transform:uppercase;
                letter-spacing:0.1em;color:{INK_60};">{key.replace('_',' ').upper()}</span><br>
                {pills}</div>""", unsafe_allow_html=True)

    rule("Score comparison", "ML vs LLM")
    gemini_score = result.get("churn_risk_score", 5)
    gemini_norm  = gemini_score / 10.0
    sc1, sc2 = st.columns(2, gap="medium")
    with sc1:
        _, c = risk_label(ml_score)
        sc1.markdown(
            f"""<div class="score"><div class="m">ML model, TF-IDF + LR</div>
            <div class="v" style="color:{c};">{ml_score:.0%}</div></div>""",
            unsafe_allow_html=True)
    with sc2:
        _, c = risk_label(gemini_norm)
        sc2.markdown(
            f"""<div class="score"><div class="m">Gemini, zero-shot</div>
            <div class="v" style="color:{c};">{gemini_score}/10</div></div>""",
            unsafe_allow_html=True)


def gemini_block(text, tfidf_model):
    """Structured LLM extraction on the review typed above. Live with a key, demo without."""
    rule("Structured extraction", "Gemini 1.5 Flash")
    st.markdown(
        f'<p class="lede">Beyond a probability, Gemini turns the same review into typed fields: '
        f'complaint category, churn intent, urgency, emotion. Semantic structure the '
        f'classifiers never see.</p>', unsafe_allow_html=True)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Gemini API key", type="password",
                                placeholder="AIzaSy...  or set GEMINI_API_KEY",
                                help="Free key at https://aistudio.google.com")

    if api_key:
        import google.generativeai as genai
        import json
        if st.button("Extract with Gemini"):
            if not text:
                st.warning("Type a review above first.")
            else:
                with st.spinner("Calling Gemini"):
                    try:
                        genai.configure(api_key=api_key)
                        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                        prompt   = EXTRACTION_PROMPT.format(text=text)
                        response = gemini_model.generate_content(prompt)
                        raw      = response.text.strip()
                        raw = re.sub(r"^```(?:json)?\s*", "", raw)
                        raw = re.sub(r"\s*```$", "", raw)
                        result   = json.loads(raw)
                        ml_score = float(tfidf_model.predict_proba([text])[0, 1])
                        render_gemini_result(result, ml_score)
                    except json.JSONDecodeError:
                        st.error("Gemini returned invalid JSON. Try again.")
                        st.code(raw)
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.markdown(
            f"""<div class="blk blk-fill" style="border-color:{RED};margin-bottom:1rem;">
            <strong>Demo mode.</strong> No API key set, showing a pre-computed example below.
            Enter a free Gemini key above to extract structure from your own review.</div>""",
            unsafe_allow_html=True)
        st.markdown(
            f"""<div class="blk" style="margin-bottom:1rem;">
            <span class="m" style="font-size:0.66rem;color:{INK_60};letter-spacing:0.1em;">EXAMPLE REVIEW</span><br>
            <span style="font-style:italic;">"{DEMO_GEMINI['review']}"</span></div>""",
            unsafe_allow_html=True)
        render_gemini_result(DEMO_GEMINI["result"], DEMO_GEMINI["ml_score"])


# ─────────────────────────────────────────────────────────────────────────────
# 05 · INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def section_business_insights(meta):
    st.markdown('<div class="idx">04 / INSIGHTS</div>', unsafe_allow_html=True)
    st.markdown("## What drives churn")
    st.markdown(
        f'<p class="lede">Three reads on the same question: which words carry the signal, '
        f'which engineered features the model leans on, and how sentiment itself splits '
        f'churn from retention.</p>', unsafe_allow_html=True)

    # ── Predictive words ────────────────────────────────────────────────────
    rule("Predictive words", "TF-IDF logistic-regression coefficients")
    st.markdown(
        f'<p class="lede">The 20 most predictive tokens, split by direction.</p>',
        unsafe_allow_html=True)
    wc1, wc2 = st.columns(2, gap="medium")
    with wc1:
        churn_w = meta["top_churn_words"]
        fig = go.Figure(go.Bar(x=list(range(len(churn_w), 0, -1)), y=churn_w,
                               orientation="h", marker_color=RED))
        fig.update_layout(**PLOTLY_LAYOUT, title="Churn signal words",
                          xaxis_title="Relative rank", height=520)
        st.plotly_chart(fig, use_container_width=True)
    with wc2:
        retain_w = meta["top_retain_words"]
        fig2 = go.Figure(go.Bar(x=list(range(len(retain_w), 0, -1)), y=retain_w,
                                orientation="h", marker_color=INK))
        fig2.update_layout(**PLOTLY_LAYOUT, title="Retention signal words",
                           xaxis_title="Relative rank", height=520)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Feature impact ──────────────────────────────────────────────────────
    rule("Feature impact", "SHAP, engineered XGBoost, 200 held-out reviews")
    st.markdown(
        f'<p class="lede">SHAP values for the engineered XGBoost model. Positive pushes the '
        f'prediction toward churn.</p>', unsafe_allow_html=True)
    shap_vals  = np.array(meta["shap_values"])
    feat_names = meta["shap_feature_names"]
    X_sample   = meta["X_test_features_sample"]
    X_arr = X_sample.values if isinstance(X_sample, pd.DataFrame) else np.array(X_sample)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = np.argsort(mean_abs)
    sorted_names = [feat_names[i] for i in order]
    sorted_shap  = shap_vals[:, order]
    sorted_X     = X_arr[:, order]
    X_norm = (sorted_X - sorted_X.min(axis=0)) / (np.ptp(sorted_X, axis=0) + 1e-9)
    rng_j = np.random.default_rng(42)
    jitter = rng_j.uniform(-0.35, 0.35, size=sorted_shap.shape)
    fig = go.Figure()
    for j, fname in enumerate(sorted_names):
        fig.add_trace(go.Scatter(
            x=sorted_shap[:, j], y=np.full(sorted_shap.shape[0], j) + jitter[:, j],
            mode="markers", name=fname, showlegend=False,
            marker=dict(size=4, color=X_norm[:, j],
                        colorscale=[[0, INK_60], [0.5, "#BFB8A8"], [1, RED]],
                        opacity=0.65,
                        colorbar=dict(title="Feature<br>value", thickness=10, len=0.7)
                        if j == len(sorted_names) - 1 else None)))
    fig.update_layout(**PLOTLY_LAYOUT, title="SHAP beeswarm, engineered features",
                      xaxis_title="SHAP value (impact on churn probability)",
                      yaxis=dict(tickvals=list(range(len(sorted_names))), ticktext=sorted_names),
                      height=600)
    st.plotly_chart(fig, use_container_width=True)

    # ── Sentiment split ─────────────────────────────────────────────────────
    rule("Sentiment split", "VADER compound score, churn vs retained")
    st.markdown(
        f'<p class="lede">Where the two distributions diverge is the signal the models '
        f'exploit.</p>', unsafe_allow_html=True)
    vader_stats = meta.get("vader_stats", {})
    churn_vals  = vader_stats.get("churn", [])
    retain_vals = vader_stats.get("retain", [])
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=churn_vals, name="Churn", marker_color=RED,
                               opacity=0.75, nbinsx=50, histnorm="probability"))
    fig.add_trace(go.Histogram(x=retain_vals, name="Retained", marker_color=INK,
                               opacity=0.6, nbinsx=50, histnorm="probability"))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay",
                      title="VADER compound score distribution",
                      xaxis_title="Compound score (minus one to plus one)",
                      yaxis_title="Probability", height=420, legend=dict(x=0.02, y=0.98))
    st.plotly_chart(fig, use_container_width=True)
    if churn_vals and retain_vals:
        churn_med  = float(np.median(churn_vals))
        retain_med = float(np.median(retain_vals))
        st.caption(
            f"Median compound, churn {churn_med:.3f} versus retained {retain_med:.3f}. "
            "Where the distributions overlap, models lean on subtler linguistic cues.")


# ─────────────────────────────────────────────────────────────────────────────
def page_no_models():
    st.markdown('<div class="idx">SETUP</div>', unsafe_allow_html=True)
    st.markdown("## Model artifacts not found")
    st.markdown("Run the training pipeline from the project root, then reload.")
    st.code("python src/train.py", language="bash")
    st.markdown(
        "Downloads the Yelp Review Full dataset on first run, trains the NLP models, "
        "and writes artifacts to `models/`. Roughly 20 to 60 minutes depending on hardware.")


def main():
    meta, tfidf_model, feat_model, emb_model = load_artifacts()
    masthead(meta)

    if meta is None:
        page_no_models()
        return

    page = top_nav()
    if page == "Overview":
        section_overview(meta)
    elif page == "Models":
        section_model_comparison(meta)
    elif page == "Analyze":
        section_analyze(meta, tfidf_model, feat_model, emb_model)
    elif page == "Insights":
        section_business_insights(meta)


if __name__ == "__main__":
    main()
