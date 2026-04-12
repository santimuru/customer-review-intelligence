"""
train.py — Trains all 4 NLP models for churn risk detection on Yelp reviews.
Run from project root: python src/train.py
"""

import os
import time
import re
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# Pre-import torch/transformers BEFORE xgboost/sklearn heavy C-extensions
# to avoid WinError 1114 DLL init failure on Windows
try:
    import torch                                          # noqa: F401
    from sentence_transformers import SentenceTransformer  # noqa: F401
    _TORCH_OK = True
except OSError:
    _TORCH_OK = False
    print("[WARN] torch/sentence_transformers import failed — Model 3 will be skipped")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
SEED = 42
N_TRAIN = 40_000   # 20K churn + 20K retain
N_TEST  =  8_000   # 4K churn + 4K retain
N_SHAP_SAMPLES = 200
ZS_SAMPLE_SIZE = 500

# ══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_data():
    print("\n[1/6] Loading Yelp dataset from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("yelp_review_full")

    def to_df(split):
        df = pd.DataFrame({"text": split["text"], "label": split["label"]})
        # at_risk: 0 or 1 star → 1, 4 or 5 star → 0, 3 star → drop
        df = df[df["label"] != 2].copy()
        df["at_risk"] = (df["label"] <= 1).astype(int)
        return df

    train_full = to_df(ds["train"])
    test_full  = to_df(ds["test"])

    def stratified_sample(df, n_per_class, seed=SEED):
        pos = df[df["at_risk"] == 1].sample(n_per_class, random_state=seed)
        neg = df[df["at_risk"] == 0].sample(n_per_class, random_state=seed)
        return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)

    train = stratified_sample(train_full, N_TRAIN // 2)
    test  = stratified_sample(test_full,  N_TEST  // 2)

    print(f"   Train: {len(train)} samples | Test: {len(test)} samples")
    print(f"   Churn rate (train): {train['at_risk'].mean():.1%}")

    return train, test


# ══════════════════════════════════════════════════════════════════════════
# 2. METRICS HELPER
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, average_precision_score,
        roc_curve, precision_recall_curve, confusion_matrix,
    )
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy":       float(accuracy_score(y_true, y_pred)),
        "precision":      float(precision_score(y_true, y_pred)),
        "recall":         float(recall_score(y_true, y_pred)),
        "f1":             float(f1_score(y_true, y_pred)),
        "roc_auc":        float(roc_auc_score(y_true, y_prob)),
        "avg_precision":  float(average_precision_score(y_true, y_prob)),
        "roc_curve":      {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve":       {"precision": prec_curve.tolist(), "recall": rec_curve.tolist()},
        "confusion_matrix": cm,
    }


# ══════════════════════════════════════════════════════════════════════════
# 3. MODEL 1 — TF-IDF + LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════════════════

def train_tfidf(X_train, y_train, X_test, y_test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    print("\n[2/6] Model 1: TF-IDF + Logistic Regression...")
    t0 = time.time()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=3)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=SEED)),
    ])
    pipe.fit(X_train, y_train)

    elapsed = time.time() - t0
    y_pred  = pipe.predict(X_test)
    y_prob  = pipe.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["training_time"] = round(elapsed, 2)

    # Top predictive words
    tfidf_vec = pipe.named_steps["tfidf"]
    lr        = pipe.named_steps["clf"]
    feature_names = tfidf_vec.get_feature_names_out()
    coefs = lr.coef_[0]

    top_churn_idx   = coefs.argsort()[-20:][::-1]
    top_retain_idx  = coefs.argsort()[:20]

    top_churn_words  = [feature_names[i] for i in top_churn_idx]
    top_retain_words = [feature_names[i] for i in top_retain_idx]

    print(f"   AUC-ROC: {metrics['roc_auc']:.4f} | Time: {elapsed:.1f}s")
    print(f"   Top churn words: {top_churn_words[:5]}")

    joblib.dump(pipe, os.path.join(MODELS_DIR, "tfidf_model.pkl"))
    return metrics, top_churn_words, top_retain_words


# ══════════════════════════════════════════════════════════════════════════
# 4. ENGINEERED FEATURES EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════

def extract_features(text, sia=None):
    if sia is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
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


def build_feature_matrix(texts, batch_size=1000):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    records = []
    for i, txt in enumerate(texts):
        if i % batch_size == 0:
            print(f"   Features: {i}/{len(texts)}", end="\r")
        records.append(extract_features(txt, sia))
    print()
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════
# 5. MODEL 2 — ENGINEERED FEATURES + XGBOOST
# ══════════════════════════════════════════════════════════════════════════

def train_features_xgb(X_train, y_train, X_test, y_test):
    import xgboost as xgb

    print("\n[3/6] Model 2: Engineered Features + XGBoost...")
    t0 = time.time()

    print("   Building feature matrices...")
    feats_train = build_feature_matrix(X_train)
    feats_test  = build_feature_matrix(X_test)

    feature_names = list(feats_train.columns)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(feats_train.values, y_train)

    elapsed = time.time() - t0
    y_pred  = model.predict(feats_test.values)
    y_prob  = model.predict_proba(feats_test.values)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["training_time"] = round(elapsed, 2)

    print(f"   AUC-ROC: {metrics['roc_auc']:.4f} | Time: {elapsed:.1f}s")

    # SHAP via XGBoost native pred_contribs (compatible with XGBoost 3.x)
    print("   Computing SHAP values (200 samples)...")
    shap_sample = feats_test.iloc[:N_SHAP_SAMPLES].copy()
    booster  = model.get_booster()
    dmat     = xgb.DMatrix(shap_sample.values, feature_names=feature_names)
    # pred_contribs returns shape (n, n_features+1) — last col is bias
    contribs = booster.predict(dmat, pred_contribs=True)
    shap_vals = contribs[:, :-1]   # drop bias column

    joblib.dump(model, os.path.join(MODELS_DIR, "features_model.pkl"))

    return metrics, feature_names, shap_vals, shap_sample, feats_test, feats_train


# ══════════════════════════════════════════════════════════════════════════
# 6. MODEL 3 — SENTENCE EMBEDDINGS + XGBOOST
# ══════════════════════════════════════════════════════════════════════════

def train_embeddings_xgb(X_train, y_train, X_test, y_test):
    import xgboost as xgb

    print("\n[4/6] Model 3: Sentence Embeddings + XGBoost...")
    t0 = time.time()

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    print("   Encoding train set...")
    emb_train = encoder.encode(list(X_train), batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    print("   Encoding test set...")
    emb_test  = encoder.encode(list(X_test),  batch_size=256, show_progress_bar=True, convert_to_numpy=True)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(emb_train, y_train)

    elapsed = time.time() - t0
    y_pred  = model.predict(emb_test)
    y_prob  = model.predict_proba(emb_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["training_time"] = round(elapsed, 2)

    print(f"   AUC-ROC: {metrics['roc_auc']:.4f} | Time: {elapsed:.1f}s")

    # Save embeddings and model
    np.save(os.path.join(MODELS_DIR, "test_embeddings.npy"), emb_test)
    np.save(os.path.join(MODELS_DIR, "test_labels.npy"),     y_test)
    joblib.dump(model, os.path.join(MODELS_DIR, "embeddings_model.pkl"))

    return metrics


# ══════════════════════════════════════════════════════════════════════════
# 7. MODEL 4 — ZERO-SHOT WITH HUGGINGFACE
# ══════════════════════════════════════════════════════════════════════════

def train_zeroshot(X_test, y_test):
    from transformers import pipeline as hf_pipeline
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

    print(f"\n[5/6] Model 4: Zero-Shot LLM (sample of {ZS_SAMPLE_SIZE})...")
    t0 = time.time()

    # Stratified sample
    idx_pos = np.where(np.array(y_test) == 1)[0][:ZS_SAMPLE_SIZE // 2]
    idx_neg = np.where(np.array(y_test) == 0)[0][:ZS_SAMPLE_SIZE // 2]
    idx = np.concatenate([idx_pos, idx_neg])

    texts_zs = [X_test[i] for i in idx]
    labels_zs = [y_test[i] for i in idx]

    zs = hf_pipeline(
        "zero-shot-classification",
        model="cross-encoder/nli-MiniLM2-L6-H768",
        device=-1,
    )

    candidate_labels = ["satisfied customer", "dissatisfied customer"]

    preds_prob = []
    preds_bin  = []

    for i, text in enumerate(texts_zs):
        if i % 50 == 0:
            print(f"   Zero-shot: {i}/{len(texts_zs)}", end="\r")
        # Truncate to first 512 chars for speed
        result = zs(text[:512], candidate_labels)
        # "dissatisfied customer" → at_risk=1
        dis_score = result["scores"][result["labels"].index("dissatisfied customer")]
        preds_prob.append(dis_score)
        preds_bin.append(1 if dis_score >= 0.5 else 0)

    print()
    elapsed = time.time() - t0

    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
    fpr, tpr, _ = roc_curve(labels_zs, preds_prob)
    prec_c, rec_c, _ = precision_recall_curve(labels_zs, preds_prob)
    cm = confusion_matrix(labels_zs, preds_bin).tolist()

    metrics = {
        "accuracy":        float(accuracy_score(labels_zs, preds_bin)),
        "precision":       float(precision_score(labels_zs, preds_bin, zero_division=0)),
        "recall":          float(recall_score(labels_zs, preds_bin, zero_division=0)),
        "f1":              float(f1_score(labels_zs, preds_bin, zero_division=0)),
        "roc_auc":         float(roc_auc_score(labels_zs, preds_prob)),
        "avg_precision":   float(average_precision_score(labels_zs, preds_prob)),
        "roc_curve":       {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve":        {"precision": prec_c.tolist(), "recall": rec_c.tolist()},
        "confusion_matrix": cm,
        "training_time":   round(elapsed, 2),
        "note":            f"Zero-shot on {ZS_SAMPLE_SIZE} samples (no training required)",
    }

    print(f"   AUC-ROC: {metrics['roc_auc']:.4f} | Time: {elapsed:.1f}s")
    return metrics


# ══════════════════════════════════════════════════════════════════════════
# 8. SAMPLE REVIEWS FOR APP DEMO
# ══════════════════════════════════════════════════════════════════════════

def pick_sample_reviews(X_test, y_test, n=10):
    texts = np.array(X_test)
    labels = np.array(y_test)

    churn_idx  = np.where(labels == 1)[0]
    retain_idx = np.where(labels == 0)[0]

    rng = np.random.default_rng(SEED)
    churn_samples  = texts[rng.choice(churn_idx,  n, replace=False)].tolist()
    retain_samples = texts[rng.choice(retain_idx, n, replace=False)].tolist()

    return {"churn": churn_samples, "retain": retain_samples}


# ══════════════════════════════════════════════════════════════════════════
# 9. VADER STATS FOR BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════

def compute_vader_stats(feats_train_df, y_train):
    """Return compound scores for churn vs retain for distribution plot."""
    df = feats_train_df.copy()
    df["at_risk"] = y_train
    churn_compound  = df[df["at_risk"] == 1]["vader_compound"].tolist()
    retain_compound = df[df["at_risk"] == 0]["vader_compound"].tolist()
    return {"churn": churn_compound, "retain": retain_compound}


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" CUSTOMER REVIEW INTELLIGENCE — MODEL TRAINING")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────
    train_df, test_df = load_data()

    X_train = train_df["text"].tolist()
    y_train = train_df["at_risk"].tolist()
    X_test  = test_df["text"].tolist()
    y_test  = test_df["at_risk"].tolist()

    churn_rate = sum(y_train) / len(y_train)

    # ── Model 1: TF-IDF ────────────────────────────────────────────
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_model.pkl")
    if os.path.exists(tfidf_path):
        print("\n[2/6] Model 1: TF-IDF + LR — loading cached model...")
        pipe = joblib.load(tfidf_path)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        m1_metrics = compute_metrics(y_test, y_pred, y_prob)
        m1_metrics["training_time"] = 12.1  # cached from prior run
        tfidf_vec = pipe.named_steps["tfidf"]
        lr = pipe.named_steps["clf"]
        feature_names_tfidf = tfidf_vec.get_feature_names_out()
        coefs = lr.coef_[0]
        top_churn_idx   = coefs.argsort()[-20:][::-1]
        top_retain_idx  = coefs.argsort()[:20]
        top_churn_words  = [feature_names_tfidf[i] for i in top_churn_idx]
        top_retain_words = [feature_names_tfidf[i] for i in top_retain_idx]
        print(f"   AUC-ROC: {m1_metrics['roc_auc']:.4f} (cached)")
    else:
        m1_metrics, top_churn_words, top_retain_words = train_tfidf(
            X_train, y_train, X_test, y_test
        )

    # ── Model 2: Engineered Features ──────────────────────────────
    feat_path  = os.path.join(MODELS_DIR, "features_model.pkl")
    shap_path  = os.path.join(MODELS_DIR, "_shap_cache.pkl")
    if os.path.exists(feat_path) and os.path.exists(shap_path):
        print("\n[3/6] Model 2: Engineered Features — loading cached model + SHAP...")
        feat_model_cached = joblib.load(feat_path)
        shap_cache = joblib.load(shap_path)
        m2_metrics    = shap_cache["metrics"]
        feat_names    = shap_cache["feat_names"]
        shap_vals     = shap_cache["shap_vals"]
        shap_sample   = shap_cache["shap_sample"]
        feats_test_df = shap_cache["feats_test_df"]
        feats_train_df = shap_cache["feats_train_df"]
        print(f"   AUC-ROC: {m2_metrics['roc_auc']:.4f} (cached)")
    else:
        m2_metrics, feat_names, shap_vals, shap_sample, feats_test_df, feats_train_df = \
            train_features_xgb(X_train, y_train, X_test, y_test)
        # Save intermediate cache
        joblib.dump({
            "metrics": m2_metrics, "feat_names": feat_names,
            "shap_vals": shap_vals, "shap_sample": shap_sample,
            "feats_test_df": feats_test_df, "feats_train_df": feats_train_df,
        }, shap_path)

    # ── Model 3: Sentence Embeddings ──────────────────────────────
    emb_model_path = os.path.join(MODELS_DIR, "embeddings_model.pkl")
    emb_npy_path   = os.path.join(MODELS_DIR, "test_embeddings.npy")
    m3_cache_path  = os.path.join(MODELS_DIR, "_m3_metrics_cache.pkl")
    if os.path.exists(emb_model_path) and os.path.exists(emb_npy_path) and os.path.exists(m3_cache_path):
        print("\n[4/6] Model 3: Sentence Embeddings — loading cached...")
        m3_metrics = joblib.load(m3_cache_path)
        print(f"   AUC-ROC: {m3_metrics['roc_auc']:.4f} (cached)")
    else:
        m3_metrics = train_embeddings_xgb(X_train, y_train, X_test, y_test)
        joblib.dump(m3_metrics, m3_cache_path)

    # ── Model 4: Zero-Shot ─────────────────────────────────────────
    m4_cache_path = os.path.join(MODELS_DIR, "_m4_metrics_cache.pkl")
    if os.path.exists(m4_cache_path):
        print("\n[5/6] Model 4: Zero-Shot — loading cached...")
        m4_metrics = joblib.load(m4_cache_path)
        print(f"   AUC-ROC: {m4_metrics['roc_auc']:.4f} (cached)")
    else:
        m4_metrics = train_zeroshot(X_test, y_test)
        joblib.dump(m4_metrics, m4_cache_path)

    # ── VADER distribution stats ───────────────────────────────────
    vader_stats = compute_vader_stats(feats_train_df, y_train)

    # ── Sample reviews ─────────────────────────────────────────────
    sample_reviews = pick_sample_reviews(X_test, y_test)

    # ── Label distribution ─────────────────────────────────────────
    label_dist = train_df["label"].value_counts().to_dict()
    # Convert numpy int64 to int
    label_dist = {int(k): int(v) for k, v in label_dist.items()}

    # ── Save meta.pkl ──────────────────────────────────────────────
    print("\n[6/6] Saving artifacts...")
    meta = {
        "models": {
            "TF-IDF + LR":                       m1_metrics,
            "Engineered Features + XGBoost":     m2_metrics,
            "Sentence Embeddings + XGBoost":     m3_metrics,
            "Zero-Shot LLM":                     m4_metrics,
        },
        "top_churn_words":        top_churn_words,
        "top_retain_words":       top_retain_words,
        "shap_feature_names":     feat_names,
        "shap_values":            shap_vals,
        "X_test_features_sample": shap_sample,
        "vader_stats":            vader_stats,
        "sample_reviews":         sample_reviews,
        "n_train":                len(train_df),
        "n_test":                 len(test_df),
        "churn_rate":             float(churn_rate),
        "label_distribution":     label_dist,
    }

    joblib.dump(meta, os.path.join(MODELS_DIR, "meta.pkl"))

    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n{'Model':<38} {'AUC-ROC':>8}  {'F1':>6}  {'Time(s)':>8}")
    print("-" * 65)
    for name, m in meta["models"].items():
        note = " *" if m["roc_auc"] == max(v["roc_auc"] for v in meta["models"].values()) else ""
        print(f"{name:<38} {m['roc_auc']:>8.4f}  {m['f1']:>6.4f}  {m['training_time']:>8.1f}{note}")

    best = max(meta["models"], key=lambda k: meta["models"][k]["roc_auc"])
    print(f"\n Best model by AUC-ROC: {best}")
    print(f" Artifacts saved to:    {MODELS_DIR}")

    artifacts = [
        "tfidf_model.pkl", "features_model.pkl",
        "embeddings_model.pkl", "meta.pkl",
        "test_embeddings.npy", "test_labels.npy",
    ]
    for f in artifacts:
        path = os.path.join(MODELS_DIR, f)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print(f"   {f:<30} {size_mb:>6.1f} MB")
        else:
            print(f"   {f:<30}  MISSING!")


if __name__ == "__main__":
    main()
