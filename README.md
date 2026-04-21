# Customer Review Intelligence

> **Live demo:** [customer-review-intelligence-santiagomuru.streamlit.app](https://customer-review-intelligence-santiagomuru.streamlit.app/)

By the time a customer leaves a 1-star review, they have already churned. The real question is whether churn signals are detectable in language itself — the choice of words, tone, and emotion — before an explicit rating is given.

This project tests four progressively sophisticated NLP methods on 650K Yelp reviews, treating 1-2 star reviews as churn signals and 4-5 star reviews as retention signals.

---

## Live Dashboard

| Section             | What you'll find                                                              |
| ------------------- | ----------------------------------------------------------------------------- |
| 📊 Overview         | Problem framing, methodology progression, KPIs, key findings                 |
| 📈 Model Comparison | ROC, Precision-Recall, metrics table, confusion matrices for all 4 models    |
| ⭐ Text Analyzer    | Score any review live across 3 models with SHAP explanation + VADER breakdown |
| 🤖 LLM Features     | Gemini-powered structured extraction: intent, emotion, complaint category    |
| 💡 Business Insights | Top churn/retention words, SHAP beeswarm, VADER sentiment distributions     |

---

## Four NLP Methods

| Method                              | Approach                                                        |
| ----------------------------------- | --------------------------------------------------------------- |
| **TF-IDF + Logistic Regression**    | Word frequencies - fast, interpretable, strong baseline         |
| **Engineered Features + XGBoost**   | 22 hand-crafted signals: VADER sentiment, keyword flags, style  |
| **Sentence Embeddings + XGBoost**   | all-MiniLM-L6-v2 maps each review to a 384-dim semantic vector  |
| **Zero-Shot LLM (Gemini)**          | No training - raw language model reasoning on the binary task   |

Each method adds a layer beyond the previous: from word counts to meaning to zero-shot reasoning.

---

## Key Findings

- Sentence embeddings outperform the TF-IDF baseline in AUC-ROC, confirming semantic meaning adds measurable signal.
- Zero-shot Gemini achieves competitive performance with no training data at all.
- VADER compound score is the single strongest engineered feature (per SHAP) - sentiment polarity dominates.
- Churn language is specific: words like *worst*, *terrible*, *waste*, *never* are highly predictive across methods.

---

## Dataset

Yelp Review Full dataset - 650K reviews, 5 rating classes, mapped to binary churn/retain. Downloaded automatically via HuggingFace Datasets (~700MB, first run only).

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Train all 4 models (downloads dataset automatically on first run):

```bash
python src/train.py
```

Estimated time: 20-60 minutes depending on hardware. GPU optional.

Run the dashboard:

```bash
streamlit run app/app.py
```

For the Gemini LLM section, set a free API key (from [aistudio.google.com](https://aistudio.google.com)):

```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

---

## Project Structure

```
customer-review-intelligence/
├── app/app.py          Streamlit dashboard (5 sections)
├── src/train.py        Training pipeline: 4 models + evaluation artifacts
├── models/             Serialized artifacts (joblib)
├── data/               Dataset cache (auto-downloaded)
├── .env                API keys (never committed)
└── requirements.txt
```

---

## Author

**Santiago Martinez** - Data Analyst

- Portfolio: https://santimuru.github.io
- LinkedIn: https://www.linkedin.com/in/santiago-martinez-pezzatti-4241a3165/
- GitHub: https://github.com/santimuru
