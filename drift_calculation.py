import pandas as pd
import numpy as np
import pickle
import re

from preprocessing import preprocess_text
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Load data & artifacts
# =========================

df = pd.read_csv("current_news.csv")

with open("baseline_mean.pkl", "rb") as f:
    mean_baseline = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("X_baseline.pkl", "rb") as f:
    X_baseline = pickle.load(f)

# =========================
# UI preprocessing (optional)
# =========================

def preprocess_ui(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.split(r"\s[-|–]\s", text)[0]
    return text.strip()

def clean_ui(df):
    df["headline"] = df["headline"].apply(preprocess_ui)
    return df

# =========================
# Model preprocessing
# =========================

def clean_model(df):
    df["headline_clean"] = df["headline"].apply(preprocess_text)
    return df

def remove_source(df):
    return df.drop(columns=["source"], errors="ignore")

# =========================
# Drift calculations
# =========================

def current(df):
    X_current = vectorizer.transform(df["headline_clean"])
    current_mean = np.asarray(X_current.mean(axis=0))
    return X_current, current_mean

def drift(df, mean_baseline):
    _, current_mean = current(df)
    similarity = cosine_similarity(current_mean, mean_baseline)[0][0]
    return 1 - similarity

def classify_drift(drift_value: float):
    if drift_value < 0.35:
        return {
            "level": "low",
            "message": "Low drift - News topics are similar to baseline"
        }
    elif drift_value < 0.65:
        return {
            "level": "moderate",
            "message": "Moderate drift - Noticeable topic change detected"
        }
    else:
        return {
            "level": "high",
            "message": "High drift - Major topic shift detected"
        }

def batch_drift(df):
    num_batches = 10
    headlines = df["headline_clean"].tolist()
    batches = np.array_split(headlines, num_batches)

    batch_drifts = []

    for i, batch in enumerate(batches):
        X_batch = vectorizer.transform(batch)
        batch_mean = np.asarray(X_batch.mean(axis=0))
        similarity = cosine_similarity(batch_mean, mean_baseline)[0][0]
        drift_value = 1 - similarity

        batch_drifts.append({
            "batch": i + 1,
            "drift": round(drift_value, 4)
        })

    return batch_drifts

# =========================
# Interpretability
# =========================

def get_top_words(tfidf_matrix, vectorizer, top_n=15):
    words = vectorizer.get_feature_names_out()
    mean_scores = tfidf_matrix.mean(axis=0).A1
    word_scores = list(zip(words, mean_scores))
    word_scores.sort(key=lambda x: x[1], reverse=True)
    return word_scores[:top_n]

# =========================
# Main pipeline
# =========================

def drift_pipeline():
    df = pd.read_csv("current_news.csv")

    df = clean_ui(df)
    df = clean_model(df)
    df = remove_source(df)

    X_current, _ = current(df)

    overall_drift = drift(df, mean_baseline)
    drift_result = classify_drift(overall_drift)

    batch_result = batch_drift(df)

    top_baseline_words = get_top_words(X_baseline, vectorizer, 15)
    top_current_words = get_top_words(X_current, vectorizer, 15)

    return {
        "overall_drift": round(overall_drift, 4),
        "drift_level": drift_result["level"],
        "message": drift_result["message"],
        "batch_drift": batch_result,
        "top_words": {
            "baseline": top_baseline_words,
            "current": top_current_words
        }
    }
