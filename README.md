# News Headline Drift Detection 📰📊

This project detects *concept drift in news headlines* by comparing recent news with a historical baseline using NLP techniques.

## 🔍 Problem Statement
News topics evolve over time.  
This project identifies whether *current news headlines differ significantly* from older news headlines using *TF-IDF embeddings and cosine similarity*.

## 🛠️ Tech Stack
- Python
- Streamlit
- scikit-learn
- Pandas, NumPy
- Matplotlib

## 🧠 Approach
1. Preprocessed news headlines (lowercase, removed URLs, stopwords, noise,etra spaces,)
2. Converted text into vectors using *TF-IDF*
3. Computed *baseline mean vector* from historical data(2023)
4. Compared recent headlines using *cosine similarity*
5. Detect drift as:  
   *Drift = 1 − cosine similarity*
6. Analyzed drift across batches of headlines

## 📊 Features
- Overall drift score (shown at top)
- Drift trend across headline batches (line chart)
- News feed preview
- Most-used words comparison (baseline vs current)

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
