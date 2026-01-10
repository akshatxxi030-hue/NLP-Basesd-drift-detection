import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle
from preprocessing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


vectorizer=pickle.load(open("vectorizer.pkl","rb"))
mean_baseline=pickle.load(open("baseline_mean.pkl","rb"))

df=pd.read_csv("current_news.csv")

def clean_ui(text):
    text= re.sub(r"http\S+|www\S+", "", text)
    if not isinstance(text,str):
        return""
    text= re.split(r"\s[-|–]\s", text)[0]
    return text.strip()

df["headline"]=df['headline'].apply(clean_ui)
df=df.drop(columns=['source'])

df["headline_preprocessed"]=df["headline"].apply(preprocess_text)
X_current=vectorizer.transform(df["headline_preprocessed"])
current_mean=np.asarray(X_current.mean(axis=0))



st.title("News Drift Detector")
st.write("Get latest news headlines and detect drift from older news headlines.")



overall_similarity = cosine_similarity(current_mean, mean_baseline)[0][0]
overall_drift = 1 - overall_similarity
st.metric(
    "📉 Overall Concept Drift (All Current Headlines)",
    f"{overall_drift:.4f}",
    help="Cosine distance between baseline and current news distributions"
)
if overall_drift > 0.85:
    st.success("🟢 Low drift — news topics are similar to baseline")
elif overall_drift > 0.65:
    st.warning("🟡 Moderate drift — noticeable topic change")
else:
    st.error("🔴 High drift — major topic shift detected")

num_batches= 10
headlines=df["headline_preprocessed"].to_list()
batches=np.array_split(headlines,num_batches)
batch_drifts = []

for i, batch in enumerate(batches):
    X_batch = vectorizer.transform(batch)
    batch_mean = np.asarray(X_batch.mean(axis=0))

    similarity = cosine_similarity(mean_baseline, batch_mean)[0][0]
    drift = 1 - similarity

    batch_drifts.append({
        "Batch": i + 1,
        "Drift": drift
    })

drift_df=pd.DataFrame(batch_drifts)

st.subheader("📊 Drift Trend Across Recent News Batches")
st.caption("Batches are made of 10 headlines out of total data of 524 headlines")

st.line_chart(
    drift_df.set_index("Batch")["Drift"]
)




st.subheader("📰 News Feed")
news=df.head(10)

if st.button("Referesh News"):
    news=df.sample(10)

col1, col2 = st.columns(2)



for i, h in enumerate(news["headline"]):
    if i % 2 == 0:
        col1.markdown(f"🔹 **{h}**")
    else:
        col2.markdown(f"🔹 **{h}**")

st.caption(
    "Headlines shown are sampled data (01/01/26 to 07/01/26), not real-time news."
)


def get_top_words(mean_vector, vectorizer, top_n=15):
    
    
    mean_vector = np.array(mean_vector).flatten()
    
    
    words = vectorizer.get_feature_names_out()
    
   
    sorted_indices = np.argsort(mean_vector)[::-1]
    
   
    top_indices = sorted_indices[:top_n]
    
    # 5. Get top words and their scores
    top_words = words[top_indices]
    top_scores = mean_vector[top_indices]
    
    
    result = pd.DataFrame({
        "word": top_words,
        "score": top_scores
    })
    
    return result

top_baseline=get_top_words(mean_baseline,vectorizer,top_n=15)
top_current=get_top_words(current_mean,vectorizer,top_n=15)


col1, col2 = st.columns(2)

with col1:
    st.subheader(" Most Used Words (Baseline)")
    st.table(top_baseline)

with col2:
    st.subheader(" Most Used Words (Current)")
    st.table(top_current)


overall_similarity = cosine_similarity(current_mean, mean_baseline)[0][0]
overall_drift = 1 - overall_similarity








