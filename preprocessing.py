def to_lower(text):
    return text.lower()

import re
def remove_punct(text):
    return re.sub(r"[^\w\s]", "",text)

def remove_space(text):
    return re.sub(r"\s+"," ",text).strip()

def remove_url(text):
    return re.sub(r"http\S+|www\S+", "", text)

def remove_sources(text):
    if not isinstance(text,str):
        return""
    return re.split(r"\s[-|–]\s", text)[0]


import nltk
from nltk.corpus import stopwords
STOP_WORDS=set(stopwords.words("english"))

def remove_stopwords(text):
    words=text.split()
    words=[w for w in words if w not in STOP_WORDS]
    return " ".join(words)

def preprocess_text(text):
    if text is None:
        return""
    text=to_lower(text)
    text=remove_punct(text)
    text=remove_space(text)
    text=remove_stopwords(text)
    text=remove_url(text)
    text=remove_sources(text)
    return text