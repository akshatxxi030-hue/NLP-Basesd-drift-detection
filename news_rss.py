import feedparser
import pandas as pd
import os
from datetime import date

# ===============================
# FILE TO STORE DATA
# ===============================
FILE = "current_news.csv"

# ===============================
# GLOBAL RSS SOURCES (NO INDIA BIAS)
# ===============================
rss_urls = [
    # Global / World
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.cnn.com/rss/edition_world.rss",
    "https://www.reuters.com/rssFeed/worldNews",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://apnews.com/rss",

    # Business / Economy (Global)
    "https://www.reuters.com/rssFeed/businessNews",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.ft.com/rss/home",

    # Technology / AI
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://www.wired.com/feed/rss",

    # Google News (FORCED GLOBAL)
    "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=technology&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=business&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=world&hl=en-US&gl=US&ceid=US:en"
]

# ===============================
# LOAD OLD DATA (IF EXISTS)
# ===============================
if os.path.exists(FILE):
    df_old = pd.read_csv(FILE)
else:
    df_old = pd.DataFrame(columns=["date", "headline", "source"])

rows = []

# ===============================
# FETCH RSS DATA
# ===============================
for url in rss_urls:
    feed = feedparser.parse(url)
    for entry in feed.entries:
        title = entry.title.strip()
        rows.append({
            "date": date.today().isoformat(),
            "headline": title,
            "source": url
        })

df_new = pd.DataFrame(rows)

# ===============================
# COMBINE + CLEAN
# ===============================
df_all = pd.concat([df_old, df_new], ignore_index=True)

# Remove duplicates
df_all.drop_duplicates(subset=["headline"], inplace=True)

# Remove empty headlines
df_all = df_all[df_all["headline"].str.len() > 20]

# ===============================
# SAVE
# ===============================
df_all.to_csv(FILE, index=False)

print("✅ Total global headlines stored:", len(df_all))