# 1. Daten crawlen
import feedparser
import pandas as pd
from datetime import datetime
import uuid
import os
import re

def clean_title(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def collect_and_clean_data():
    os.makedirs("data", exist_ok=True)
    feeds = {
        "welt_top": "https://www.welt.de/feeds/topnews.rss",
        "welt_latest": "https://www.welt.de/feeds/latest.rss",
        "welt_politik": "https://www.welt.de/politik/index.rss",
        "welt_wirtschaft": "https://www.welt.de/wirtschaft/index.rss",
        "welt_meinung": "https://www.welt.de/debatte/index.rss",
        "bild": "https://www.bild.de/feed/alles.xml"
    }

    rows = []
    for medium, url in feeds.items():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            row = {
                "id": str(uuid.uuid4()),
                "medium": medium,
                "title": entry.title.strip(),
                "date": entry.get("published", datetime.now().isoformat())
            }
            rows.append(row)

    df_new = pd.DataFrame(rows)

    # Datei anlegen oder erweitern
    filename = "data/headlines_dataset.csv"
    if os.path.exists(filename):
        df_old = pd.read_csv(filename)
        df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset=["medium", "title"])
    else:
        df_combined = df_new

    df_combined.to_csv(filename, index=False)
    print(f"Aktuell gespeicherte Rohdaten-Headlines: {len(df_combined)}")

    # Bereinigen, Duplikate, doppelte Leerzeichen und "Horoskop" entfernen
    df = df_combined.copy()
    df = df[~df['title'].str.contains("horoskop", case=False, na=False)]
    df['title_clean'] = df['title'].apply(clean_title)
    df_deduplicated = df.drop_duplicates(subset=["title_clean"])
    df_deduplicated = df_deduplicated.drop(columns=["title_clean"])

    cleaned_filename = "data/headlines_dataset_cleaned.csv"
    df_deduplicated.to_csv(cleaned_filename, index=False)
    print(f"Bereinigte Headlines gespeichert: {len(df_deduplicated)} eindeutige Titel")

    return cleaned_filename

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    collect_and_clean_data()

