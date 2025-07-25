# 6. Post Processing

import torch
import json
import spacy
import pandas as pd

# Modell und Tokenizer laden
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("final_model")
tokenizer = AutoTokenizer.from_pretrained("final_model")

# Labelnamen rekonstruieren (muss ggf. angepasst werden)
label_names = ["negative", "neutral", "positive"]

# spaCy-Model laden
nlp = spacy.load("de_core_news_sm")

# Negative Schlüsselwörter laden
with open("negative_keywords_news.json", "r", encoding="utf-8") as f:
    negative_keywords = set(json.load(f))


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=1).item()
    return label_names[pred]


def predict_sentiment_probs(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
    return {label: round(float(p), 3) for label, p in zip(label_names, probs)}


def predict_with_rules(text):
    model_prediction = predict_sentiment(text)
    doc = nlp(text.lower())

    found_negatives = [token.text for token in doc if token.lemma_ in negative_keywords]

    if model_prediction == "neutral" and found_negatives:
        return {
            "final_label": "negative",
            "modell_label": model_prediction,
            "regel_override": True,
            "wahrscheinlichkeiten": predict_sentiment_probs(text),
            "begründung": f"Folgende Wörter deuten auf Negativstimmung: {', '.join(found_negatives)}",
            "text": text
        }

    return {
        "final_label": model_prediction,
        "modell_label": model_prediction,
        "regel_override": False,
        "wahrscheinlichkeiten": predict_sentiment_probs(text),
        "begründung": "Keine kritischen Schlüsselwörter erkannt.",
        "text": text
    }

def check_rules_on_dataset(texts):
    regel_count = 0
    for text in texts:
        result = predict_with_rules(text)
        if result["regel_override"]:
            regel_count += 1
    total = len(texts)
    print(f"Regel angewendet bei {regel_count} von {total} Titeln ({regel_count/total:.1%})")
    return regel_count, total

