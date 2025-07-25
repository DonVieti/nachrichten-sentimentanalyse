# Stimmungsanalyse deutscher Nachrichtentitel

**Dieses Projekt untersucht die Stimmung deutscher Nachrichtentitel mit einem feinjustierten BERT-Modell ([`deepset/gbert-base`](https://huggingface.co/deepset/gbert-base)) und erweitert die Modellentscheidungen um regelbasiertes Post-Processing (spaCy).**  
Sowohl die sprachliche Emotion als auch die inhaltliche Wirkung von Headlines werden dabei bewertet.

---

## Schnellstart

### Installation
pip install -r requirements.txt
### Gegebenenfalls manuell ausführen
python -m spacy download de_core_news_sm

### 1. Komplett-Pipeline ausführen
python main_pipeline.py

### 2. Manuelles Labeln (optional, Jupyter Notebook erforderlich)
from labeling_ui import prepare_label_file, start_labeling_ui
df = prepare_label_file()
start_labeling_ui(df)
*Hinweis:* Starte dein Jupyter Notebook im Projektverzeichnis, importiere die Funktionen und folge der interaktiven UI zum Labeln.

### 3. Gradio-WebApp starten (optional)
python gradio_app.py

---

## Projektstruktur
```
.
├── data/
│   ├── negative_keywords_news.json    # Lemmatisierte Negativ-Wörter für Regelabgleich
│   ├── headlines_dataset.csv          # Rohdaten mit Schlagzeilen (unbearbeitet)
│   ├── headlines_dataset_cleaned.csv  # Bereinigte, eindeutige Schlagzeilen
│   ├── headlines_labeled.csv          # Manuell gelabelte Schlagzeilen (positiv, neutral, negativ)
├── experiments/                       
│   ├── results_v3_earlystop           # Modell mit Early Stopping (Basis) 
│   ├── results_v3_earlystop_finetuned # Feinjustiertes Early-Stopping-Modell  
│   ├── results_v3_fineval             # Eval-fokussiertes Modell  
│   ├── results_v3_robust              # Robuste Trainingsvariante  
│   ├── results_v4_tuned               # Optimierte Hyperparameter 
│   ├── model_results.csv              # F1,Accuracy,Train & Eval Loss  Ergebnisse
├── final_model/                       
│   ├── final_model_result.csv         # F1,Accuracy,Train & Eval Loss  FinaleErgebnisse
├── pipeline/     
│   ├── collect_and_clean_data.py      # Crawlt RSS-Feeds, bereinigt Titel, entfernt Duplikate
│   ├── labeling_ui.py                 # Interaktive UI zur manuellen Sentiment-Beschriftung
│   ├── data_split.py                  # Aufteilung des Datensatzes in Train/Val/Test
│   ├── prepare_dataset.py             # Tokenisierung, Label-Encoding und Dataset-Erstellung
│   ├── train_model.py                 # Training eines BERT-Modells mit Early Stopping
│   ├── evaluate_model.py              # Evaluiert das Modell mit Klassifikationsmetriken
│   ├── predict_with_rules.py          # Regelbasiertes Post-Processing zur Stimmungskorrektur
│   ├── gradio_app.py                  # Mini-WebApp zur hybriden Sentiment-Analyse
├── results/                           # Trainingsergebnisse
├── saved_best_model/                  # Gespeichertes Trainingsmoddell
│   main.ipynb                         # Interaktive Dokumentation & Analyse
└── main_pipeline.py                   # Zentrale Pipeline zur Ausführung aller Schritte
```
**Hinweis:**  
Die trainierten Modellgewichte in den Ordnern `final_model/` und `saved_best_model/` sowie einige Modelle im Ordner `experiments/` wurden aus Platzgründen nicht mit ins Repository aufgenommen.  
Die Codebasis ist vollständig enthalten, um das Modell mit `train_model.py` neu zu trainieren.     

---

## Features

- **Datensammlung:** Automatisches Crawlen mehrerer RSS-Feeds (Welt.de, Bild.de)  
- **Datenbereinigung:** Entfernen von Duplikaten und unerwünschten Schlagwörtern (z. B. „Horoskop“)  
- **Interaktives Labeling:** Einfache manuelle Annotation via ipywidgets im Notebook  
- **Modelltraining:** Finetuning eines deutschen BERT-Modells (GBERT) mit Early Stopping  
- **Regelbasierte Korrektur:** Nachbearbeitung neutraler Vorhersagen mit negativen Schlüsselworten (spaCy)  
- **Evaluation:** Umfangreiche Metriken und Visualisierungen (Confusion Matrix, Balanced Accuracy)  
- **WebApp:** Einfache Nutzung und Testbarkeit über Gradio-Webinterface  

---

## Beispielausgabe

Titel: Mann tötet Frau auf offener Straße
Sprachebene: neutral
Inhaltsebene: negative
Regel angewendet: True
Begründung: Folgende Wörter deuten auf Negativstimmung: tötet

---

## Evaluationsergebnisse (nach 10 Epochen)

- **Accuracy:** ~67 %  
- **F1-Score (macro):** ~0.67  
- **Balanced Accuracy:** ~0.69  

*Diese Werte zeigen, dass das Modell die Stimmung deutscher Nachrichtentitel gut erfassen kann.*
Weitere experimentelle Modellversionen befinden sich im Ordner experiments/.
---

## Quellen & Technologien

- [`deepset/gbert-base`](https://huggingface.co/deepset/gbert-base)  
- [spaCy de_core_news_sm](https://spacy.io/models/de)  
- Scikit-learn, Transformers, Datasets, Evaluate  
- Gradio, ipywidgets, pandas, numpy  

---

## Datenquellen

- [Welt.de Topnews](https://www.welt.de/feeds/topnews.rss)
- [Welt.de Politik](https://www.welt.de/politik/index.rss)
- [Welt.de Wirtschaft](https://www.welt.de/wirtschaft/index.rss)
- [Bild.de News](https://www.bild.de/feed/alles.xml)

---

## Team

> Nguyen Viet Hoang (589182)
> Alex Lieu (579111)

---

