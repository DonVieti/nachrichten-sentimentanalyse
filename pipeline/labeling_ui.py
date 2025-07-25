# 2. Manuelles Labeling mit UI
import pandas as pd
import os

INPUT_FILE = "data/headlines_dataset_cleaned.csv"
OUTPUT_FILE = "data/headlines_labeled.csv"

def prepare_label_file():
    df_base = pd.read_csv(INPUT_FILE)

    if os.path.exists(OUTPUT_FILE):
        print("Bestehende Label-Datei geladen.")
        return pd.read_csv(OUTPUT_FILE)
    else:
        df = df_base.copy()
        df['sentiment'] = None
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print("Neue Label-Datei erstellt.")
        return df

# Ab hier die interaktive Labeling-Funktion (optional für Notebook)
import ipywidgets as widgets
from IPython.display import display

def start_labeling_ui(df, output_file=OUTPUT_FILE):
    df_to_label = df[df['sentiment'].isna()].copy()
    indices = df_to_label.index.tolist()
    current = {'index': 0}

    output = widgets.Output()

    def save_and_continue(label=None):
        idx = indices[current['index']]
        if label in ["positive", "neutral", "negative"]:
            df.at[idx, 'sentiment'] = label
            df.to_csv(output_file, index=False, encoding="utf-8")
            print(f"Gespeichert: {label}")
        current['index'] += 1
        show_next()

    def show_next():
        output.clear_output()
        with output:
            if current['index'] >= len(indices):
                print("Alle Headlines wurden gelabelt!")
                return

            idx = indices[current['index']]
            row = df.loc[idx]

            print(f"\nNoch zu labeln: {len(indices) - current['index']}")
            print(f"Titel: {row['title']}")
            print(f"Datum: {row['date']}")

            buttons = widgets.HBox([
                widgets.Button(description="+ Positiv"),
                widgets.Button(description="# Neutral"),
                widgets.Button(description="- Negativ"),
                widgets.Button(description="Skip"),
                widgets.Button(description="Beenden")
            ])

            def on_click(btn):
                label_map = {
                    "+ Positiv": "positive",
                    "# Neutral": "neutral",
                    "- Negativ": "negative",
                    "Skip": None,
                    "Beenden": "quit"
                }
                label = label_map[btn.description]
                if label == "quit":
                    output.clear_output()
                    print("Beendet.")
                else:
                    save_and_continue(label)

            for b in buttons.children:
                b.on_click(on_click)

            display(buttons)

    show_next()
    display(output)

# Optional direkt testen
if __name__ == "__main__":
    df = prepare_label_file()
    print(df.head())
