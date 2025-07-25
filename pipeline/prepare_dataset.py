# 4. Dataset vorbereiten und Tokenizen
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import os

def prepare_datasets(
    train_path="data/train_split.csv",
    val_path="data/val_split.csv",
    test_path="data/test_split.csv",
    model_checkpoint="deepset/gbert-base"
):
    # Tokenizer laden
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Tokenisierungsfunktion
    def tokenize_function(example):
        return tokenizer(example["title"], padding="max_length", truncation=True, max_length=64)

    # Lade CSV-Dateien
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Konvertiere in HuggingFace Datasets
    train_ds = Dataset.from_pandas(train_df[["title", "label"]])
    val_ds = Dataset.from_pandas(val_df[["title", "label"]])
    test_ds = Dataset.from_pandas(test_df[["title", "label"]])

    # Tokenisierung anwenden
    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)

    # Datasets speichern 
    os.makedirs("data", exist_ok=True)
    train_ds.save_to_disk("data/train_dataset")
    val_ds.save_to_disk("data/val_dataset")
    test_ds.save_to_disk("data/test_dataset")

    print("Tokenisierte Datensätze gespeichert.")
    return train_ds, val_ds, test_ds, tokenizer

# Optional für Einzelstart
if __name__ == "__main__":
    prepare_datasets()
