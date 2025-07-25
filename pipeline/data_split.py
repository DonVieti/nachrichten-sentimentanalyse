# 3. Daten splitten
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def split_data(input_file="data/headlines_labeled.csv"):
    # Lade gelabelte Daten
    df = pd.read_csv(input_file)
    df = df.dropna(subset=["sentiment"])

    # Schritt 1: 80 % Training, 20 % Rest
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["sentiment"],
        random_state=42
    )

    # Schritt 2: Rest → 10 % Validation, 10 % Test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["sentiment"],
        random_state=42
    )

    # Label-Encoding
    label_encoder = LabelEncoder()
    train_df["label"] = label_encoder.fit_transform(train_df["sentiment"])
    val_df["label"] = label_encoder.transform(val_df["sentiment"])
    test_df["label"] = label_encoder.transform(test_df["sentiment"])

    label_names = label_encoder.classes_.tolist()

    # Speicherordner anlegen
    os.makedirs("data", exist_ok=True)

    # Speichern
    train_df.to_csv("data/train_split.csv", index=False, encoding="utf-8")
    val_df.to_csv("data/val_split.csv", index=False, encoding="utf-8")
    test_df.to_csv("data/test_split.csv", index=False, encoding="utf-8")

    with open("data/label_classes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(label_names))

    print("Daten wurden gesplittet und gespeichert:")
    print(f"Train: {len(train_df)}\tValidation: {len(val_df)}\tTest: {len(test_df)}")

    return train_df, val_df, test_df, label_names

# Optionaler Direktstart für Standalone-Nutzung
if __name__ == "__main__":
    split_data()