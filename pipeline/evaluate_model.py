# 5. Evaluation

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from transformers import Trainer

def run(model, tokenizer, label_names, test_ds):
    # 1. Vorhersage
    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions = trainer.predict(test_ds)
    pred_labels = predictions.predictions.argmax(axis=1)
    true_labels = predictions.label_ids

    # 2. Bericht
    print("Klassifikationsbericht:")
    print(classification_report(true_labels, pred_labels, target_names=label_names))

    balanced_acc = balanced_accuracy_score(true_labels, pred_labels)
    print(f"Balanced Accuracy: {balanced_acc:.3f}")

    # 3. Konfusionsmatrix
    cm = confusion_matrix(true_labels, pred_labels, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(cm, index=label_names, columns=label_names), annot=True, cmap="Blues", fmt=".2f")
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

    return {
        "balanced_accuracy": balanced_acc,
        "true_labels": true_labels,
        "pred_labels": pred_labels
    }

if __name__ == "__main__":
    print("Bitte führe diesen Code nur über main.py aus.")
