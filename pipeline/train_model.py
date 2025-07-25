# 5. Modelltraining
import random
import numpy as np
import torch
import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_from_disk
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_model(
    model_checkpoint="deepset/gbert-base",
    train_path="data/train_dataset",
    val_path="data/val_dataset",
    test_path="data/test_dataset",
    label_path="label_classes.txt"
):
    set_seed(42)

    # Tokenizer & Datasets laden
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_ds = load_from_disk(train_path)
    val_ds = load_from_disk(val_path)
    test_ds = load_from_disk(test_path)

    # Labels rekonstruieren
    with open(label_path, "r", encoding="utf-8") as f:
        label_names = f.read().splitlines()

    # Modell initialisieren
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label_names)
    )

    # Metriken
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = predictions.argmax(axis=1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"]
        }

    # Trainingsargumente
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=7e-6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.05,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=42
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    trainer.save_model("saved_best_model")
    tokenizer.save_pretrained("saved_best_model")

    return model, tokenizer, label_names, test_ds

if __name__ == "__main__":
    model, tokenizer, label_names, test_ds = train_model()
