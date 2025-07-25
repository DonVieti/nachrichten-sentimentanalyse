def main():
    print("Starte Pipeline...")

    # 1. Daten sammeln & bereinigen
    print("Sammle und bereinige Daten...")
    from pipeline.collect_and_clean_data import collect_and_clean_data
    cleaned_file = collect_and_clean_data()

    # 2. Hinweis: Manuelles Labeln (optional interaktiv), Nur in Jupyter (da Nutzung mit ipywidgets) 
    # Folgende Codes ausführen: 
    # from labeling_ui import prepare_label_file, start_labeling_ui
    # df = prepare_label_file(); start_labeling_ui(df)

    # 3. Daten splitten
    print("Splitte Daten in Trainings-, Validierungs- und Testset...")
    from pipeline.data_split import split_data
    train_df, val_df, test_df, label_names = split_data()

    # 4. Daten tokenisieren
    print("Bereite Datasets vor und tokenisiere Titel...")
    from pipeline.prepare_dataset import prepare_datasets
    train_ds, val_ds, test_ds, tokenizer = prepare_datasets()

    # 5. Modell trainieren
    print("Trainiere Modell...")
    from pipeline.train_model import train_model
    model, tokenizer, label_names, test_ds = train_model()

    # 6. Modell evaluieren
    print("Evaluiere Modell...")
    import pipeline.evaluate_model as evaluate_model
    evaluate_model.run(model, tokenizer, label_names, test_ds)

    # 7. Regeln (Post-Processing testen)
    print("Überprüfe, wie oft Regeln angewendet wurden...")
    from pipeline.predict_with_rules import check_rules_on_dataset
    import pandas as pd
    test_df = pd.read_csv("data/test_split.csv")
    regel_count, total = check_rules_on_dataset(test_df["title"])
    print(f"{regel_count} von {total} durch Regel überschrieben.")

    # 8. Mini Web-App starten (optional)
    start_gradio = input("Mini-Webinterface starten? (j/n): ").lower() == "j"
    if start_gradio:
        import pipeline.gradio_app

    print("Pipeline abgeschlossen.")


if __name__ == "__main__":
    main()
