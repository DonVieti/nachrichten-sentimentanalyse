# 7. Mini Web-UI
import gradio as gr
from predict_with_rules import predict_with_rules

def hybrid_sentiment(text):
    if not text.strip():
        return (
            "Kein Text eingegeben.",
            "-", "-", "-", "-"
        )
    result = predict_with_rules(text)
    return (
        result["final_label"],
        result["modell_label"],
        str(result["regel_override"]),
        result["begrundung"],
        str(result["wahrscheinlichkeiten"])
    )

demo = gr.Interface(
    fn=hybrid_sentiment,
    inputs=gr.Textbox(label="Nachrichtentitel", placeholder="z. B. 'Mann tötet Frau in Berlin'"),
    outputs=[
        gr.Textbox(label="Inhaltsebene (Regelbasiert)"),
        gr.Textbox(label="Sprachebene (Modell-Vorhersage)"),
        gr.Textbox(label="Regel angewendet"),
        gr.Textbox(label="Begründung"),
        gr.Textbox(label="Modell-Wahrscheinlichkeiten")
    ],
    title="Stimmungsanalyse deutscher Nachrichtentitel",
    description="Dieses Tool kombiniert ein maschinell gelerntes Modell mit regelbasierten Filtern zur Analyse der Stimmung in Nachrichtentiteln."
)

if __name__ == "__main__":
    demo.launch()