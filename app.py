import gradio as gr
import pickle
import re

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

def predict(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    
    return "Fake Review ❌" if pred == 1 else "Real Review ✅"

iface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="Fake Review Detector"
)

iface.launch()
