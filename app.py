import streamlit as st
import pickle
import re

# -------------------------------
# Load model and vectorizer
# -------------------------------
@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_models()

# -------------------------------
# Text cleaning function
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

# -------------------------------
# Prediction function
# -------------------------------
def predict_review(text):
    text_clean = clean_text(text)

    if text_clean.strip() == "":
        return "Invalid input ❗"

    text_vec = vectorizer.transform([text_clean])

    prediction = model.predict(text_vec)[0]

    if prediction == 1:
        return "Fake Review ❌"
    else:
        return "Real Review ✅"

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🕵️ Fake Review Detection System")

st.write("Enter a product review below to check if it is fake or real.")

review = st.text_area("✍️ Enter Review")

if st.button("🔍 Check Review"):
    if review.strip() == "":
        st.warning("Please enter a review!")
    else:
        result = predict_review(review)

        if "Fake" in result:
            st.error(result)
        elif "Real" in result:
            st.success(result)
        else:
            st.warning(result)
