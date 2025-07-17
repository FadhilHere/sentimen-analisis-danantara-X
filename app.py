import streamlit as st
import joblib
import numpy as np

# Load model dan vectorizer

# Load model dan vectorizer
model = joblib.load("model_svm.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Analisis Sentimen Komentar (SVM)")

# Input dari user
text_input = st.text_area("Masukkan komentar:", "")

if st.button("Prediksi Sentimen"):
    if text_input.strip() == "":
        st.warning("Komentar tidak boleh kosong!")
    else:
        # Transformasi input
        text_vectorized = vectorizer.transform([text_input])
        prediction = model.predict(text_vectorized)

        # Label sentimen
        result = prediction[0].capitalize()

        st.subheader("Hasil Prediksi:")
        st.success(f"Komentar ini bersentimen: **{result}**")
