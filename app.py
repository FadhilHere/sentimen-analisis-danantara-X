import streamlit as st
import pickle
import numpy as np

# Load model dan vectorizer
with open("model_svm.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

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
        label_dict = {0: "Negatif", 1: "Netral", 2: "Positif"}
        result = label_dict.get(prediction[0], "Tidak diketahui")

        st.subheader("Hasil Prediksi:")
        st.success(f"Komentar ini bersentimen: **{result}**")
