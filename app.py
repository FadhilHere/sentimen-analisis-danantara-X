import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load model dan vectorizer
model = joblib.load("model_svm.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Analisis Sentimen Komentar (SVM)")

# Bagian Input Prediksi
st.header("💬 Coba Prediksi Komentar")
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

# Tambahan: Tampilkan Visualisasi EDA
st.markdown("---")
st.header("📊 Visualisasi Data Sentimen")

st.subheader("Distribusi Sentimen Sebelum Preprocessing")
bar_sebelum = Image.open("bar_sebelum.png")
st.image(bar_sebelum, use_container_width=True)

st.subheader("Distribusi Sentimen Sesudah Preprocessing")
bar_sesudah = Image.open("bar_sesudah.png")
st.image(bar_sesudah, use_container_width=True)

st.subheader("Wordcloud Sentimen Positif")
wordcloud_positif = Image.open("wordcloud_positif.png")
st.image(wordcloud_positif, use_container_width=True)

st.subheader("Wordcloud Sentimen Netral")
wordcloud_netral = Image.open("wordcloud_netral.png")
st.image(wordcloud_netral, use_container_width=True)

st.subheader("Wordcloud Sentimen Negatif")
wordcloud_negatif = Image.open("wordcloud_negatif.png")
st.image(wordcloud_negatif, use_container_width=True)
