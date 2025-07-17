import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load model dan vectorizer
model = joblib.load("model_svm.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Judul utama
st.title("📘 Analisis Sentimen terhadap Pembentukan Danantara pada Media Sosial X")
st.markdown("### Menggunakan Algoritma Support Vector Machine (SVM)")

# Deskripsi awal
st.markdown("""
Media sosial kini menjadi wadah utama bagi masyarakat dalam menyampaikan opini, termasuk mengenai program atau kebijakan baru seperti pembentukan Danantara. 
Pada analisis ini, kami menggunakan **algoritma Support Vector Machine (SVM)** untuk mengklasifikasikan sentimen masyarakat berdasarkan data yang diperoleh dari media sosial **X (Twitter)**.

Data dikumpulkan menggunakan **Tweet Harvester**, sebuah tools scraping data dari Twitter, yang berhasil menghimpun sebanyak **58.893 baris tweet** dari kata kunci relevan.
""")

# Penjelasan Text Preprocessing
st.markdown("### 🧹 Tahapan Text Preprocessing")
st.markdown("""
Sebelum dilakukan klasifikasi sentimen, data mentah dari Twitter diproses menggunakan beberapa tahapan preprocessing untuk meningkatkan kualitas analisis, yaitu:

1. **Case Folding**: Mengubah seluruh huruf menjadi huruf kecil.
2. **Cleansing**: Menghapus karakter non-alfabet, simbol, dan tanda baca.
3. **Stopword Removal**: Menghapus kata-kata umum yang tidak memiliki makna penting seperti "dan", "yang", "itu".
4. **Tokenizing**: Memecah teks menjadi kata-kata individual.
5. **Stemming**: Mengubah kata ke bentuk dasar menggunakan Sastrawi.

Proses ini membantu dalam mengurangi noise dan memperjelas makna dalam data sebelum dianalisis.
""")

# Visualisasi
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

# Prediksi Sentimen
st.markdown("---")
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
