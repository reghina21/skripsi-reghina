import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Peramalan Kurs", layout="wide")

# CSS untuk gaya pastel dan warna lembut
st.markdown("""
    <style>
    body {
        background-color: #fdf6f0;
    }
    .stApp {
        background-color: #fdf6f0;
    }
    .title {
        color: #6c5ce7;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }
    .subheader {
        color: #00b894;
        font-size: 24px;
        font-weight: bold;
    }
    .block-container {
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Judul Dashboard
st.markdown('<div class="title">🌸 Dashboard Peramalan Kurs Mata Uang 🌸</div>', unsafe_allow_html=True)

# Tabs
tabs = st.tabs(["🏠 Home", "🧹 Preprocessing", "📈 Prediksi Kurs Jual", "📉 Prediksi Kurs Beli"])

# Tab Home
with tabs[0]:
    st.markdown("## Selamat Datang di Dashboard Peramalan Kurs!")
    st.image("https://cdn-icons-png.flaticon.com/512/189/189001.png", width=150)
    st.markdown("""
    Dashboard ini berfungsi untuk melakukan peramalan nilai tukar (kurs) mata uang dengan metode fuzzy.
    
    - Anda dapat mengunggah dataset mata uang pada tab *Preprocessing*
    - Lakukan prediksi untuk *Kurs Jual* dan *Kurs Beli* secara terpisah
    - Warna pastel digunakan untuk memberikan nuansa yang lebih lembut dan ramah pengguna 🌈
    """)

# Tab Preprocessing
with tabs[1]:
    st.markdown("## 🧹 Upload dan Preprocessing Data")
    uploaded_file = st.file_uploader("Unggah file Excel atau CSV berisi data kurs:", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("✅ File berhasil diunggah!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")

# Tab Prediksi Kurs Jual
with tabs[2]:
    st.markdown("## 📈 Prediksi Kurs Jual")
    st.markdown("Fitur ini akan menampilkan hasil prediksi kurs jual menggunakan metode fuzzy.")
    st.info("💡 Fitur prediksi akan aktif setelah data berhasil diunggah.")
    if 'df' in locals():
        if 'Kurs Jual' in df.columns:
            st.line_chart(df['Kurs Jual'])
            st.success("Grafik Kurs Jual berhasil ditampilkan.")
        else:
            st.warning("Kolom 'Kurs Jual' tidak ditemukan dalam data.")

# Tab Prediksi Kurs Beli
with tabs[3]:
    st.markdown("## 📉 Prediksi Kurs Beli")
    st.markdown("Fitur ini akan menampilkan hasil prediksi kurs beli menggunakan metode fuzzy.")
    st.info("💡 Fitur prediksi akan aktif setelah data berhasil diunggah.")
    if 'df' in locals():
        if 'Kurs Beli' in df.columns:
            st.line_chart(df['Kurs Beli'])
            st.success("Grafik Kurs Beli berhasil ditampilkan.")
        else:
            st.warning("Kolom 'Kurs Beli' tidak ditemukan dalam data.")
