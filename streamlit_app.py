import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math

st.set_page_config(page_title="Dashboard Peramalan Kurs", layout="wide")
st.title("📊 Dashboard Peramalan Kurs Yuan & Dollar")

# Navigasi atas dengan tabs
tabs = st.tabs([
    "🏠 Home",
    "📁 Dataset",
    "🧹 Preprocessing",
    "📈 Prediksi Kurs Jual",
    "📉 Prediksi Kurs Beli"
])

# Tab Home
with tabs[0]:
    st.header("Selamat Datang di Dashboard Peramalan Kurs")
    st.markdown("""
    Dashboard ini dirancang untuk membantu Anda melakukan:
    - Upload dan eksplorasi dataset kurs
    - Preprocessing data
    - Visualisasi kurs jual dan beli
    - Prediksi kurs jual dan beli di masa depan

    Silakan mulai dengan mengunggah dataset Anda di tab *📁 Dataset*.
    """)

# Tab 1: Upload Dataset
with tabs[1]:
    st.subheader("Upload Dataset Kurs")
    uploaded_file = st.file_uploader(
        "Upload file Excel dengan kolom: NO, Nilai, Kurs Jual, Kurs Beli, Tanggal",
        type=["xlsx"]
    )

    if uploaded_file:
        try:
            df_raw = pd.read_excel(uploaded_file)
            required_columns = {"NO", "Nilai", "Kurs Jual", "Kurs Beli", "Tanggal"}

            if not required_columns.issubset(df_raw.columns):
                st.error(f"❌ Kolom tidak lengkap! Harus ada kolom: {', '.join(required_columns)}")
            else:
                st.session_state.df_raw = df_raw
                st.session_state.preprocessed = False
                st.success("✅ Dataset berhasil diupload!")
                st.dataframe(df_raw)
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
    elif 'df_raw' in st.session_state:
        st.info("📄 Dataset sebelumnya:")
        st.dataframe(st.session_state.df_raw)

# Tab 2: Preprocessing
with tabs[2]:
    st.subheader("Hasil Preprocessing Data Kurs")

    if 'df_raw' in st.session_state:
        df = st.session_state.df_raw.copy()
        df.drop(columns=["NO", "Nilai"], inplace=True)
        df["Tanggal"] = pd.to_datetime(df["Tanggal"])
        df.sort_values("Tanggal", inplace=True)
        df.set_index("Tanggal", inplace=True)

        df_kurs_jual = df[["Kurs Jual"]].copy()
        df_kurs_beli = df[["Kurs Beli"]].copy()

        st.session_state.df = df
        st.session_state.df_kurs_jual = df_kurs_jual
        st.session_state.df_kurs_beli = df_kurs_beli
        st.session_state.preprocessed = True

        st.success("✅ Data berhasil diproses!")

        st.markdown("### 💹 Kurs Jual")
        st.dataframe(df_kurs_jual.tail())

        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(df_kurs_jual, label="Kurs Jual", color="green")
        ax1.set_title("Visualisasi Kurs Jual")
        ax1.set_xlabel("Tanggal")
        ax1.set_ylabel("Nilai Kurs")
        ax1.legend()
        st.pyplot(fig1)

        st.markdown("### 💰 Kurs Beli")
        st.dataframe(df_kurs_beli.tail())

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(df_kurs_beli, label="Kurs Beli", color="blue")
        ax2.set_title("Visualisasi Kurs Beli")
        ax2.set_xlabel("Tanggal")
        ax2.set_ylabel("Nilai Kurs")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.warning("Mohon upload file terlebih dahulu di tab Dataset.")
