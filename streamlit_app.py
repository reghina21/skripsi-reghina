import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Peramalan Kurs", layout="wide")
st.title("📊 Dashboard Peramalan Kurs Yuan & Dollar")

# Inisialisasi state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'predicted_df' not in st.session_state:
    st.session_state.predicted_df = None

# Navigasi atas dengan tabs
tabs = st.tabs(["📁 Dataset", "📈 Visualisasi Dataset", "🧠 Model", "🔮 Hasil Prediksi"])

# 1. Dataset Tab
with tabs[0]:
    st.subheader("Upload Dataset Kurs")
    uploaded_file = st.file_uploader(
        "Upload file Excel dengan kolom: NO, Nilai, Kurs Jual, Kurs Beli, Tanggal",
        type=["xlsx"]
    )

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)

            required_columns = {"NO", "Nilai", "Kurs Jual", "Kurs Beli", "Tanggal"}
            if not required_columns.issubset(df.columns):
                st.error(f"❌ Kolom tidak lengkap! Harus ada kolom: {', '.join(required_columns)}")
            else:
                df["Tanggal"] = pd.to_datetime(df["Tanggal"])
                df.sort_values("Tanggal", inplace=True)
                st.session_state.df = df
                st.success("✅ Dataset berhasil diupload!")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat membaca file: {e}")

    if st.session_state.df is not None:
        st.write("📄 Data Kurs:")
        st.dataframe(st.session_state.df)
