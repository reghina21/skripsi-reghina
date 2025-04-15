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
    uploaded_file = st.file_uploader("Upload file CSV dengan kolom: tanggal, beli_yuan, jual_yuan, beli_dollar, jual_dollar", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=["tanggal"])
        df.sort_values("tanggal", inplace=True)
        st.session_state.df = df
        st.success("✅ Dataset berhasil diupload!")

    if st.session_state.df is not None:
        st.write("📄 Data Kurs:")
        st.dataframe(st.session_state.df)
