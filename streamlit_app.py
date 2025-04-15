import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Peramalan Kurs", layout="wide")

st.title("📊 Dashboard Peramalan Kurs Yuan & Dollar")

# Sidebar Navigation
menu = st.sidebar.radio("Pilih Fitur", ["📁 Dataset", "📈 Visualisasi Dataset", "🧠 Model", "🔮 Hasil Prediksi"])

# Global Data Container
if 'df' not in st.session_state:
    st.session_state.df = None
