import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Peramalan Kurs", layout="wide")
st.title("📊 Dashboard Peramalan Kurs Yuan & Dollar")

# Navigasi atas dengan tabs
tabs = st.tabs([
    "🏠 Home",
    "📁 Dataset",
    "🧹 Preprocessing",
    "📊 Hasil Prediksi",
    "⏭️ Prediksi Masa Depan"
])

# Tab Home
with tabs[0]:
    st.header("Selamat Datang di Dashboard Peramalan Kurs")
    st.markdown("""
    Dashboard ini dirancang untuk membantu Anda melakukan:
    - Upload dan eksplorasi dataset kurs
    - Preprocessing data
    - Visualisasi kurs jual dan beli
    - Prediksi kurs jual di masa depan

    Silakan mulai dengan mengunggah dataset Anda di tab **📁 Dataset**.
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

# Tab 3: Hasil Prediksi
with tabs[3]:
    st.subheader("📊 Hasil Prediksi Kurs Beli dengan Interval Fuzzy")

    if st.session_state.get('preprocessed', False):
        df_kurs_beli = st.session_state.df_kurs_beli.copy()

        # Hitung jumlah kelas dengan Aturan Sturges
        n = len(df_kurs_beli)
        K = 1 + 3.322 * math.log10(n)
        K = round(K)

        Dmax = df_kurs_beli['Kurs Beli'].max()
        Dmin = df_kurs_beli['Kurs Beli'].min()
        D1 = 5  # Nilai penyesuaian bawah
        D2 = 5  # Nilai penyesuaian atas

        R = (Dmax + D2) - (Dmin - D1)
        I = R / K  # Lebar interval

        intervals = []
        midpoints = []
        for i in range(K):
            lower = Dmin - D1 + i * I
            upper = lower + I
            midpoint = (lower + upper) / 2
            intervals.append((lower, upper))
            midpoints.append(midpoint)

        # Tampilkan tabel interval dan midpoint
        df_intervals = pd.DataFrame({
            'Interval': [f"[{low:.2f}, {high:.2f}]" for (low, high) in intervals],
            'Midpoint': [round(mp, 2) for mp in midpoints]
        })

        st.markdown("### 📐 Interval Fuzzy Berdasarkan Aturan Sturges")
        st.dataframe(df_intervals)

        # Fuzzifikasi nilai Kurs Beli
        def fuzzy_label(value):
            for idx, (low, high) in enumerate(intervals):
                if low <= value <= high:
                    return f"A{idx + 1}"
            return None

        df_kurs_beli['Fuzzy Set'] = df_kurs_beli['Kurs Beli'].apply(fuzzy_label)

        hasil_list = []

        for i in range(3, len(df_kurs_beli)):
            E_i = df_kurs_beli['Kurs Beli'].iloc[i - 1]
            E_i_1 = df_kurs_beli['Kurs Beli'].iloc[i - 2]
            E_i_2 = df_kurs_beli['Kurs Beli'].iloc[i - 3]

            D_i = abs(abs(E_i - E_i_1) - abs(E_i_1 - E_i_2))

            values_to_check = [
                E_i + D_i / 2, E_i - D_i / 2,
                E_i + D_i, E_i - D_i,
                E_i + D_i / 4, E_i - D_i / 4,
                E_i + 2 * D_i, E_i - 2 * D_i,
                E_i + D_i / 6, E_i - D_i / 6,
                E_i + 3 * D_i, E_i - 3 * D_i,
            ]

            fuzzy_i1 = df_kurs_beli['Fuzzy Set'].iloc[i]
            interval_idx = int(fuzzy_i1[1:]) - 1 if fuzzy_i1 and fuzzy_i1[1:].isdigit() else -1

            if interval_idx < 0 or interval_idx >= len(intervals):
                df_kurs_beli.at[i, 'Prediksi'] = None
                continue

            low, high = intervals[interval_idx]
            mid = (low + high) / 2

            R_sum = sum(val for val in values_to_check if low <= val <= high)
            S = sum(1 for val in values_to_check if low <= val <= high)

            F_j = (R_sum + mid) / (S + 1) if S > 0 else mid
            df_kurs_beli.at[i, 'Prediksi'] = round(F_j, 2)

            hasil_list.append({
                'i': i,
                'Tanggal': df_kurs_beli.index[i],
                'Aktual': df_kurs_beli['Kurs Beli'].iloc[i],
                'Prediksi': round(F_j, 2)
            })

        # Tambahkan 3 baris awal tanpa prediksi
        for j in range(3):
            hasil_list.insert(j, {
                'i': j,
                'Tanggal': df_kurs_beli.index[j],
                'Aktual': df_kurs_beli['Kurs Beli'].iloc[j],
                'Prediksi': None
            })

        df_hasil_perhitungan_beli = pd.DataFrame(hasil_list)

        st.markdown("### 🔍 Tabel Hasil Prediksi Kurs Beli")
        st.dataframe(df_hasil_perhitungan_beli[['Tanggal', 'Aktual', 'Prediksi']])

        st.markdown("### 📈 Grafik Aktual vs Prediksi Kurs Beli")
        st.line_chart(df_hasil_perhitungan_beli.set_index("Tanggal")[["Aktual", "Prediksi"]])

    else:
        st.warning("Mohon lakukan preprocessing data terlebih dahulu.")


# Tab 4: Prediksi Masa Depan
with tabs[4]:
    st.subheader("🔮 Prediksi Kurs Jual ke Depan")
    if st.session_state.get('preprocessed', False):
        df_kurs_jual = st.session_state.df_kurs_jual

        def fuzzy_label(value):
            return "A1"  # placeholder

        intervals = [(14000, 14500), (14501, 15000), (15001, 15500)]

        n_forecast = st.number_input("Jumlah hari ke depan yang ingin diramal:", 1, 30, 5)

        df_hasil = df_kurs_jual.copy()
        df_hasil.rename(columns={"Kurs Jual": "Prediksi"}, inplace=True)
        df_hasil.reset_index(inplace=True)

        last_known = df_hasil.dropna().copy().tail(3)
        future_preds = []

        start_forecast_date = pd.to_datetime("2025-01-13")

        for i in range(n_forecast):
            E_i = last_known['Prediksi'].iloc[-1]
            E_i_1 = last_known['Prediksi'].iloc[-2]
            E_i_2 = last_known['Prediksi'].iloc[-3]
            D_i = abs(abs(E_i - E_i_1) - abs(E_i_1 - E_i_2))

            values = [
                E_i + D_i / 2, E_i - D_i / 2,
                E_i + D_i, E_i - D_i,
                E_i + D_i / 4, E_i - D_i / 4,
                E_i + 2 * D_i, E_i - 2 * D_i,
                E_i + D_i / 6, E_i - D_i / 6,
                E_i + 3 * D_i, E_i - 3 * D_i,
            ]

            fuzzy = fuzzy_label(E_i)
            interval_idx = int(fuzzy[1:]) - 1 if fuzzy and fuzzy[1:].isdigit() else -1

            if interval_idx < 0 or interval_idx >= len(intervals):
                low, high = min(v[0] for v in intervals), max(v[1] for v in intervals)
            else:
                low, high = intervals[interval_idx]

            mid = (low + high) / 2
            R = sum(v for v in values if low <= v <= high)
            S = sum(1 for v in values if low <= v <= high)
            pred = round((R + mid) / (S + 1), 2) if S > 0 else round(mid, 2)

            next_date = start_forecast_date + pd.Timedelta(days=i)
            future_preds.append({"Tanggal": next_date, "Prediksi Kurs Jual": pred})

            last_known = pd.concat([last_known, pd.DataFrame([{"Tanggal": next_date, "Prediksi": pred}])], ignore_index=True).tail(3)

        df_future = pd.DataFrame(future_preds)
        st.dataframe(df_future)
        st.line_chart(df_future.set_index("Tanggal"))
    else:
        st.warning("Mohon lakukan preprocessing data terlebih dahulu.")
