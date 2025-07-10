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

# Tab 3: Hasil Prediksi
with tabs[3]:
    st.subheader("📊 Hasil Prediksi Kurs dengan Interval Fuzzy")

    if st.session_state.get('preprocessed', False):
        # Dropdown pemilihan jenis kurs
        tipe_kurs = st.selectbox("Pilih jenis kurs yang ingin diprediksi:", ["Kurs Jual", "Kurs Beli"])

        if tipe_kurs == "Kurs Jual":
            df_kurs = st.session_state.df_kurs_jual.copy()
            kolom_kurs = "Kurs Jual"
        else:
            df_kurs = st.session_state.df_kurs_beli.copy()
            kolom_kurs = "Kurs Beli"

        # Hitung jumlah kelas dengan Aturan Sturges
        n = len(df_kurs)
        K = round(1 + 3.322 * math.log10(n))

        Dmax = df_kurs[kolom_kurs].max()
        Dmin = df_kurs[kolom_kurs].min()
        D1, D2 = 0, 0  # Penyesuaian bawah dan atas

        R = (Dmax + D2) - (Dmin - D1)
        I = R / K

        # Buat interval fuzzy
        intervals = []
        for i in range(K):
            lower = Dmin - D1 + i * I
            upper = lower + I
            intervals.append((lower, upper))

        # Fuzzifikasi
        def fuzzy_label(value):
            for idx, (low, high) in enumerate(intervals):
                if low <= value <= high:
                    return f"A{idx + 1}"
            return None

        df_kurs["Fuzzy Set"] = df_kurs[kolom_kurs].apply(fuzzy_label)

        hasil_list = []

        for i in range(3, len(df_kurs)):
            E_i = df_kurs[kolom_kurs].iloc[i - 1]
            E_i_1 = df_kurs[kolom_kurs].iloc[i - 2]
            E_i_2 = df_kurs[kolom_kurs].iloc[i - 3]

            D_i = abs(abs(E_i - E_i_1) - abs(E_i_1 - E_i_2))

            values_to_check = [
                E_i + D_i / 2, E_i - D_i / 2,
                E_i + D_i, E_i - D_i,
                E_i + D_i / 4, E_i - D_i / 4,
                E_i + 2 * D_i, E_i - 2 * D_i,
                E_i + D_i / 6, E_i - D_i / 6,
                E_i + 3 * D_i, E_i - 3 * D_i,
            ]

            fuzzy_i1 = df_kurs['Fuzzy Set'].iloc[i]
            interval_idx = int(fuzzy_i1[1:]) - 1 if fuzzy_i1 and fuzzy_i1[1:].isdigit() else -1

            if interval_idx < 0 or interval_idx >= len(intervals):
                df_kurs.at[i, 'Prediksi'] = None
                continue

            low, high = intervals[interval_idx]
            mid = (low + high) / 2

            R_sum = sum(val for val in values_to_check if low <= val <= high)
            S = sum(1 for val in values_to_check if low <= val <= high)

            F_j = (R_sum + mid) / (S + 1) if S > 0 else mid
            df_kurs.at[i, 'Prediksi'] = round(F_j, 2)

            hasil_list.append({
                'i': i,
                'Tanggal': df_kurs.index[i],
                'Aktual': df_kurs[kolom_kurs].iloc[i],
                'Prediksi': round(F_j, 2)
            })

        # Tambahkan 3 baris awal tanpa prediksi
        for j in range(3):
            hasil_list.insert(j, {
                'i': j,
                'Tanggal': df_kurs.index[j],
                'Aktual': df_kurs[kolom_kurs].iloc[j],
                'Prediksi': None
            })

        df_hasil_perhitungan = pd.DataFrame(hasil_list)

        st.markdown(f"### 🔍 Tabel Hasil Prediksi {tipe_kurs}")
        st.dataframe(df_hasil_perhitungan[['Tanggal', 'Aktual', 'Prediksi']])

        # Tambahan: Grafik Matplotlib
        st.markdown(f"### 📊 Grafik {tipe_kurs} Aktual vs Prediksi")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_hasil_perhitungan['Tanggal'], df_hasil_perhitungan['Aktual'], label='Aktual', marker='o')
        ax.plot(df_hasil_perhitungan['Tanggal'], df_hasil_perhitungan['Prediksi'], label='Prediksi', marker='x')
        ax.set_xlabel("Tanggal")
        ax.set_ylabel(tipe_kurs)
        ax.set_title(f"Perbandingan {tipe_kurs} vs Prediksi")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.warning("Mohon lakukan preprocessing data terlebih dahulu.")

# Tab 4: Prediksi Masa Depan
with tabs[4]:
    st.subheader("🔮 Prediksi Kurs ke Depan")

    if st.session_state.get('preprocessed', False):
        tipe_kurs = st.selectbox("Pilih jenis kurs yang ingin diramal:", ["Kurs Jual", "Kurs Beli"], key="prediksi_kurs_ke_depan")
        kolom_kurs = "Kurs Jual" if tipe_kurs == "Kurs Jual" else "Kurs Beli"
        df_kurs = st.session_state.df_kurs_jual.copy() if tipe_kurs == "Kurs Jual" else st.session_state.df_kurs_beli.copy()

        # Hitung interval
        n = len(df_kurs)
        K = round(1 + 3.322 * math.log10(n))
        Dmax = df_kurs[kolom_kurs].max()
        Dmin = df_kurs[kolom_kurs].min()
        R = Dmax - Dmin
        I = R / K

        intervals = []
        for i in range(K):
            lower = Dmin - D1 + i * I
            upper = lower + I
            intervals.append((lower, upper))

        # Fungsi fuzzy label
        def fuzzy_label(val, intervals):
            for idx, (low, high) in enumerate(intervals):
                if low <= val <= high:
                    return f"A{idx + 1}"
            return None

        # Fuzzifikasi dan prediksi awal
        df_kurs["Fuzzy Set"] = df_kurs[kolom_kurs].apply(lambda x: fuzzy_label(x, intervals))
        df_kurs["Prediksi"] = None

        for i in range(3, len(df_kurs)):
            E_i = df_kurs[kolom_kurs].iloc[i - 1]
            E_i_1 = df_kurs[kolom_kurs].iloc[i - 2]
            E_i_2 = df_kurs[kolom_kurs].iloc[i - 3]
            D_i = abs(abs(E_i - E_i_1) - abs(E_i_1 - E_i_2))

            values = [
                E_i + D_i / 2, E_i - D_i / 2,
                E_i + D_i, E_i - D_i,
                E_i + D_i / 4, E_i - D_i / 4,
                E_i + 2 * D_i, E_i - 2 * D_i,
                E_i + D_i / 6, E_i - D_i / 6,
                E_i + 3 * D_i, E_i - 3 * D_i,
            ]

            fuzzy = df_kurs["Fuzzy Set"].iloc[i]
            interval_idx = int(fuzzy[1:]) - 1 if isinstance(fuzzy, str) and fuzzy[1:].isdigit() else -1

            if 0 <= interval_idx < len(intervals):
                low, high = intervals[interval_idx]
            else:
                low, high = Dmin, Dmax

            mid = (low + high) / 2
            R_sum = sum(v for v in values if low <= v <= high)
            S = sum(1 for v in values if low <= v <= high)
            pred = round((R_sum + mid) / (S + 1), 2) if S > 0 else round(mid, 2)

            df_kurs.at[df_kurs.index[i], "Prediksi"] = pred

        # Siapkan data awal untuk prediksi ke depan
        df_pred_awal = df_kurs[["Fuzzy Set", "Prediksi"]].dropna().copy().iloc[-3:].copy()

        # Input jumlah hari
        n_forecast = st.number_input("Jumlah hari ke depan:", min_value=1, max_value=30, value=5)

        start_date = pd.to_datetime("2025-01-13")  # sesuai permintaan
        prediksi_ke_depan = []

        for step in range(n_forecast):
            E_i_2 = df_pred_awal['Prediksi'].iloc[-3]
            E_i_1 = df_pred_awal['Prediksi'].iloc[-2]
            E_i = df_pred_awal['Prediksi'].iloc[-1]

            D_i = abs(abs(E_i - E_i_1) - abs(E_i_1 - E_i_2))

            values_to_check = [
                E_i + D_i / 2, E_i - D_i / 2,
                E_i + D_i, E_i - D_i,
                E_i + D_i / 4, E_i - D_i / 4,
                E_i + 2 * D_i, E_i - 2 * D_i,
                E_i + D_i / 6, E_i - D_i / 6,
                E_i + 3 * D_i, E_i - 3 * D_i,
            ]

            fuzzy_i1 = df_pred_awal['Fuzzy Set'].iloc[-1]
            interval_idx = int(fuzzy_i1[1:]) - 1 if isinstance(fuzzy_i1, str) and fuzzy_i1[1:].isdigit() else None

            if 0 <= interval_idx < len(intervals):
                low, high = intervals[interval_idx]
            else:
                low, high = Dmin, Dmax

            mid = (low + high) / 2
            R = sum(val for val in values_to_check if low <= val <= high)
            S = sum(1 for val in values_to_check if low <= val <= high)
            F_j = (R + mid) / (S + 1) if S > 0 else mid
            F_j = round(F_j, 2)

            next_date = start_date + pd.Timedelta(days=step)
            prediksi_ke_depan.append({"Tanggal": next_date, f"Prediksi {tipe_kurs}": F_j})
            fuzzy_new = fuzzy_label(F_j, intervals)

            df_pred_awal = pd.concat([df_pred_awal, pd.DataFrame([{
                "Fuzzy Set": fuzzy_new,
                "Prediksi": F_j
            }])], ignore_index=True).tail(3)

        df_future = pd.DataFrame(prediksi_ke_depan)

        st.markdown(f"### 🔮 Tabel Prediksi {tipe_kurs} {n_forecast} Hari ke Depan")
        st.dataframe(df_future)

        st.markdown(f"### 📈 Grafik Prediksi {tipe_kurs}")
        st.line_chart(df_future.set_index("Tanggal"))

    else:
        st.warning("Mohon lakukan preprocessing data terlebih dahulu.")
