# -*- coding: utf-8 -*-
"""
GlucoSense: Dashboard Interaktif Deteksi Dini Risiko Diabetes Melitus Tipe 2
Mata Kuliah Praktikum Penambangan Data - Pertemuan XI (Streamlit)
Kelompok 3 - TRPL SV UGM
"""

import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Pengaturan halaman Streamlit
st.set_page_config(
    page_title="GlucoSense - Deteksi Dini Diabetes Melitus",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
LOGO_UGM = BASE_DIR.parent.parent / "images" / "UGM.png"
MODEL_PATH = BASE_DIR / "model" / "glucosense_model.joblib"
METADATA_PATH = BASE_DIR / "model" / "model_metadata.json"
DATASET_PATH = BASE_DIR / "diabetes_data_upload.csv"

# Daftar gejala klinis terstruktur (nama kolom, label bahasa Indonesia, deskripsi medis singkat)
GEJALA_INFO = [
    ("Polyuria", "Sering Buang Air Kecil (Poliuria)", "Produksi urine berlebih akibat ginjal menyaring kelebihan glukosa darah"),
    ("Polydipsia", "Sering Merasa Haus (Polidipsia)", "Rasa haus berlebihan akibat dehidrasi akibat sering buang air kecil"),
    ("sudden weight loss", "Penurunan Berat Badan Mendadak", "Penurunan berat badan tanpa diet akibat sel kekurangan energi glukosa"),
    ("weakness", "Kelemahan Fisik & Kelelahan", "Tubuh terasa sangat lesu karena glukosa tidak dapat masuk ke sel tubuh"),
    ("Polyphagia", "Nafsu Makan Berlebih (Polifagia)", "Rasa lapar berlebih karena sel tubuh kelaparan energi"),
    ("Genital thrush", "Infeksi Jamur Area Genital", "Pertumbuhan jamur Candida yang dipicu kadar gula tinggi pada urine"),
    ("visual blurring", "Penglihatan Kabur", "Perubahan bentuk lensa mata akibat fluktuasi cairan dan kadar gula darah"),
    ("Itching", "Gatal-Gatal pada Kulit", "Pruritus akibat sirkulasi darah yang buruk dan kulit kering"),
    ("Irritability", "Perubahan Emosi / Lekas Marah", "Sensitivitas emosi meningkat akibat fluktuasi tajam kadar gula darah"),
    ("delayed healing", "Penyembuhan Luka Lambat", "Luka sulit sembuh akibat gangguan vaskular dan penekanan sistem imun"),
    ("partial paresis", "Kelemahan Otot Sebagian (Paresis)", "Kelemahan anggota gerak akibat komplikasi neuropati diabetik"),
    ("muscle stiffness", "Kekakuan Otot", "Kram atau otot kaku akibat gangguan sirkulasi elektrolit dan saraf"),
    ("Alopecia", "Kerontokan Rambut (Alopesia)", "Rambut rontok abnormal dipicu gangguan vaskular folikel rambut"),
    ("Obesity", "Kondisi Obesitas", "Indeks Massa Tubuh (IMT) tinggi yang meningkatkan resistensi hormon insulin"),
]


@st.cache_resource(show_spinner=True)
def muat_model():
    """Memuat model serialisasi SVM dan berkas metadata konfigurasi."""
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


@st.cache_data
def muat_dataset():
    """Memuat dataset PPD untuk visualisasi eksploratif."""
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    return None


model, metadata = muat_model()
df_sample = muat_dataset()
FITUR = metadata["feature_names"]

# Nilai awal / Preset
PRESET_PASIEN = {
    "Kosong": {
        "Age": 35,
        "Gender": "Male",
        "Polyuria": "Tidak",
        "Polydipsia": "Tidak",
        "sudden weight loss": "Tidak",
        "weakness": "Tidak",
        "Polyphagia": "Tidak",
        "Genital thrush": "Tidak",
        "visual blurring": "Tidak",
        "Itching": "Tidak",
        "Irritability": "Tidak",
        "delayed healing": "Tidak",
        "partial paresis": "Tidak",
        "muscle stiffness": "Tidak",
        "Alopecia": "Tidak",
        "Obesity": "Tidak",
    },
    "Risiko Tinggi (Positif)": {
        "Age": 55,
        "Gender": "Male",
        "Polyuria": "Ya",
        "Polydipsia": "Ya",
        "sudden weight loss": "Ya",
        "weakness": "Ya",
        "Polyphagia": "Ya",
        "Genital thrush": "Tidak",
        "visual blurring": "Ya",
        "Itching": "Ya",
        "Irritability": "Ya",
        "delayed healing": "Ya",
        "partial paresis": "Ya",
        "muscle stiffness": "Tidak",
        "Alopecia": "Ya",
        "Obesity": "Ya",
    },
    "Risiko Rendah (Negatif)": {
        "Age": 28,
        "Gender": "Female",
        "Polyuria": "Tidak",
        "Polydipsia": "Tidak",
        "sudden weight loss": "Tidak",
        "weakness": "Tidak",
        "Polyphagia": "Tidak",
        "Genital thrush": "Tidak",
        "visual blurring": "Tidak",
        "Itching": "Tidak",
        "Irritability": "Tidak",
        "delayed healing": "Tidak",
        "partial paresis": "Tidak",
        "muscle stiffness": "Tidak",
        "Alopecia": "Tidak",
        "Obesity": "Tidak",
    },
}

# Inisialisasi Session State
if "form_values" not in st.session_state:
    st.session_state["form_values"] = dict(PRESET_PASIEN["Kosong"])

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    if LOGO_UGM.exists():
        st.image(str(LOGO_UGM), width=100)
    st.title("GlucoSense")
    st.markdown(
        """
        **Sistem Deteksi Dini Risiko Diabetes Melitus Tipe 2**  
        *Praktikum Penambangan Data - Pertemuan XI*  
        **Teknologi Rekayasa Perangkat Lunak**  
        Sekolah Vokasi Universitas Gadjah Mada  
        """
    )
    st.divider()

    st.subheader("⚙️ Preset Data Pasien")
    preset_pilihan = st.selectbox(
        "Muat Contoh Kasus Uji Cepat:",
        options=list(PRESET_PASIEN.keys()),
        index=0,
    )

    if st.button("Terapkan Preset Terpilih", use_container_width=True):
        st.session_state["form_values"] = dict(PRESET_PASIEN[preset_pilihan])
        st.rerun()

    if st.button("🔄 Reset Formulir", use_container_width=True):
        st.session_state["form_values"] = dict(PRESET_PASIEN["Kosong"])
        st.rerun()

    st.divider()
    st.markdown(
        f"""
        **Informasi Model:**
        - **Algoritma:** `{metadata.get('model_name', 'SVM RBF')}`
        - **Akurasi Uji:** `{metadata.get('test_accuracy', 0.8684) * 100:.1f}%`
        - **F1-Score Uji:** `{metadata.get('test_f1', 0.90) * 100:.1f}%`
        - **Sklearn:** `{metadata.get('sklearn_version', '1.9.1')}`
        """
    )


# ==============================================================================
# MAIN PAGE HEADER
# ==============================================================================
st.title("🩸 GlucoSense: Deteksi Dini Risiko Diabetes Melitus")
st.markdown(
    """
    Aplikasi *data deployment* interaktif berbasis **Streamlit** untuk melakukan inferensi prediksi 
    risiko Diabetes Melitus Tipe 2 menggunakan model *Support Vector Machine (SVM)* terkalibrasi secara *real-time*.
    """
)

tab1, tab2, tab3 = st.tabs([
    "🩺 Formulir Skrining Klinis",
    "📊 Metrik Model & Eksplorasi Data",
    "📖 Panduan Gejala & Edukasi",
])

# ==============================================================================
# TAB 1: FORMULIR SKRINING KLINIS
# ==============================================================================
with tab1:
    st.subheader("📝 Masukkan Data Demografis dan Gejala Pasien")
    st.caption("Silakan isi seluruh parameter di bawah ini secara cermat.")

    with st.form("form_skrining"):
        col1, col2, col3 = st.columns(3)

        # Kolom 1: Data Demografis & Gejala Trias Utama
        with col1:
            st.markdown("#### 👤 Data Demografis")
            usia = st.number_input(
                "Usia Pasien (Tahun):",
                min_value=1,
                max_value=120,
                value=int(st.session_state["form_values"].get("Age", 35)),
                step=1,
            )
            gender_opt = ["Male", "Female"]
            gender_default_idx = (
                0 if st.session_state["form_values"].get("Gender", "Male") == "Male" else 1
            )
            gender = st.selectbox(
                "Jenis Kelamin:",
                options=gender_opt,
                index=gender_default_idx,
            )

            st.markdown("#### 🚽 Gejala Osmotik (Trias)")
            p_polyuria = st.radio(
                "1. Sering Buang Air Kecil (Poliuria)?",
                options=["Ya", "Tidak"],
                index=0 if st.session_state["form_values"].get("Polyuria") == "Ya" else 1,
                horizontal=True,
            )
            p_polydipsia = st.radio(
                "2. Sering Merasa Haus (Polidipsia)?",
                options=["Ya", "Tidak"],
                index=0 if st.session_state["form_values"].get("Polydipsia") == "Ya" else 1,
                horizontal=True,
            )
            p_polyphagia = st.radio(
                "3. Nafsu Makan Berlebih (Polifagia)?",
                options=["Ya", "Tidak"],
                index=0 if st.session_state["form_values"].get("Polyphagia") == "Ya" else 1,
                horizontal=True,
            )

        # Kolom 2: Gejala Sistemik & Metabolisme
        with col2:
            st.markdown("#### ⚡ Gejala Sistemik & Energi")
            p_weight = st.radio(
                "4. Penurunan Berat Badan Mendadak?",
                options=["Ya", "Tidak"],
                index=0 if st.session_state["form_values"].get("sudden weight loss") == "Ya" else 1,
                horizontal=True,
            )
            p_weakness = st.radio(
                "5. Tubuh Terasa Sangat Lemas / Lelah?",
                options=["Ya", "Tidak"],
                index=0 if st.session_state["form_values"].get("weakness") == "Ya" else 1,
                horizontal=True,
            )
            p_blur = st.radio(
                "6. Penglihatan Kabur?",
                options=["Ya", "Tidak"],
                index=0 if st.session_state["form_values"].get("visual blurring") == "Ya" else 1,
                horizontal=True,
            )
            p_itching = st.radio(
                "7. Gatal-Gatal pada Kulit (Pruritus)?",
                options=["Ya", "Tidak"],
                index=0 if st.session_state["form_values"].get("Itching") == "Ya" else 1,
                horizontal=True,
            )
            p_irritability = st.radio(
                "8. Mudah Marah / Emosi Tidak Stabil?",
                options=["Ya", "Tidak"],
                index=0 if st.session_state["form_values"].get("Irritability") == "Ya" else 1,
                horizontal=True,
            )

        # Kolom 3: Komplikasi, Neuropati, dan Fisik
        with col3:
            st.markdown("#### 🩺 Kondisi Fisik & Komplikasi")
            p_healing = st.radio(
                "9. Luka Sulit / Lama Sembuh?",
                options=["Ya", "Tidak"],
                index=0 if st.session_state["form_values"].get("delayed healing") == "Ya" else 1,
                horizontal=True,
            )
            p_paresis = st.radio(
                "10. Kelemahan Sebagian Otot (Paresis)?",
                options=["Ya", "Tidak"],
                index=0 if st.session_state["form_values"].get("partial paresis") == "Ya" else 1,
                horizontal=True,
            )
            p_stiffness = st.radio(
                "11. Otot Terasa Kaku / Kram?",
                options=["Ya", "Tidak"],
                index=0 if st.session_state["form_values"].get("muscle stiffness") == "Ya" else 1,
                horizontal=True,
            )
            p_thrush = st.radio(
                "12. Infeksi Jamur Area Genital?",
                options=["Ya", "Tidak"],
                index=0 if st.session_state["form_values"].get("Genital thrush") == "Ya" else 1,
                horizontal=True,
            )
            p_alopecia = st.radio(
                "13. Kerontokan Rambut Abnormal?",
                options=["Ya", "Tidak"],
                index=0 if st.session_state["form_values"].get("Alopecia") == "Ya" else 1,
                horizontal=True,
            )
            p_obesity = st.radio(
                "14. Mengalami Obesitas (IMT Tinggi)?",
                options=["Ya", "Tidak"],
                index=0 if st.session_state["form_values"].get("Obesity") == "Ya" else 1,
                horizontal=True,
            )

        submit_btn = st.form_submit_button(
            "🔬 Jalankan Analisis Prediksi Risiko",
            use_container_width=True,
            type="primary",
        )

    # Proses Inferensi Model Saat Tombol Ditekan
    if submit_btn:
        with st.spinner("Mengolah fitur klinis dan mengeksekusi inferensi model SVM..."):
            # Enkoding fitur masukan sesuai urutan pelatihan model
            input_dict = {
                "Age": int(usia),
                "Gender": 1 if gender == "Male" else 0,
                "Polyuria": 1 if p_polyuria == "Ya" else 0,
                "Polydipsia": 1 if p_polydipsia == "Ya" else 0,
                "sudden weight loss": 1 if p_weight == "Ya" else 0,
                "weakness": 1 if p_weakness == "Ya" else 0,
                "Polyphagia": 1 if p_polyphagia == "Ya" else 0,
                "Genital thrush": 1 if p_thrush == "Ya" else 0,
                "visual blurring": 1 if p_blur == "Ya" else 0,
                "Itching": 1 if p_itching == "Ya" else 0,
                "Irritability": 1 if p_irritability == "Ya" else 0,
                "delayed healing": 1 if p_healing == "Ya" else 0,
                "partial paresis": 1 if p_paresis == "Ya" else 0,
                "muscle stiffness": 1 if p_stiffness == "Ya" else 0,
                "Alopecia": 1 if p_alopecia == "Ya" else 0,
                "Obesity": 1 if p_obesity == "Ya" else 0,
            }

            df_inference = pd.DataFrame([input_dict], columns=FITUR)
            pred_class = int(model.predict(df_inference)[0])
            pred_prob = float(model.predict_proba(df_inference)[0][1])

            st.divider()
            st.subheader("🎯 Hasil Evaluasi Risiko Klinis")

            m_col1, m_col2, m_col3 = st.columns([1.5, 1, 1])

            with m_col1:
                if pred_class == 1:
                    st.error("### ⚠️ TERDETEKSI RISIKO TINGGI (POSITIF)")
                    st.markdown(
                        """
                        Hasil inferensi model mengindikasikan **probabilitas signifikan** 
                        terhadap gejala awal Diabetes Melitus Tipe 2. Pasien disarankan 
                        segera melakukan pemeriksaan laboratorium konfirmasi.
                        """
                    )
                else:
                    st.success("### ✅ TERDETEKSI RISIKO RENDAH (NEGATIF)")
                    st.markdown(
                        """
                        Hasil inferensi model menunjukkan pola parameter klinis masih 
                        berada pada **kategori risiko rendah**. Pertahankan pola makan sehat 
                        dan aktivitas fisik secara teratur.
                        """
                    )

            with m_col2:
                prob_persen = pred_prob * 100
                st.metric(
                    label="Peluang Positif Diabetes",
                    value=f"{prob_persen:.2f}%",
                    delta=f"{'+' if pred_prob >= 0.5 else '-'}{abs(prob_persen - 50):.1f}% dari ambang 50%",
                    delta_color="inverse",
                )
                st.progress(pred_prob)

            with m_col3:
                gejala_aktif = [
                    label for col, label, _ in GEJALA_INFO if input_dict.get(col) == 1
                ]
                st.metric(
                    label="Total Gejala Aktif",
                    value=f"{len(gejala_aktif)} / {len(GEJALA_INFO)}",
                )

            # Rincian Gejala yang Dialami
            with st.expander("🔍 Lihat Rincian Fitur & Rekomendasi Medis", expanded=True):
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    st.markdown("**Daftar Gejala Positif yang Dialami:**")
                    if gejala_aktif:
                        for g in gejala_aktif:
                            st.write(f"- 🔴 {g}")
                    else:
                        st.info("Tidak ada gejala diabetes klasik yang dilaporkan.")

                with r_col2:
                    st.markdown("**Rekomendasi Tindak Lanjut Medis:**")
                    if pred_class == 1:
                        st.markdown(
                            """
                            1. **Pemeriksaan Laboratorium:** Lakukan uji HbA1c, Gula Darah Puasa (GDP), dan Gula Darah 2 Jam Post-Prandial (GD2PP).
                            2. **Konsultasi Dokter:** Kunjungi dokter spesialis penyakit dalam atau fasilitas kesehatan terdekat.
                            3. **Modifikasi Gaya Hidup:** Kurangi konsumsi karbohidrat sederhana dan perbanyak hidrasi air putih.
                            """
                        )
                    else:
                        st.markdown(
                            """
                            1. **Skrining Berkala:** Lakukan pemeriksaan glukosa darah berkala minimal setahun sekali.
                            2. **Pola Makan Gizi Seimbang:** Perbanyak serat dan batasi makanan cepat saji serta minuman manis.
                            3. **Aktivitas Fisik:** Luangkan olahraga aerobik intensitas sedang minimal 150 menit per minggu.
                            """
                        )

# ==============================================================================
# TAB 2: METRIK MODEL & EKSPLORASI DATA
# ==============================================================================
with tab2:
    st.subheader("📊 Karakteristik Model dan Data Latih")
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    col_kpi1.metric("Model Terpilih", metadata.get("model_name", "SVM (RBF)"))
    col_kpi2.metric("Cross-Validation F1", f"{metadata.get('cv_f1', 0.9673)*100:.2f}%")
    col_kpi3.metric("Test Accuracy", f"{metadata.get('test_accuracy', 0.8684)*100:.2f}%")
    col_kpi4.metric("Test F1-Score", f"{metadata.get('test_f1', 0.9000)*100:.2f}%")

    st.markdown("---")
    st.markdown("#### 📂 Ringkasan Parameter Terbaik (GridSearchCV)")
    st.json(metadata.get("best_params", {}))

    if df_sample is not None:
        st.markdown("#### 📑 Cuplikan Dataset Deteksi Dini Diabetes (520 Entri)")
        st.dataframe(df_sample.head(10), use_container_width=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Distribusi Kelas Target (Positif vs Negatif):**")
            class_counts = df_sample["class"].value_counts()
            st.bar_chart(class_counts)
        with col_c2:
            st.markdown("**Distribusi Usia Berdasarkan Kelas Target:**")
            st.area_chart(df_sample.groupby(["Age", "class"]).size().unstack(fill_value=0))

# ==============================================================================
# TAB 3: PANDUAN GEJALA & EDUKASI
# ==============================================================================
with tab3:
    st.subheader("📖 Ensiklopedia Gejala Dini Diabetes Melitus")
    st.markdown(
        """
        Diabetes Melitus Tipe 2 sering kali berkembang secara bertahap tanpa gejala yang mencolok 
        pada stadium awal. Memahami gejala klinis sejak dini sangat krusial untuk mencegah 
        komplikasi vaskular dan neuropati kronis.
        """
    )

    for nama, label, desc in GEJALA_INFO:
        with st.expander(f"📌 {label}"):
            st.write(f"**Identifikasi Klinis:** {nama}")
            st.write(f"**Mekanisme Patofisiologis:** {desc}")
