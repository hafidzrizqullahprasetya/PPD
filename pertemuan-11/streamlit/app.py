# -*- coding: utf-8 -*-
"""
Aplikasi Web Prediksi Penyakit Hati (HCV) Menggunakan Streamlit
Praktikum Penambangan Data - Pertemuan XI
"""

from typing import Optional
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HCV Tabular Predictor", page_icon="🩺", layout="centered")

st.title("🩺 Hepatitis (HCV) Tabular Predictor")
st.caption("Prediction app using the trained Decision Tree model")

MODEL_PATH = "dt_model.model"

# Nilai default parameter masukan
DEFAULTS = {
    "age": 30,
    "sex": "Male",
    "alb": 39.0,
    "alp": 100.0,
    "alt": 28.0,
    "ast": 143.0,
    "bil": 15.0,
    "che": 5.0,
    "chol": 4.0,
    "crea": 72.0,
    "ggt": 400.0,
    "prot": 70.0,
}

# Inisialisasi session state jika belum ada
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_resource(show_spinner=True)
def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def _to_float(x: Optional[float]) -> float:
    try:
        return float(x) if x is not None else float("nan")
    except Exception:
        return float("nan")


def _encode_sex(x: Optional[str]) -> int:
    if x is None:
        return 0
    s = str(x).strip().lower()
    if s in ("m", "male", "1", "true"):
        return 1
    if s in ("f", "female", "0", "false"):
        return 0
    try:
        return int(float(s))
    except Exception:
        return 0


# Bilah samping (Sidebar)
st.sidebar.title("ℹ️ Panduan & Kontrol")
st.sidebar.markdown(
    """
    **Aplikasi Prediksi HCV**  
    Aplikasi ini dibangun menggunakan framework **Streamlit** untuk menguji model **Decision Tree** 
    pada klasifikasi data klinis Hepatitis C Virus (HCV).

    **Kategori Status:**
    - `0`: Suspect Blood Donor
    - `1`: Hepatitis
    - `2`: Fibrosis
    - `3`: Cirrhosis
    """
)

if st.sidebar.button("🔄 Reset ke Nilai Bawaan"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

# Formulir Input Masukan Laboratorium
with st.form("input-form"):
    st.subheader("Enter Patient Lab Values")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=int(st.session_state["age"]),
            step=1,
            key="age",
        )
        sex = st.selectbox(
            "Sex",
            options=["Male", "Female"],
            index=0 if st.session_state["sex"] == "Male" else 1,
            key="sex",
        )
        alb = st.number_input(
            "ALB (Albumin)",
            min_value=0.0,
            value=float(st.session_state["alb"]),
            step=0.1,
            key="alb",
        )
        alp = st.number_input(
            "ALP (Alkaline Phosphatase)",
            min_value=0.0,
            value=float(st.session_state["alp"]),
            step=0.1,
            key="alp",
        )
        alt = st.number_input(
            "ALT (Alanine Transaminase)",
            min_value=0.0,
            value=float(st.session_state["alt"]),
            step=0.1,
            key="alt",
        )
        ast = st.number_input(
            "AST (Aspartate Transaminase)",
            min_value=0.0,
            value=float(st.session_state["ast"]),
            step=0.1,
            key="ast",
        )

    with col2:
        bil = st.number_input(
            "BIL (Bilirubin)",
            min_value=0.0,
            value=float(st.session_state["bil"]),
            step=0.1,
            key="bil",
        )
        che = st.number_input(
            "CHE (Cholinesterase)",
            min_value=0.0,
            value=float(st.session_state["che"]),
            step=0.1,
            key="che",
        )
        chol = st.number_input(
            "CHOL (Cholesterol)",
            min_value=0.0,
            value=float(st.session_state["chol"]),
            step=0.1,
            key="chol",
        )
        crea = st.number_input(
            "CREA (Creatinine)",
            min_value=0.0,
            value=float(st.session_state["crea"]),
            step=0.1,
            key="crea",
        )
        ggt = st.number_input(
            "GGT (Gamma-Glutamyl Transferase)",
            min_value=0.0,
            value=float(st.session_state["ggt"]),
            step=0.1,
            key="ggt",
        )
        prot = st.number_input(
            "PROT (Total Protein)",
            min_value=0.0,
            value=float(st.session_state["prot"]),
            step=0.1,
            key="prot",
        )

    submitted = st.form_submit_button("Predict")

# Pengolahan Inferensi Model
if submitted:
    with st.spinner("Loading model and predicting..."):
        model = load_model(MODEL_PATH)
        sex_enc = _encode_sex(sex)

        columns_order = [
            "Sex",
            "Age",
            "ALB",
            "ALP",
            "ALT",
            "AST",
            "BIL",
            "CHE",
            "CHOL",
            "CREA",
            "GGT",
            "PROT",
        ]
        df_input = pd.DataFrame(columns=columns_order)
        df_input.loc[0] = [
            sex_enc,
            _to_float(age),
            _to_float(alb),
            _to_float(alp),
            _to_float(alt),
            _to_float(ast),
            _to_float(bil),
            _to_float(che),
            _to_float(chol),
            _to_float(crea),
            _to_float(ggt),
            _to_float(prot),
        ]

        result = model.predict(df_input)
        int_result = int(result[0])
        if int_result == 0:
            decision = "Suspect Blood Donor"
        elif int_result == 1:
            decision = "Hepatitis"
        elif int_result == 2:
            decision = "Fibrosis"
        elif int_result == 3:
            decision = "Cirrhosis"
        else:
            decision = "Not defined"

    st.success("Prediction complete")
    st.metric(label="Predicted Status", value=decision)

    with st.expander("Input Summary"):
        st.write(
            {
                "age": age,
                "sex": sex,
                "alb": alb,
                "alp": alp,
                "alt": alt,
                "ast": ast,
                "bil": bil,
                "che": che,
                "chol": chol,
                "crea": crea,
                "ggt": ggt,
                "prot": prot,
            }
        )
