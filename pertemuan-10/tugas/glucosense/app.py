"""GlucoSense: layanan web Flask untuk skrining dini risiko diabetes."""
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model' / 'glucosense_model.joblib'
METADATA_PATH = BASE_DIR / 'model' / 'model_metadata.json'

# Deserialisasi dilakukan sekali saat server mulai, bukan di setiap request.
model = joblib.load(MODEL_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)
FITUR = metadata['feature_names']

# (nama kolom saat pelatihan, label yang tampil di form)
GEJALA = [
    ('Polyuria', 'Sering buang air kecil (poliuria)'),
    ('Polydipsia', 'Sering merasa haus (polidipsia)'),
    ('sudden weight loss', 'Berat badan turun mendadak'),
    ('weakness', 'Badan terasa lemas'),
    ('Polyphagia', 'Nafsu makan berlebihan (polifagia)'),
    ('Genital thrush', 'Infeksi jamur di area genital'),
    ('visual blurring', 'Penglihatan kabur'),
    ('Itching', 'Gatal-gatal'),
    ('Irritability', 'Mudah marah'),
    ('delayed healing', 'Luka lama sembuh'),
    ('partial paresis', 'Kelemahan otot sebagian (paresis)'),
    ('muscle stiffness', 'Otot terasa kaku'),
    ('Alopecia', 'Rambut rontok (alopesia)'),
    ('Obesity', 'Obesitas'),
]
YA_TIDAK = {'yes': 1, 'ya': 1, '1': 1, 'true': 1,
            'no': 0, 'tidak': 0, '0': 0, 'false': 0}
GENDER = {'male': 1, 'laki-laki': 1, '1': 1,
          'female': 0, 'perempuan': 0, '0': 0}

app = Flask(__name__)


def validasi(data):
    """Ubah masukan mentah menjadi nilai numerik. Kembalikan (nilai, galat)."""
    nilai, galat = {}, {}
    try:
        usia = int(str(data.get('Age', '')).strip())
        if not 1 <= usia <= 120:
            raise ValueError
        nilai['Age'] = usia
    except ValueError:
        galat['Age'] = 'Usia harus berupa bilangan bulat 1 sampai 120.'

    gender = GENDER.get(str(data.get('Gender', '')).strip().lower())
    if gender is None:
        galat['Gender'] = 'Pilih jenis kelamin.'
    else:
        nilai['Gender'] = gender

    for kolom, label in GEJALA:
        jawaban = YA_TIDAK.get(str(data.get(kolom, '')).strip().lower())
        if jawaban is None:
            galat[kolom] = f'Pilih Ya atau Tidak untuk "{label}".'
        else:
            nilai[kolom] = jawaban
    return nilai, galat


def prediksi(nilai):
    """Jalankan inferensi untuk satu pasien."""
    # Urutan kolom diambil dari metadata agar sama dengan saat pelatihan.
    df_input = pd.DataFrame([nilai], columns=FITUR)
    kelas = int(model.predict(df_input)[0])
    peluang = float(model.predict_proba(df_input)[0][1])
    return {
        'kelas': kelas,
        'label': metadata['target_classes'][str(kelas)],
        'peluang_positif': round(peluang, 4),
    }


@app.route('/')
def index():
    return render_template('index.html', gejala=GEJALA, isian={}, galat={},
                           meta=metadata)


@app.route('/predict', methods=['POST'])
def predict():
    nilai, galat = validasi(request.form)
    if galat:
        return render_template('index.html', gejala=GEJALA,
                               isian=request.form, galat=galat,
                               meta=metadata), 400
    hasil = prediksi(nilai)
    dialami = [label for kolom, label in GEJALA if nilai[kolom] == 1]
    return render_template('result.html', hasil=hasil, nilai=nilai,
                           dialami=dialami, jumlah_gejala=len(GEJALA),
                           meta=metadata)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error='Body request harus berupa objek JSON.'), 400
    nilai, galat = validasi(data)
    if galat:
        return jsonify(error='Input tidak valid.', detail=galat), 400
    return jsonify(model=metadata['model_name'], input=nilai,
                   **prediksi(nilai))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)
