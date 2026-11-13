"""Latih, bandingkan, dan serialisasi model deteksi dini diabetes (GlucoSense).

Dataset: UCI Early Stage Diabetes Risk Prediction (proyek kelompok 3, PL5A1).
Keluaran: model/glucosense_model.joblib dan model/model_metadata.json
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'model'
SEED = 42

GEJALA = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
          'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching',
          'Irritability', 'delayed healing', 'partial paresis',
          'muscle stiffness', 'Alopecia', 'Obesity']


def muat_data():
    df = pd.read_csv(BASE_DIR / 'diabetes_data_upload.csv')
    data = df.drop_duplicates().reset_index(drop=True)
    print(f'Jumlah data awal       : {len(df)}')
    print(f'Setelah hapus duplikat : {len(data)}')
    for kolom in GEJALA:
        data[kolom] = data[kolom].map({'Yes': 1, 'No': 0})
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data['class'] = data['class'].map({'Positive': 1, 'Negative': 0})
    return data.drop(columns='class'), data['class']


def kandidat_model():
    return {
        'Logistic Regression': (
            LogisticRegression(random_state=SEED, max_iter=1000),
            {'model__C': [0.01, 0.1, 1.0, 10.0]}),
        'Random Forest': (
            RandomForestClassifier(random_state=SEED),
            {'model__n_estimators': [50, 100, 200],
             'model__max_depth': [None, 5, 10],
             'model__min_samples_split': [2, 5]}),
        'SVM (RBF)': (
            # SVC(probability=True) sudah deprecated di scikit-learn 1.9,
            # jadi probabilitas dikalibrasi lewat CalibratedClassifierCV.
            CalibratedClassifierCV(SVC(random_state=SEED), ensemble=False),
            {'model__estimator__C': [0.1, 1.0, 10.0],
             'model__estimator__gamma': ['scale', 0.01, 0.1]}),
    }


if __name__ == '__main__':
    X, y = muat_data()
    print(f'Distribusi kelas       : {y.value_counts().to_dict()}')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)
    print(f'Data latih / uji       : {len(X_train)} / {len(X_test)}')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    hasil, terbaik = [], None
    for nama, (estimator, grid) in kandidat_model().items():
        pipa = Pipeline([('scaler', StandardScaler()), ('model', estimator)])
        cari = GridSearchCV(pipa, grid, cv=cv, scoring='f1', n_jobs=-1)
        cari.fit(X_train, y_train)
        pred = cari.predict(X_test)
        proba = cari.predict_proba(X_test)[:, 1]
        hasil.append({
            'Model': nama,
            'CV F1': cari.best_score_,
            'Akurasi': accuracy_score(y_test, pred),
            'Precision': precision_score(y_test, pred),
            'Recall': recall_score(y_test, pred),
            'F1': f1_score(y_test, pred),
            'ROC-AUC': roc_auc_score(y_test, proba),
        })
        if terbaik is None or cari.best_score_ > terbaik[1].best_score_:
            terbaik = (nama, cari)

    print('\nPerbandingan model (CV F1 pada data latih, sisanya pada data uji):')
    print(pd.DataFrame(hasil).round(4).to_string(index=False))

    nama, cari = terbaik
    model = cari.best_estimator_
    pred = model.predict(X_test)
    print(f'\nModel terbaik berdasarkan CV F1: {nama}')
    print(f'Hyperparameter terbaik: {cari.best_params_}')
    print('\nConfusion matrix (baris = aktual, kolom = prediksi):')
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred,
                                target_names=['Negative', 'Positive'],
                                digits=4))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / 'glucosense_model.joblib')
    metadata = {
        'model_name': nama,
        'best_params': cari.best_params_,
        'feature_names': X.columns.tolist(),
        'target_classes': {'0': 'Negative', '1': 'Positive'},
        'n_train': len(X_train),
        'n_test': len(X_test),
        'cv_f1': round(float(cari.best_score_), 4),
        'test_accuracy': round(float(accuracy_score(y_test, pred)), 4),
        'test_f1': round(float(f1_score(y_test, pred)), 4),
        'sklearn_version': sklearn.__version__,
    }
    with open(MODEL_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    ukuran = (MODEL_DIR / 'glucosense_model.joblib').stat().st_size / 1024
    print(f'Model disimpan: model/glucosense_model.joblib ({ukuran:.1f} KB)')
    print('Metadata disimpan: model/model_metadata.json')
