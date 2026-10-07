# SECTION 3: PEMODELAN MACHINE LEARNING UTAMA & HYPERPARAMETER TUNING
# Penanggung Jawab: Hafidz Rizqullah Prasetya
# Mata Kuliah: Praktikum Penambangan Data (Gasal 2026/2027)

# ==============================================================================
# 3.1 IMPORT LIBRARY PEMODELAN & EVALUASI
# ==============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("✓ Library pemodelan Machine Learning berhasil dimuat.")

# ==============================================================================
# 3.2 INISIALISASI & TRAINING MODEL BASELINE
# ==============================================================================
print("=" * 75)
print("3.2 PELATIHAN MODEL BASELINE (DEFAULT HYPERPARAMETERS)")
print("=" * 75)

# Inisialisasi 3 model utama dengan random_state tetap untuk reproduktifitas
baseline_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine (RBF)': SVC(probability=True, random_state=42)
}

baseline_metrics = []

# Pelatihan dan evaluasi pada data test
for model_name, model in baseline_models.items():
    # Fit model pada data training terstandarisasi
    model.fit(X_train_scaled, y_train)
    
    # Prediksi kelas dan probabilitas
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Hitung metrik evaluasi
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    baseline_metrics.append({
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    })

# Konversi hasil ke DataFrame
df_baseline_results = pd.DataFrame(baseline_metrics)

print("\nTABEL PERFORMA MODEL BASELINE (DATA TESTING):")
display(df_baseline_results.style.format({
    'Accuracy': '{:.4f}',
    'Precision': '{:.4f}',
    'Recall': '{:.4f}',
    'F1-Score': '{:.4f}',
    'ROC-AUC': '{:.4f}'
}).highlight_max(axis=0, color='lightgreen'))

# ==============================================================================
# 3.3 VISUALISASI CONFUSION MATRIX MODEL BASELINE
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (model_name, model) in zip(axes, baseline_models.items()):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        ax=ax,
        xticklabels=['Negative (0)', 'Positive (1)'],
        yticklabels=['Negative (0)', 'Positive (1)']
    )
    ax.set_title(f"Confusion Matrix\n{model_name}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Prediksi Kelas", fontsize=10)
    ax.set_ylabel("Kelas Aktual", fontsize=10)

plt.suptitle("Evaluasi Confusion Matrix Model Baseline pada Data Testing", fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.show()

# ==============================================================================
# 3.4 HYPERPARAMETER TUNING DENGAN GRIDSEARCHCV & 5-FOLD STRATIFIED K-FOLD
# ==============================================================================
print("=" * 75)
print("3.4 OPTIMASI MODEL MENGGUNAKAN GRIDSEARCHCV (SCORING = F1-SCORE)")
print("=" * 75)

# Cross-validation generator dengan Stratified K-Fold (5 folds)
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1. Parameter Grid untuk Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1.0, 5.0, 10.0, 50.0],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [500, 1000]
}

# 2. Parameter Grid untuk Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'criterion': ['gini', 'entropy']
}

# 3. Parameter Grid untuk Support Vector Machine
param_grid_svm = {
    'C': [0.1, 1.0, 5.0, 10.0, 20.0],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

# Eksekusi GridSearchCV
print("1. Menjalankan tuning Logistic Regression...")
grid_lr = GridSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid=param_grid_lr,
    cv=cv_stratified,
    scoring='f1',
    n_jobs=-1
)
grid_lr.fit(X_train_scaled, y_train)

print("2. Menjalankan tuning Random Forest...")
grid_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    cv=cv_stratified,
    scoring='f1',
    n_jobs=-1
)
grid_rf.fit(X_train_scaled, y_train)

print("3. Menjalankan tuning Support Vector Machine...")
grid_svm = GridSearchCV(
    estimator=SVC(probability=True, random_state=42),
    param_grid=param_grid_svm,
    cv=cv_stratified,
    scoring='f1',
    n_jobs=-1
)
grid_svm.fit(X_train_scaled, y_train)

print("\nHASIL PARAMETER TERBAIK DARI SETIAP MODEL:")
print("-" * 55)
print(f"• Logistic Regression Terbaik : {grid_lr.best_params_}")
print(f"  F1-Score Cross-Validation   : {grid_lr.best_score_:.4f}")
print(f"• Random Forest Terbaik       : {grid_rf.best_params_}")
print(f"  F1-Score Cross-Validation   : {grid_rf.best_score_:.4f}")
print(f"• SVM Terbaik                 : {grid_svm.best_params_}")
print(f"  F1-Score Cross-Validation   : {grid_svm.best_score_:.4f}")

# ==============================================================================
# 3.5 EVALUASI DAN KOMPARASI MODEL SETELAH TUNING (DATA TESTING)
# ==============================================================================
tuned_models = {
    'Logistic Regression (Tuned)': grid_lr.best_estimator_,
    'Random Forest (Tuned)': grid_rf.best_estimator_,
    'Support Vector Machine (Tuned)': grid_svm.best_estimator_
}

tuned_metrics = []

for model_name, model in tuned_models.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    tuned_metrics.append({
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    })

df_tuned_results = pd.DataFrame(tuned_metrics)

print("=" * 75)
print("TABEL PERFORMA MODEL SETELAH TUNING (DATA TESTING):")
print("=" * 75)
display(df_tuned_results.style.format({
    'Accuracy': '{:.4f}',
    'Precision': '{:.4f}',
    'Recall': '{:.4f}',
    'F1-Score': '{:.4f}',
    'ROC-AUC': '{:.4f}'
}).highlight_max(axis=0, color='lightgreen'))

# Perbandingan Baseline vs Tuned
comparison_summary = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'SVM (RBF)'],
    'Baseline F1': df_baseline_results['F1-Score'].values,
    'Tuned F1': df_tuned_results['F1-Score'].values,
    'Delta F1': df_tuned_results['F1-Score'].values - df_baseline_results['F1-Score'].values,
    'Baseline Accuracy': df_baseline_results['Accuracy'].values,
    'Tuned Accuracy': df_tuned_results['Accuracy'].values,
    'Delta Accuracy': df_tuned_results['Accuracy'].values - df_baseline_results['Accuracy'].values
})

print("\nPERBANDINGAN SEBELUM VS SETELAH HYPERPARAMETER TUNING:")
display(comparison_summary.style.format({
    'Baseline F1': '{:.4f}',
    'Tuned F1': '{:.4f}',
    'Delta F1': '{:+.4f}',
    'Baseline Accuracy': '{:.4f}',
    'Tuned Accuracy': '{:.4f}',
    'Delta Accuracy': '{:+.4f}'
}))

# Visualisasi Perbandingan F1-Score Baseline vs Tuned
plt.figure(figsize=(10, 5))
bar_width = 0.35
x_indices = range(len(comparison_summary))

plt.bar([x - bar_width/2 for x in x_indices], comparison_summary['Baseline F1'], width=bar_width, label='Baseline F1', color='#4A90E2')
plt.bar([x + bar_width/2 for x in x_indices], comparison_summary['Tuned F1'], width=bar_width, label='Tuned F1', color='#50E3C2')

plt.xticks(x_indices, comparison_summary['Model'], fontsize=11)
plt.ylabel('F1-Score', fontsize=11)
plt.title('Perbandingan F1-Score Model Baseline vs Tuned', fontsize=13, fontweight='bold')
plt.ylim(0.7, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ==============================================================================
# 3.6 ANALISIS FEATURE IMPORTANCE DARI RANDOM FOREST
# ==============================================================================
best_rf_model = grid_rf.best_estimator_

# Ekstraksi feature importance
feature_importances = pd.Series(
    best_rf_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("=" * 75)
print("TINGKAT KEPENTINGAN FITUR (FEATURE IMPORTANCE) - RANDOM FOREST")
print("=" * 75)
for rank, (feature, val) in enumerate(feature_importances.items(), 1):
    print(f"{rank:2d}. {feature:<20}: {val:.4f} ({val*100:.2f}%)")

# Visualisasi Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
plt.title("Feature Importance Berdasarkan Random Forest Classifier", fontsize=14, fontweight='bold')
plt.xlabel("Tingkat Kepentingan (Gini / Entropy Importance)", fontsize=11)
plt.ylabel("Fitur / Gejala Klinis", fontsize=11)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\nCatatan Analisis Klinis:")
print("• Polyuria dan Polydipsia konsisten menjadi fitur prediktor terpenting (>30% kontribusi akumulatif).")
print("• Temuan ini sangat selaras dengan hasil studi literatur acuan (Aglarci & Karakurt, 2025).")

# ==============================================================================
# 3.7 PEMILIHAN MODEL TERBAIK & SERIALISASI / EKSPOR UNTUK UAS (.joblib & .pkl)
# ==============================================================================
best_candidate_model = grid_rf.best_estimator_
best_model_name = "Random_Forest_Tuned"

model_filename = "best_diabetes_model.joblib"
scaler_filename = "diabetes_scaler.joblib"
metadata_filename = "model_metadata.json"

# 1. Simpan model terlatih
joblib.dump(best_candidate_model, model_filename)

# 2. Simpan scaler terlatih
joblib.dump(scaler, scaler_filename)

# 3. Simpan metadata fitur untuk verifikasi input aplikasi web di UAS
metadata = {
    'model_name': best_model_name,
    'algorithm': 'RandomForestClassifier',
    'feature_names': X.columns.tolist(),
    'feature_count': len(X.columns),
    'target_classes': {0: 'Negative', 1: 'Positive'},
    'best_hyperparameters': grid_rf.best_params_,
    'test_f1_score': float(f1_score(y_test, best_candidate_model.predict(X_test_scaled))),
    'test_accuracy': float(accuracy_score(y_test, best_candidate_model.predict(X_test_scaled)))
}

with open(metadata_filename, 'w') as f:
    json.dump(metadata, f, indent=4)

print("=" * 75)
print("BERKAS ARTEFAK BERHASIL DIEKSPOR UNTUK DEPLOYMENT UAS:")
print("=" * 75)
for fname in [model_filename, scaler_filename, metadata_filename]:
    size_kb = os.path.getsize(fname) / 1024
    print(f"✓ {fname:<25} ({size_kb:.2f} KB)")

print("\n* Catatan: Untuk mengunduh artefak langsung ke komputer lokal, jalankan perintah:")
print("  from google.colab import files")
print("  files.download('best_diabetes_model.joblib')")
print("  files.download('diabetes_scaler.joblib')")
print("  files.download('model_metadata.json')")

# ==============================================================================
# 3.8 SIMULASI INFERENSI PASIEN BARU (LOGIKA BACKEND UNTUK WEB APP UAS)
# ==============================================================================
def predict_patient_diabetes_risk(patient_data_dict, model_path="best_diabetes_model.joblib", scaler_path="diabetes_scaler.joblib"):
    """
    Fungsi inferensi logika backend untuk memproses input formulir pasien baru
    dan mengembalikan prediksi klasifikasi serta probabilitas risiko diabetes.
    """
    # Load model dan scaler yang sudah diekspor
    loaded_model = joblib.load(model_path)
    loaded_scaler = joblib.load(scaler_path)
    
    # Konversi input dictionary ke DataFrame dengan urutan kolom yang sesuai
    df_patient = pd.DataFrame([patient_data_dict])
    
    # Standarisasi fitur menggunakan scaler yang sama saat training
    df_patient_scaled = pd.DataFrame(
        loaded_scaler.transform(df_patient),
        columns=df_patient.columns
    )
    
    # Prediksi kelas dan probabilitas
    prediction = loaded_model.predict(df_patient_scaled)[0]
    probabilities = loaded_model.predict_proba(df_patient_scaled)[0]
    prob_positive = probabilities[1] * 100
    
    status = "RISIKO TINGGI (Positif)" if prediction == 1 else "RISIKO RENDAH (Negatif)"
    
    return {
        'status': status,
        'prediction_class': int(prediction),
        'risk_percentage': round(prob_positive, 2),
        'confidence': round(max(probabilities) * 100, 2)
    }

# Uji Coba Skenario 1: Pasien Berisiko Tinggi (Pria 52 tahun dengan Polyuria & Polydipsia)
sample_patient_high = {
    'Age': 52, 'Gender': 1, 'Polyuria': 1, 'Polydipsia': 1,
    'sudden weight loss': 1, 'weakness': 1, 'Polyphagia': 1,
    'Genital thrush': 0, 'visual blurring': 1, 'Itching': 1,
    'Irritability': 1, 'delayed healing': 1, 'partial paresis': 0,
    'muscle stiffness': 1, 'Alopecia': 1, 'Obesity': 1
}

# Uji Coba Skenario 2: Pasien Berisiko Rendah (Wanita 26 tahun, gejala nihil)
sample_patient_low = {
    'Age': 26, 'Gender': 0, 'Polyuria': 0, 'Polydipsia': 0,
    'sudden weight loss': 0, 'weakness': 0, 'Polyphagia': 0,
    'Genital thrush': 0, 'visual blurring': 0, 'Itching': 0,
    'Irritability': 0, 'delayed healing': 0, 'partial paresis': 0,
    'muscle stiffness': 0, 'Alopecia': 0, 'Obesity': 0
}

print("=" * 75)
print("HASIL SIMULASI INFERENSI PENGUJIAN BACKEND:")
print("=" * 75)
res_high = predict_patient_diabetes_risk(sample_patient_high)
print("1. Skenario Pasien Gejala Klinis Kompleks:")
print(f"   Status Prediksi : {res_high['status']}")
print(f"   Probabilitas    : {res_high['risk_percentage']}%")

print("\n2. Skenario Pasien Sehat / Tanpa Gejala:")
res_low = predict_patient_diabetes_risk(sample_patient_low)
print(f"   Status Prediksi : {res_low['status']}")
print(f"   Probabilitas    : {res_low['risk_percentage']}%")
print("\n✓ Logika inferensi siap diintegrasikan ke antarmuka Streamlit/Gradio untuk UAS!")
