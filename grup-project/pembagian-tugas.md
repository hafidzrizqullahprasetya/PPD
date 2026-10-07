*PEMBAGIAN TUGAS GROUP PROJECT PPD - KELOMPOK 3*
Topik: Deteksi Dini Risiko Diabetes Melitus Tipe 2 Berdasarkan Gejala Klinis Awal Menggunakan Machine Learning
Dataset: UCI Early Stage Diabetes Risk Prediction (520 Pasien)

Catatan: Pembagian dibagi rata. Setiap anggota memegang porsi coding di Google Colab, materi slide presentasi UTS, dan pengembangan aplikasi web di UAS.

========================================
1. LILIS PUSPITA HARIA
========================================
• Coding di Colab:
- Data Cleaning & Preprocessing (handling missing value, encoding biner Yes/No -> 1/0, feature scaling).
- Splitting data (Train-Test Split & Stratified Sampling).
• Slide & Presentasi UTS:
- Bagian Pendahuluan: Latar belakang urgensi deteksi dini diabetes, rumusan masalah, dan tujuan proyek.
• Web App Deployment UAS:
- Merancang form kuesioner input gejala pasien di web (antarmuka input data klinis).

========================================
2. AURELL ACHMAD MADINA HARTAMA
========================================
• Coding di Colab:
- Exploratory Data Analysis / EDA (analisis statistik deskriptif, distribusi usia & gender).
- Visualisasi korelasi fitur (heatmap korelasi, perbandingan gejala pada pasien positif vs negatif).
• Slide & Presentasi UTS:
- Bagian Karakteristik Data & Metodologi: Penjelasan dataset 520 pasien Sylhet Hospital dan tahapan workflow data mining.
• Web App Deployment UAS:
- Membuat visualisasi hasil diagnosa di web (risk meter gauge chart & radar chart probabilitas gejala).

========================================
3. HAFIDZ RIZQULLAH PRASETYA
========================================
• Coding di Colab:
- Training Model ML Klasifikasi (Logistic Regression, Random Forest, dan SVM).
- Hyperparameter tuning dengan GridSearch / RandomSearch untuk optimasi performa.
- Ekspor model terbaik (serialisasi ke file .pkl / .joblib).
• Slide & Presentasi UTS:
- Bagian Pemodelan Machine Learning: Menjelaskan arsitektur algoritma yang digunakan dan proses training model.
• Web App Deployment UAS:
- Membangun backend pipeline prediksi web (loading model .pkl dan fungsi inferensi data baru).

========================================
4. MUHAMMAD ZIDAN ALHILALI
========================================
• Coding di Colab:
- Training Model Pembanding (XGBoost & KNN) serta K-Fold Cross Validation.
- Evaluasi & Komparasi Model (Akurasi, Precision, Recall, F1-Score, Confusion Matrix, dan ROC-AUC Curve).
• Slide & Presentasi UTS:
- Bagian Hasil Evaluasi, Pembahasan, & Kesimpulan: Analisis tabel performa model dan fitur paling berpengaruh (Polyuria/Polydipsia).
• Web App Deployment UAS:
- Mengintegrasikan halaman edukasi medis, rekomendasi pencegahan, dan deployment hosting aplikasi ke cloud (Streamlit Cloud).

========================================
ALUR KERJA BERSAMA (AGAR TETAP SINKRON):
========================================
1. Pengerjaan Colab dilakukan bersama di 1 link Google Colab (setiap orang mengerjakan section notebook masing-masing).
2. Slide presentasi UTS dibuat di 1 link Google Slides bersama (masing-masing mengisi slide bagiannya).
3. Presentasi 10 menit dibagi rata: masing-masing anggota berbicara sekitar 2,5 menit saat ujian UTS.
