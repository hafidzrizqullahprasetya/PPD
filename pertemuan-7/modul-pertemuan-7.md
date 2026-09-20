# Modul Praktikum Penambangan Data – Teknologi Rekayasa Perangkat Lunak – 2025
## PERTEMUAN VII: IMBALANCED DATASET

### 7.1 TUJUAN PEMBELAJARAN
A. Mahasiswa memahami konsep *imbalanced dataset* serta permasalahan yang muncul akibat ketidakseimbangan kelas dalam model *machine learning*.  
B. Mahasiswa mampu menerapkan teknik *balancing data* seperti *Random Under-Sampling*, *Random Over-Sampling*, SMOTE, dan ADASYN untuk menyeimbangkan distribusi kelas pada dataset.  
C. Mahasiswa memahami konsep dan metode *feature selection* (*filter*, *wrapper*, *embedded*) untuk memilih fitur yang paling relevan.  

---

### 7.2 DASAR TEORI

**• Imbalanced Class**  
*Imbalance class* adalah kondisi di mana dalam suatu himpunan data terdapat satu kelas (kelas mayoritas) yang memiliki jumlah sampel jauh lebih besar dibandingkan kelas lainnya (kelas minoritas). Kondisi ini menyebabkan algoritma klasifikasi standar cenderung bias memprediksi kelas mayoritas dan gagal mengenali kelas minoritas dengan baik.  
Selain itu, *imbalance class* mengakibatkan nilai sensitivitas (*recall*) yang rendah pada kelas target yang sering kali justru merupakan kelas yang paling krusial untuk dideteksi (misalnya deteksi penyakit atau *fraud detection*). Ketidakseimbangan kelas dapat ditangani melalui manipulasi level data (*resampling*), modifikasi algoritma (*cost-sensitive learning*), atau kombinasi keduanya (Istiana & Mustafiril, 2023).

**• Confusion Matrix**  
*Confusion Matrix* adalah matriks evaluasi berbentuk tabel kontingensi yang membandingkan label aktual dengan hasil prediksi model *machine learning* (Normawati & Prayogi, 2021). Matriks ini memuat empat komponen fundamental:
- **True Positive (TP)**: Sampel positif yang berhasil diprediksi positif.
- **True Negative (TN)**: Sampel negatif yang berhasil diprediksi negatif.
- **False Positive (FP)**: Sampel negatif yang salah diprediksi sebagai positif (*Type I Error*).
- **False Negative (FN)**: Sampel positif yang salah diprediksi sebagai negatif (*Type II Error*).  
Berdasarkan nilai-nilai tersebut, dapat dihitung metrik evaluasi seperti *Accuracy*, *Precision*, *Recall*, dan *F1-Score* (Septiana, Susanto, & Tukiyat, 2021).

**• Feature Selection**  
*Feature Selection* merupakan proses identifikasi dan pemilihan subset fitur yang paling informatif dan relevan dari sekumpulan fitur awal untuk meningkatkan kinerja dan efisiensi algoritma (Pratama & Kusnawi, 2023). Ketika dataset memiliki dimensi fitur yang terlalu besar (*curse of dimensionality*), model rentan mengalami *overfitting* dan waktu pelatihan meningkat drastis. Seleksi fitur bertujuan mereduksi dimensi ruang data dengan membuang fitur redundan dan tidak relevan (Septiana, Susanto, & Tukiyat, 2021).

**• Filter Method**  
*Filter Method* merupakan pendekatan seleksi fitur independen yang mengevaluasi relevansi fitur berdasarkan sifat statistik dan korelasinya terhadap variabel target tanpa melibatkan model *machine learning* (Guyon & Elisseeff, 2003). Metrik yang umum digunakan meliputi korelasi Pearson/Spearman, uji Chi-Square ($\chi^2$), serta uji ANOVA (*Analysis of Variance*) untuk mengukur kemampuan fitur membedakan antar-kelompok kelas. Karena prosesnya tidak melatih model prediktif, *filter method* sangat cepat dan hemat komputasi.

**• Wrapper Method**  
*Wrapper Method* mengevaluasi kombinasi subset fitur dengan memanfaatkan algoritma *machine learning* secara langsung sebagai fungsi evaluasi (Kohavi & John, 1997). Metode ini secara iteratif mencoba berbagai kombinasi fitur—misalnya melalui teknik *Forward Selection*, *Backward Elimination*, atau *Recursive Feature Elimination* (RFE)—lalu memilih kombinasi yang menghasilkan performa model tertinggi. Pendekatan ini lebih akurat dibanding *filter method*, namun memerlukan komputasi yang jauh lebih berat karena melatih model secara berulang.

**• Embedded Method**  
*Embedded Method* memadukan keunggulan *filter* dan *wrapper* dengan melakukan proses seleksi fitur secara bersamaan (*built-in*) selama pelatihan model. Contoh paling umum adalah regresi LASSO (*L1 Regularization*) yang memaksa koefisien fitur yang tidak signifikan menjadi tepat nol (Tibshirani, 1996), serta model berbasis pohon keputusan (*Random Forest*, *Gradient Boosting*) yang menghitung *feature importance* berdasarkan penurunan *Gini impurity* atau *entropy*.

**• SMOTE**  
*Synthetic Minority Over-sampling Technique* (SMOTE) merupakan metode *oversampling* yang bekerja dengan membangkitkan sampel sintetis buatan pada kelas minoritas (bukan sekadar duplikasi data). SMOTE beroperasi di ruang fitur dengan mencari $k$-tetangga terdekat (*k-nearest neighbors*) dari setiap sampel kelas minoritas, lalu membuat titik sintetis baru di sepanjang garis linier yang menghubungkan sampel tersebut dengan salah satu tetangganya yang dipilih secara acak (Wijayanti, Kencana, & Sumarjaya, 2021).

**• ADASYN**  
*Adaptive Synthetic Sampling Approach* (ADASYN) adalah pengembangan dari SMOTE yang menggunakan pendekatan adaptif. ADASYN menghitung bobot distribusi kesulitan belajar bagi setiap sampel minoritas; sampel yang berada di daerah batas keputusan (*decision boundary*) dan lebih sulit dipelajari oleh model akan diberikan proporsi pembangkitan sampel sintetis yang lebih banyak dibanding sampel yang berada di area mudah (Ramadhanti, Santoso, & Widiharih, 2022).

---

### 7.3 ALAT DAN BAHAN

**A. Perangkat Keras**  
1. Laptop dengan minimal RAM 4 GB  
2. Koneksi internet stabil  

**B. Perangkat Lunak**  
1. Web Browser (Google Chrome, Mozilla Firefox, Microsoft Edge, dll.)  
2. Google Colab / Jupyter Notebook  
3. Pustaka Python: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn` (`imblearn`), `matplotlib`, `seaborn`  

---

### 7.4 LANGKAH PERCOBAAN

#### 1. Persiapkan IDE
Mahasiswa membuka Google Colab sebagai lingkungan kerja utama. Google Colab telah menyediakan dukungan lingkungan Python interaktif beserta akses CPU/GPU gratis untuk pengolahan dataset dan komputasi penyeimbangan kelas secara efisien.

#### 2. Instalasi dan Impor Library
Menginstal pustaka `imbalanced-learn` (jika belum tersedia) dan mengimpor seluruh modul yang dibutuhkan:
- `pandas` dan `numpy` untuk manipulasi data tabular.
- `matplotlib.pyplot` dan `seaborn` untuk visualisasi sebaran dan distribusi kelas.
- `RandomUnderSampler`, `RandomOverSampler`, `SMOTE`, dan `ADASYN` dari `imblearn.over_sampling` / `imblearn.under_sampling`.
- Modul pemodelan dan evaluasi dari `sklearn`: `train_test_split`, `DecisionTreeClassifier`, `classification_report`, `confusion_matrix`.

![Gambar 7.4.1 Impor Pustaka](images/gambar-7-4-1-impor-pustaka.png)  
*Gambar 7.4.1 Impor Pustaka*

#### 3. Import dan Load Dataset Information
Memuat *Stroke Prediction Dataset* yang memuat informasi klinis dan demografis pasien (`age`, `gender`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, serta label target `stroke`):

1. Membaca dataset dan menampilkan 5 baris awal:

![Gambar 7.4.2 Pembacaan Dataset](images/gambar-7-4-2-pembacaan-dataset.png)  
*Gambar 7.4.2 Pembacaan Dataset*

2. Memeriksa nilai kosong (*missing values*) dan tipe data tiap fitur:

![Gambar 7.4.3 Mengecek Null Value](images/gambar-7-4-3-mengecek-null-value.png)  
*Gambar 7.4.3 Mengecek Null Value*

#### 4. Exploratory Data Analysis (EDA)
Memvisualisasikan proporsi distribusi kelas target (`stroke`) menggunakan diagram lingkaran (*pie chart*) untuk memverifikasi tingkat ketidakseimbangan kelas (*class imbalance ratio*).

![Gambar 7.4.4 Distribusi data](images/gambar-7-4-4-distribusi-data.png)  
*Gambar 7.4.4 Distribusi data*

#### 5. Data Cleaning
Ditemukan 201 nilai hilang pada kolom `bmi`. Karena BMI bertipe data numerik kontinu dengan sebaran stabil, penanganan nilai hilang dilakukan dengan imputasi nilai rata-rata (*mean*):

![Gambar 7.4.5 Data Cleaning](images/gambar-7-4-5-data-cleaning.png)  
*Gambar 7.4.5 Data Cleaning*

Memeriksa kembali untuk memastikan tidak ada nilai null yang tersisa:

![Gambar 7.4.6 Mengecek jumlah data null](images/gambar-7-4-6-mengecek-jumlah-data-null.png)  
*Gambar 7.4.6 Mengecek jumlah data null*

#### 6. Random Under-Sampling
Memisahkan fitur ($X$) dan label target ($y$), melakukan *encoding* pada fitur-fitur kategorikal (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`), serta membagi data menjadi set pelatihan dan pengujian dengan rasio 75:25.

Menerapkan `RandomUnderSampler` pada data pelatihan untuk mereduksi jumlah sampel kelas mayoritas hingga seimbang dengan kelas minoritas:

![Gambar 7.4.7 Random Sampling](images/gambar-7-4-7-random-sampling.png)  
*Gambar 7.4.7 Random Sampling*

Menampilkan data sampel hasil *resampling*:

![Gambar 7.4.8 Melihat data](images/gambar-7-4-8-melihat-data.png)  
*Gambar 7.4.8 Melihat data*

Perbandingan distribusi jumlah sampel sebelum vs sesudah *Random Under-Sampling*:

![Gambar 7.4.9 Perbandingan sebelum random sampling dan sesudah](images/gambar-7-4-9-perbandingan-sebelum-random-sampling-dan-sesudah.png)  
*Gambar 7.4.9 Perbandingan sebelum random sampling dan sesudah*

Visualisasi sebaran titik data (*scatter plot*) kelas mayoritas dan minoritas setelah *under-sampling*:

![Gambar 7.4.10 Visualisasi Scatter](images/gambar-7-4-10-visualisasi-scatter.png)  
*Gambar 7.4.10 Visualisasi Scatter*

#### 7. Random Over-Sampling
Menerapkan `RandomOverSampler` pada data pelatihan untuk menduplikasi sampel kelas minoritas secara acak hingga jumlahnya setara dengan kelas mayoritas:

![Gambar 7.4.11 Random over sampling](images/gambar-7-4-11-random-over-sampling.png)  
*Gambar 7.4.11 Random over sampling*

Perbandingan distribusi jumlah sampel sebelum vs sesudah *Random Over-Sampling*:

![Gambar 7.4.12 Sebelum vs Sesudah over sampling](images/gambar-7-4-12-sebelum-vs-sesudah-over-sampling.png)  
*Gambar 7.4.12 Sebelum vs Sesudah over sampling*

Visualisasi sebaran data setelah *Random Over-Sampling*:

![Gambar 7.4.13 Visualisasi scatter](images/gambar-7-4-13-visualisasi-scatter.png)  
*Gambar 7.4.13 Visualisasi scatter*

#### 8. SMOTE (Synthetic Minority Over-sampling Technique)
Menerapkan algoritma SMOTE untuk membangkitkan data sintetis baru bagi kelas minoritas di sepanjang garis penghubung tetangga terdekat:

![Gambar 7.4.14 SMOTE](images/gambar-7-4-14-smote.png)  
*Gambar 7.4.14 SMOTE*

Perbandingan distribusi jumlah sampel sebelum vs sesudah penerapan SMOTE:

![Gambar 7.4.15 Sebelum vs Sesudah SMOTE](images/gambar-7-4-15-sebelum-vs-sesudah-smote.png)  
*Gambar 7.4.15 Sebelum vs Sesudah SMOTE*

Visualisasi sebaran ruang fitur hasil penambahan data sintetis SMOTE:

![Gambar 7.4.16 Visualisasi Scatter](images/gambar-7-4-16-visualisasi-scatter.png)  
*Gambar 7.4.16 Visualisasi Scatter*

#### 9. ADASYN (Adaptive Synthetic Sampling)
Menerapkan ADASYN yang secara adaptif menghasilkan lebih banyak sampel sintetis pada titik-titik batas kelas minoritas yang sulit dipelajari oleh *classifier*:

![Gambar 7.4.17 ADASYN](images/gambar-7-4-17-adasyn.png)  
*Gambar 7.4.17 ADASYN*

Perbandingan distribusi jumlah sampel sebelum vs sesudah penerapan ADASYN:

![Gambar 7.4.18 Sebelum dan Sesudah Adasyn](images/gambar-7-4-18-sebelum-dan-sesudah-adasyn.png)  
*Gambar 7.4.18 Sebelum dan Sesudah Adasyn*

Visualisasi sebaran ruang fitur hasil pembentukan data sintetis ADASYN:

![Gambar 7.4.19 Visualisasi scatter](images/gambar-7-4-19-visualisasi-scatter.png)  
*Gambar 7.4.19 Visualisasi scatter*

---

### 7.5 TUGAS DAN ANALISIS
1. **Tugas 1 (Eksplorasi Dataset):**
   - Unduh dataset *Stroke Prediction Dataset*: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
   - Unggah file dataset ke Google Drive atau repositori GitHub masing-masing.
   - Tampilkan dataset ke dalam DataFrame pandas.
   - Jelaskan tujuan penggunaan dataset ini dalam konteks penambangan data.
   - Jelaskan seluruh fitur yang terdapat pada dataset.
2. **Tugas 2 (Komparasi Balancing & Klasifikasi):**
   - Gunakan dataset klasifikasi lain yang bersifat *imbalanced*.
   - Kombinasikan teknik SMOTE atau ADASYN dengan algoritma pengklasifikasi (misalnya Decision Tree, Random Forest, atau SVM).
   - Tampilkan tabel performa model (*Accuracy*, *Precision*, *Recall*, *F1-Score*) sebelum dan sesudah penerapan *data balancing* pada data pelatihan (*training set*).

---

### 7.6 REFERENSI
- Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157–1182.
- Istiana, A., & Mustafiril, M. (2023). Perbandingan Metode Klasifikasi pada Data dengan Imbalance Class dan Missing Value. *JURNAL INFORMATIKA*, 101-108.
- Kohavi, R., & John, G. H. (1997). Wrappers for feature subset selection. *Artificial Intelligence*, 97(1–2), 273–324.
- Normawati, & Prayogi, T. (2021). Implementasi Naïve Bayes Classifier Dan Confusion Matrix Pada Analisis Sentimen Berbasis Teks Pada Twitter. *J-SAKTI (Jurnal Sains Komputer & Informatika)*, 697-711.
- Pratama, R., & Kusnawi, K. (2023). Komparasi Algoritma Supervised Learning dan Feature Selection pada Klasifikasi Penyakit Gagal Jantung. *Indonesian Journal of Computer Science*, 3722-3733.
- Ramadhanti, A. P., Santoso, H. B., & Widiharih, T. (2022). Perbandingan SMOTE dan ADASYN pada Data Imbalance untuk Klasifikasi Rumah Tangga Miskin Kabupaten Temanggung dengan Algoritma K-Nearest Neighbor. *JURNAL GAUSSIAN*, 499-505.
- Septiana, R., Susanto, A., & Tukiyat. (2021). Analisis Sentimen Vaksinasi Covid-19 Pada Twitter Menggunakan Naive Bayes Classifier Dengan Feature Selection Chi-Squared Statistic Dan Particle Swarm Optimization. *Jurnal Sistem Komputer dan Kecerdasan Buatan*, 49-56.
- Tibshirani, R. (1996). Regression shrinkage and selection via the Lasso. *Journal of the Royal Statistical Society: Series B (Methodological)*, 58(1), 267–288.
- Wijayanti, N. L. P., Kencana, I. W., & Sumarjaya, I. M. (2021). SMOTE: Potensi dan Kekurangannya pada Survei. *E-Jurnal Matematika*, 235-240.
