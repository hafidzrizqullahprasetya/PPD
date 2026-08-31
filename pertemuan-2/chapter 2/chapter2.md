# Rangkuman Lengkap Materi: Artificial Intelligence, Machine Learning, dan Data Preprocessing

Berikut adalah rangkuman komprehensif dari seluruh materi yang telah dipelajari, disusun secara sistematis per topik.

---

## 📌 BAGIAN 1: ARTIFICIAL INTELLIGENCE (AI)

### 1.1 Apa itu Kecerdasan (Intelligence)?

Kecerdasan manusia terdiri dari beberapa komponen utama:

- **Persepsi (Perception):** Kemampuan penginderaan seperti penglihatan, pendengaran, dan sentuhan.
- **Pemahaman (Understanding):** Kemampuan memahami bahasa, bicara, dan visual.
- **Penalaran (Reasoning):** Kemampuan menarik fakta baru dari fakta yang sudah ada.
- **Pembelajaran (Learning):** Kemampuan meningkatkan kinerja berdasarkan pengalaman.
- **Adaptasi & Kreativitas:** Kemampuan menyesuaikan diri dan berinovasi.

### 1.2 Definisi Artificial Intelligence

**Artificial Intelligence (AI)** adalah kemampuan komputer atau robot untuk melakukan tugas-tugas yang biasanya membutuhkan kecerdasan manusia. Tujuan AI adalah:
- Menemukan cara manusia berpikir/bertindak secara cerdas.
- Mengembangkan sistem yang melakukan tugas cerdas.

> **Definisi John McCarthy (1955):** *"The branch of computer science concerned with making computers behave like humans."*

Sistem AI dirancang untuk: berpikir seperti manusia, bertindak seperti manusia, berpikir rasional, dan bertindak rasional.

### 1.3 Sejarah & 3 Boom AI

**📊 Deskripsi Grafik Timeline Sejarah AI:**
Grafik garis waktu menunjukkan tiga periode "boom" (ledakan) AI yang dipisahkan oleh masa resesi:
- **1950s:** Boom pertama dimulai
- **1960s:** Boom pertama, lalu resesi
- **1970s:** Masa resesi
- **1980s:** Boom kedua (Expert System)
- **1990s:** Boom ketiga (Machine Learning), transisi dari Analog ke Digital
- **2000s:** Deep Learning, transisi Digital ke Semantic

**Boom Pertama (1950s-1960s):**
- Upaya pemecahan masalah melalui pencarian dan inferensi.
- Mereda karena kesulitan menyelesaikan masalah kompleks.

**Boom Kedua (1980s):**
- Era "Expert System" (Sistem Pakar).
- AI berpengetahuan dengan memasukkan knowledge ke komputer.
- Mereda karena knowledge terlalu besar untuk dikelola.

**Boom Ketiga (1990s-2000s):**
- AI belajar dari data (Machine Learning).
- Dimulai dari kemunculan search engine.
- Berkembang pesat dengan Deep Learning.

### 1.4 Hubungan AI, ML, dan DL

> **📊 Deskripsi Diagram Venn:**
> Tiga lingkaran konsentris (saling bertumpuk):
> - Lingkaran terluar: **Artificial Intelligence (AI)**
> - Lingkaran tengah: **Machine Learning (ML)** — subset dari AI
> - Lingkaran terdalam: **Deep Learning (DL)** — subset dari ML
>
> Artinya: Semua Deep Learning adalah ML, semua ML adalah AI, tetapi tidak semua AI adalah ML.

### 1.5 Sub-bidang & Aplikasi AI

**Sub-bidang AI:**
- Reasoning (Fuzzy control, Expert systems)
- Vision (Face recognition, autonomous vehicle, object detection)
- Speech recognition
- Natural language understanding
- Machine learning (Clustering, classification, reinforcement learning)
- Robotics
- Game playing (Chess, Go)

**Contoh Aplikasi Nyata:**
- **Google Assistant:** Virtual assistant untuk kontrol perangkat, mencari informasi, dan percakapan natural (termasuk fitur Duplex yang meniru suara manusia).
- **AlphaGo:** Program DeepMind Google yang mengalahkan juara Eropa (5-0, 2015) dan juara Korea (4-1, 2016) dalam permainan Go.
- **Smart Home** dan **Intelligent Healthcare.**

### 1.6 Expert System (Sistem Pakar)

Sistem berbasis pengetahuan yang memberikan jawaban untuk masalah di domain spesifik.

**Knowledge Engineering:**
- **Knowledge acquisition:** Mengambil pengetahuan dari domain expert (merupakan bottleneck utama).
- **Knowledge engineer:** Memilih tools, mengekstrak pengetahuan melalui interview, membangun Knowledge Base (KB).

> **📊 Deskripsi Diagram Alur:**
> Domain Expert → (interview) → Knowledge Engineer → (membangun) → Knowledge Base (KB) dalam Expert System

### 1.7 Rule-Based Reasoning vs Case-Based Reasoning

**Rule-Based Reasoning:**
- Merepresentasikan pengetahuan sebagai aturan eksplisit (IF-THEN).
- **Kelebihan:** Performa baik di domain terbatas, penjelasan mungkin.
- **Kekurangan:** Butuh akuisisi pengetahuan, tidak bisa menangani noisy data.

**Case-Based Reasoning:**
- Penalaran berdasarkan kasus-kasus masa lalu.
- **Prosedur:**
  1. Retrieve kasus serupa untuk masalah baru.
  2. Modify kasus yang ditemukan.
  3. Apply kasus yang sudah dimodifikasi.
  4. Save kasus baru.

**Contoh Kasus:**
| Customer | Sex | Age | Action |
|----------|-----|-----|--------|
| c1 | M | 40 | N |
| c2 | M | 20 | Y |
| c3 | F | 30 | N |
| c4 | M | 30 | Y |
| **c5** | **F** | **20** | **?** |

- **Rule-based:** IF sex=F → N. Maka jawabannya N.
- **Case-based:** c5 paling mirip dengan c3 (sama-sama F). Maka jawabannya N.

---

## 📌 BAGIAN 2: MACHINE LEARNING

### 2.1 Definisi Machine Learning

Machine Learning adalah cabang AI yang berkaitan dengan desain dan pengembangan algoritma yang memungkinkan komputer mengembangkan perilaku berdasarkan **data empiris**. Karena kecerdasan membutuhkan pengetahuan, komputer perlu memperoleh pengetahuan dari data.

**Alur ML:**
```
Training Data (past) → Model/Predictor → Testing Data (future)
```

### 2.2 ML vs Traditional Programming

> **📊 Deskripsi Diagram Perbandingan:**
> - **Traditional Programming:** Input + Rules → Output
> - **Machine Learning:** Input + Output → Rules (Model)
>
> Pada ML, model "belajar" rules dari data, bukan di-hardcode oleh programmer.

### 2.3 Jenis-jenis Learning

**1. Supervised Learning**
- Training data memiliki **target class/label**.
- Contoh: Classification, Regression/Prediction.

**2. Unsupervised Learning**
- Training data **tidak memiliki target class**.
- Contoh: Clustering, Association rules.

**3. Semi-supervised Learning**
- Sebagian training data memiliki output, sebagian tidak.

**4. Reinforcement Learning**
- Agent diberi **reward** saat sukses dan **punishment** saat salah.
- Belajar melalui trial and error.

### 2.4 Supervised Learning: Classification

**Tahap Learning:**
Contoh dataset:
| NAME | RANK | YEARS | TENURED |
|------|------|-------|---------|
| Mike | Assistant Prof | 3 | no |
| Mary | Assistant Prof | 7 | yes |
| Bill | Professor | 2 | yes |
| Jim | Associate Prof | 7 | yes |
| Dave | Assistant Prof | 6 | no |
| Anne | Associate Prof | 3 | no |

**Model yang terbentuk (Classifier):**
```
IF rank = 'professor' OR years > 6 THEN tenured = 'yes'
```

**Tahap Prediction:**
- Input data baru: (Jeff, Professor, 4)
- Classifier memprediksi: **Tenured = Yes**

### 2.5 Classification vs Regression vs Clustering

| Jenis | Tipe Output | Contoh |
|-------|-------------|--------|
| **Classification** | Nilai diskrit | Stroke/Normal, Spam/Bukan Spam |
| **Regression** | Nilai kontinu | Harga rumah, gaji, usia |
| **Clustering** | Kelompok (tanpa label) | Pengelompokan customer |

- **Classification & Regression** = Supervised Learning
- **Clustering** = Unsupervised Learning

### 2.6 Apa itu Model yang Baik?

**Model Validity:**
- Model harus perform baik pada **sampel baru**, bukan hanya pada data training.
- Kemampuan ini disebut **generalization** atau **robustness**.

**Underfitting vs Overfitting:**
- **Underfitting:** Model terlalu sederhana, tidak bisa menangkap pola data.
- **Overfitting:** Model terlalu kompleks, menghafal data training tapi gagal di data baru.

**Bias-Variance Tradeoff:**

> **📊 Deskripsi Grafik Bias-Variance:**
> Grafik dengan sumbu X "Model Complexity" dan sumbu Y "Error":
> - Garis **Bias** menurun dari kiri ke kanan (model kompleks → bias rendah).
> - Garis **Variance** naik dari kiri ke kanan (model kompleks → variance tinggi).
> - Garis **Total Error** berbentuk kurva U.
> - Titik optimal berada di tengah, menyeimbangkan bias dan variance.

- **Bias:** Perbedaan rata-rata prediksi model dengan nilai sebenarnya. Bias tinggi = model terlalu sederhana (oversimplified).
- **Variance:** Variabilitas prediksi. Variance tinggi = model hanya cocok di training data (overfitting).

### 2.7 Tahapan Machine Learning

1. **Data Preprocessing:** Cleaning, filling missing value, remove outlier.
2. **Train Models:** Select algorithm, feature selection & extraction.
3. **Evaluate Model:** Assess performance, model comparison.
4. **Deploy Model:** Apply ke data baru, real-time demonstration.

---

## 📌 BAGIAN 3: DATA PREPROCESSING (TEORI)

### 3.1 Apa itu Data Preprocessing?

Proses persiapan data sebelum analisis/pemodelan untuk meningkatkan kualitas data.

**Mengapa penting?**
> "No quality data, no quality mining results!" — Keputusan berkualitas harus berdasarkan data berkualitas.

**Data di dunia nyata itu "kotor":**
- **Incomplete/Missing:** Contoh: occupation = ""
- **Noisy:** Nilai salah atau outlier. Contoh: salary = "-10"
- **Inconsistent:** Contoh: sex = "Girl" vs sex = "Female"
- **Duplicate:** Data yang sama berulang.

### 3.2 Jenis-jenis Fitur

**Fitur Kategoris:**
- **Nominal:** Hanya membedakan, tidak ada tingkatan. Contoh: Gender, Warna Rambut.
- **Ordinal:** Ada tingkatan/orde. Contoh: Jenjang Pendidikan, Kepuasan Pelanggan.

**Fitur Numerik:**
- **Discrete:** Item yang dapat dihitung. Contoh: Jumlah siswa.
- **Continuous:** Item yang dapat diukur. Contoh: Tinggi, Suhu, Kecepatan.

### 3.3 Tugas Utama Data Preprocessing

1. **Data Cleaning:** Fill missing values, smooth noisy data, remove outliers, resolve inconsistencies.
2. **Data Integration:** Menggabungkan multiple databases/files.
3. **Data Reduction:** Dimensionality reduction.
4. **Data Transformation:** Normalization, Standardization, Encoding.

### 3.4 Handling Missing Data

**Cara menangani:**
- Abaikan baris.
- Isi manual (butuh waktu lama).
- Isi otomatis dengan:
  - Konstanta global (misal: "unknown")
  - Mean/Median (untuk numerik)
  - Rata-rata per kelas
  - Modus (untuk kategoris)

### 3.5 Handling Noisy Data

**Teknik:**
- **Binning:** Urutkan data, partisi ke bin, ganti outlier dengan mean/median bin.
- **Regression:** Smooth data dengan fungsi regresi.
- **Clustering:** Deteksi dan hapus outlier.
- **Combined computer & human inspection.**

### 3.6 Feature Encoding

**One-Hot Encoding:**
- Mengubah setiap kategori menjadi nilai 0 atau 1 (binary).
- Contoh: Warna [Merah, Hijau, Biru] → [1,0,0], [0,1,0], [0,0,1]

**Label Encoding:**
- Mengubah setiap kategori menjadi angka 1, 2, 3, dst.
- Contoh: [Rendah, Sedang, Tinggi] → [1, 2, 3]

### 3.7 Normalization & Standardization

**Normalization:**
- Mengubah nilai fitur ke skala **[0, 1]**.
- Rumus: `(x - min) / (max - min)`

**Standardization:**
- Mengubah nilai fitur sehingga **mean = 0** dan **std = 1**.
- Rumus: `(x - mean) / std`

**Tujuan:**
- Algoritma memperlakukan semua fitur secara adil.
- Mempercepat proses training.
- Mempermudah interpretasi model.

**Kapan menggunakan?**
- **Standardization:** Jika data berdistribusi normal/Gaussian.
- **Normalization:** Untuk data dengan skala berbeda.

### 3.8 Train-Test Split

- **Training set:** Subset untuk melatih model.
- **Test set:** Subset untuk menguji model.
- Membagi dataset karena kurangnya data, biasanya rasio 80:20 atau 70:30.

---

## 📌 BAGIAN 4: EXPLORATORY DATA ANALYSIS (EDA)

### 4.1 Definisi EDA

**EDA (Exploratory Data Analysis)** adalah proses analisis awal untuk memahami dataset sebelum menjalankan model statistik atau pengambilan keputusan.

**Tujuan EDA:**
1. **Deskripsi Data:** Statistik deskriptif (mean, median, modus, kuartil).
2. **Visualisasi Data:** Melihat pola, tren, anomali.
3. **Identifikasi Anomali:** Deteksi outlier.
4. **Pemrosesan Data:** Cleaning & preparation.
5. **Pemeriksaan Korelasi:** Hubungan antar variabel.
6. **Pemilihan Fitur:** Fitur yang relevan.
7. **Eksplorasi Lanjut & Interpretasi Hasil.**

### 4.2 Studi Kasus: Credit Card Customers Dataset

Dataset dari Kaggle berisi informasi nasabah kartu kredit dengan atribut seperti:
- CLIENTNUM, Attrition_Flag (Existing/Attrited Customer)
- Customer_Age, Gender, Dependent_count
- Education_Level, Marital_Status, Income_Category
- Card_Category, Months_on_book
- Months_Inactive_12_mon, Contacts_Count_12_m

**Input (fitur):** Semua atribut kecuali Attrition_Flag.
**Output/Label:** Attrition_Flag (target klasifikasi).

### 4.3 Jenis-jenis Visualisasi Data

#### 📈 Line Chart
Menampilkan hubungan/tren antara dua variabel melalui garis yang menghubungkan titik-titik data.

> **📊 Deskripsi Visual:**
> Sumbu X: nilai [2, 4, 6, 8]
> Sumbu Y: dua garis dengan marker berbeda
> - Garis 1 (marker 'o'): [3, 8, 1, 10]
> - Garis 2 (marker 'X'): [4, 7, 10, 12]
>
> Contoh lain: Line chart IoT sensor dengan sumbu X = tanggal, sumbu Y = Humidity, menunjukkan fluktuasi kelembaban dari waktu ke waktu.

#### 🥧 Pie Chart
Grafik lingkaran yang menunjukkan komposisi relatif bagian-bagian terhadap keseluruhan.

> **📊 Deskripsi Visual Pie Chart Marital Status:**
> Lingkaran dibagi menjadi 4 slice dengan persentase:
> - Married: 46.35%
> - Single: 39.00%
> - Unknown: 7.41%
> - Divorced: 7.40%

#### 📊 Bar Plot
Menampilkan nilai kategori dengan batang vertikal/horizontal.

> **📊 Deskripsi Visual Bar Plot Marital Status:**
> Dua grup batang berdampingan (existing vs attrited customer):
> - Married: Existing ≈ 3978, Attrited ≈ 709
> - Single: Existing ≈ 3275, Attrited ≈ 668
> - Divorced: Existing ≈ 627, Attrited ≈ 121
> - Unknown: Existing ≈ 620, Attrited ≈ 129

**Stacked Bar Plot:**
Batang ditumpuk untuk menunjukkan komposisi. Memungkinkan melihat proporsi relatif dalam satu kategori.

> **📊 Deskripsi Visual Stacked Bar:**
> Setiap batang dibagi menjadi dua segmen (existing & attrited) dengan persentase:
> - Married: 84.87% existing, 15.13% attrited
> - Single: 83.06% existing, 16.94% attrited
> - Divorced: 83.82% existing, 16.18% attrited
> - Unknown: 82.78% existing, 17.22% attrited

#### 📊 Histogram
Menampilkan distribusi frekuensi data numerik dengan membagi rentang nilai menjadi interval (bin).

> **📊 Deskripsi Visual Histogram Customer Age:**
> Dua histogram tumpang tindih:
> - Existing customer (biru): Distribusi usia dengan puncak sekitar 40-50 tahun.
> - Attrited customer (oranye): Distribusi usia yang lebih rendah, menunjukkan customer yang churn cenderung lebih muda atau tersebar lebih luas.
>
> Sumbu X: Age (25-75 tahun)
> Sumbu Y: Count (0-2000)

#### 🔵 Scatter Plot
Menampilkan titik-titik data dalam bidang kartesian untuk mengeksplorasi hubungan dua variabel.

> **📊 Deskripsi Visual Scatter Plot:**
> Sumbu X: Customer Age
> Sumbu Y: Months on Book
> - Titik merah: Existing customer
> - Titik biru: Attrited customer
>
> Terlihat pola sebaran — tidak ada korelasi kuat antara usia dan lama keanggotaan.

#### 📦 Box Plot
Menampilkan distribusi data numerik melalui kotak (kuartil), garis median, dan titik outlier.

> **📊 Deskripsi Visual Box Plot Customer Age:**
> - Kotak: Rentang interkuartil (Q1-Q3)
> - Garis tengah: Median
> - Whiskers: Rentang data normal
> - Titik di luar whiskers: **Outlier potensial**
>
> Perbandingan existing vs attrited customer menunjukkan distribusi usia yang mirip, dengan beberapa outlier di kedua grup.

#### 🎻 Violin Plot
Menggabungkan box plot dengan distribusi kerapatan (density), berbentuk seperti biola.

> **📊 Deskripsi Visual:**
> Bentuk biola menunjukkan kerapatan data di berbagai rentang nilai. Lebih detail daripada histogram karena menampilkan distribusi probabilitas.

### 4.4 Komponen Visualisasi Tambahan

**Subplot:**
Membuat beberapa grafik dalam satu figure. Contoh: Figure 2x2 berisi boxplot dan histogram dari existing & attrited customer secara paralel.

**Annotation:**
Menambahkan teks/penjelasan pada grafik. Contoh: Menambahkan teks "Median Age: 46.0" dan "Mean Age: 45.6" pada histogram.

**Axis (Sumbu):**
- X-axis: Variabel independen.
- Y-axis: Variabel dependen.
- Dapat diatur range-nya dengan `plt.ylim()` dan `plt.xlim()`.

**Legend:**
Keterangan warna/marker dalam grafik. Contoh: "Existing customer" (biru) dan "Attrited customer" (oranye).

### 4.5 Ringkasan Penggunaan Grafik

| Tujuan | Jenis Grafik |
|--------|--------------|
| **Distribusi** | Histogram, Scatter plot |
| **Hubungan** | Scatter plot |
| **Perbandingan** | Bar plot, Line plot |
| **Komposisi** | Pie chart, Stacked bar chart |

---

## 📌 BAGIAN 5: PRAKTIKUM DATA PREPROCESSING

### 5.1 Studi Kasus: Titanic Dataset

Dataset dari Kaggle berisi informasi penumpang Titanic dengan tujuan memprediksi siapa yang selamat (Survived).

**Atribut:**
- PassengerId, Survived (target), Pclass, Name, Sex, Age
- SibSp (saudara/pasangan), Parch (orang tua/anak)
- Ticket, Fare, Cabin, Embarked

### 5.2 Langkah-langkah Preprocessing

#### Step 1: Membaca Dataset
```python
import pandas as pd
df = pd.read_csv('train.csv')
```

#### Step 2: Cek Target Class
> **📊 Deskripsi Pie Chart Survived:**
> - Tidak Survived (0): ≈ 61.62%
> - Survived (1): ≈ 38.38%
>
> Dataset tidak seimbang (imbalanced), lebih banyak yang tidak selamat.

#### Step 3: Distribusi Data
> **📊 Deskripsi Histogram:**
> Grid histogram untuk semua kolom numerik:
> - PassengerId: Distribusi uniform 1-891
> - Survived: Binary 0 dan 1
> - Pclass: 3 kategori (1, 2, 3)
> - Age: Distribusi normal dengan puncak di usia 20-30
> - SibSp, Parch: Mayoritas 0, beberapa 1-5
> - Fare: Skew kanan, banyak di harga rendah, beberapa sangat tinggi

#### Step 4: Korelasi Antar Variabel

> **📊 Deskripsi Heatmap Korelasi:**
> Matriks korelasi dengan nilai:
> - **Survived vs Pclass:** -0.34 (korelasi negatif sedang — kelas lebih rendah = survival lebih tinggi)
> - **Survived vs Fare:** +0.26 (korelasi positif lemah — tarif lebih tinggi = survival lebih tinggi)
> - **Pclass vs Fare:** -0.55 (korelasi negatif kuat — kelas 1 = tarif mahal)
> - **SibSp vs Parch:** +0.41 (korelasi positif sedang)

#### Step 5: Descriptive Statistics

| Statistic | Age | Fare |
|-----------|-----|------|
| Count | 714 | 891 |
| Mean | 29.70 | 32.20 |
| Std | 14.53 | 49.69 |
| Min | 0.42 | 0.00 |
| 25% | 20.13 | 7.91 |
| 50% (Median) | 28.00 | 14.45 |
| 75% | 38.00 | 31.00 |
| Max | 80.00 | 512.33 |

#### Step 6: Cek Missing Values

| Kolom | Missing |
|-------|---------|
| Age | 177 |
| Cabin | 687 |
| Embarked | 2 |
| Lainnya | 0 |

#### Step 7: Cek Duplikasi
Tidak ada data duplikat (0).

#### Step 8: Imputation (Mengisi Missing Values)

```python
# Numerik: isi dengan median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Kategorikal: isi dengan modus (nilai paling sering)
df['Cabin'].fillna(df['Cabin'].value_counts().index[0], inplace=True)
df['Embarked'].fillna(df['Embarked'].value_counts().index[0], inplace=True)
```

Setelah imputation, semua missing values = 0.

#### Step 9: Feature Selection
Menghapus kolom yang tidak relevan: PassengerId, Name, Cabin, Ticket.

**Fitur (X):** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
**Target (y):** Survived

#### Step 10: Encoding

**One-Hot Encoding (untuk input features):**
```python
df_onehot = pd.get_dummies(df_X, columns=['Sex','Embarked'], drop_first=True)
```

> **📊 Deskripsi Hasil One-Hot:**
> Kolom baru: Sex_male, Embarked_Q, Embarked_S
> - Sex_male: 1 jika laki-laki, 0 jika perempuan
> - Embarked_Q: 1 jika pelabuhan Q, 0 jika bukan
> - Embarked_S: 1 jika pelabuhan S, 0 jika bukan

**Label Encoding (untuk input features):**
```python
from sklearn.preprocessing import LabelEncoder
# Mengubah kategori menjadi angka: 0, 1, 2, ...
```

**Label Encoding (untuk target class):**
```python
le = LabelEncoder()
df_y = le.fit_transform(df_y)  # 0 = Not Survived, 1 = Survived
```

#### Step 11: Korelasi Setelah Encoding

> **📊 Deskripsi Heatmap Korelasi Baru:**
> - **Survived vs Sex:** -0.54 (korelasi negatif kuat — perempuan lebih mungkin selamat)
> - **Survived vs Pclass:** -0.34
> - **Survived vs Fare:** +0.26
> - **Sex vs Fare:** -0.18
> - **Pclass vs Fare:** -0.55

#### Step 12: Train-Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df_X, df_y, test_size=0.3, random_state=42
)
```

- **Training set:** 70% data (623 baris)
- **Test set:** 30% data (268 baris)

#### Step 13: Standardization

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

> **📊 Deskripsi Hasil Standardization:**
> Data berubah menjadi nilai dengan mean ≈ 0 dan std ≈ 1.
> Contoh: Pclass = [-1.64, 0.80, 0.80, ...]

#### Step 14: Normalization

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

> **📊 Deskripsi Hasil Normalization:**
> Semua nilai berada dalam rentang [0, 1].
> Contoh: Pclass = [0.0, 1.0, 1.0, ...]

---

## 📌 KESIMPULAN

### Alur Kerja Data Science yang Lengkap:

```
1. Understanding Data (EDA)
   ↓
2. Data Preprocessing
   - Data Cleaning (missing values, outliers, duplicates)
   - Feature Engineering (encoding, transformation)
   - Scaling (normalization/standardization)
   ↓
3. Train-Test Split
   ↓
4. Modeling (Machine Learning)
   ↓
5. Evaluation
   ↓
6. Deployment
```

### Poin Kunci:
1. **AI** adalah bidang luas; **ML** adalah subset-nya; **DL** adalah subset dari ML.
2. **Data berkualitas** adalah fondasi model ML yang baik.
3. **EDA** membantu memahami data sebelum pemodelan.
4. **Preprocessing** (cleaning, encoding, scaling) wajib dilakukan sebelum training.
5. **Train-test split** penting untuk evaluasi generalisasi model.
6. **Bias-variance tradeoff** membantu memilih kompleksitas model yang optimal.

---

Semoga rangkuman ini membantu Anda memahami seluruh materi dengan lebih terstruktur! 🚀