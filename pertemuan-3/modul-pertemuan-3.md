# Modul Praktikum Penambangan Data – Teknologi Rekayasa Perangkat Lunak – 2025
## PERTEMUAN III: Classification

### 3.1. TUJUAN PEMBELAJARAN
A. Mahasiswa mampu memahami konsep dasar beberapa algoritma klasifikasi, seperti Logistic Regression, KNN, Decision Tree, dan Random Forest.
B. Mahasiswa mengetahui fungsi dan peran metrik evaluasi klasifikasi, seperti *accuracy*, *precision*, dan *recall* dalam menilai performa model.
C. Mahasiswa dapat menerapkan berbagai algoritma klasifikasi untuk membangun model.
D. Mahasiswa mampu melakukan evaluasi model dengan menyusun tabel performa berisi *accuracy*, *precision*, dan *recall* dari setiap model.

---

### 3.2. DASAR TEORI

**• Classification**
Klasifikasi merupakan salah satu teknik *supervised learning* yang bertujuan untuk memprediksi label atau kategori dari sebuah sampel berdasarkan fitur yang dimiliki. Model klasifikasi belajar dari data berlabel, kemudian menghasilkan suatu fungsi yang mampu menentukan kelas dari data baru yang belum pernah dilihat. Klasifikasi bekerja dengan cara mencari pola dari data historis untuk menentukan keputusan kelas yang paling mendekati. Klasifikasi umum digunakan dalam aplikasi seperti deteksi spam, diagnosis medis, hingga rekomendasi konten.

**• Algoritma Classification**
Ada banyak metode yang bisa dipakai untuk melakukan klasifikasi. Beberapa algoritma di bawah ini termasuk yang paling sering digunakan dan dijelaskan dalam dokumentasi Scikit-Learn dan TensorFlow karena sederhana, efektif, dan cocok untuk berbagai jenis dataset.

1. **Logistic Regression**
   Logistic Regression merupakan algoritma klasifikasi berbasis regresi linier yang memodelkan probabilitas suatu sampel termasuk ke dalam kelas tertentu menggunakan fungsi sigmoid. Output model berada dalam rentang 0–1, yang kemudian dikonversi menjadi label kelas. Meskipun sederhana, algoritma ini efisien dan bekerja baik pada data yang memiliki hubungan *linear separable*.
2. **K-Nearest Neighbors (KNN)**
   K-Nearest Neighbors mengklasifikasikan suatu data baru berdasarkan mayoritas kelas dari *k* tetangga terdekatnya. Kedekatan dihitung menggunakan metrik jarak seperti *Euclidean distance*. Algoritma ini bersifat *non-parametric* dan sangat bergantung pada struktur distribusi data serta nilai *k* yang dipilih.
3. **Decision Tree**
   Decision Tree bekerja dengan cara membagi dataset secara rekursif berdasarkan fitur yang memberikan informasi terbaik. Proses pemilihan fitur dilakukan menggunakan ukuran seperti *Gini impurity* atau *entropy/information gain*. Kelebihannya adalah interpretasi yang mudah, tetapi mudah mengalami *overfitting* tanpa pengaturan parameter yang tepat.
4. **Random Forest**
   Random Forest adalah algoritma *ensemble* yang menggabungkan banyak Decision Tree melalui teknik *bagging*. Setiap pohon dilatih pada subset data yang berbeda, dan hasil akhirnya ditentukan melalui voting. Random Forest membuat model lebih stabil, mengurangi *overfitting*, dan meningkatkan akurasi dibanding satu pohon tunggal.
5. **AdaBoost (Adaptive Boosting)**
   AdaBoost bekerja dengan menggabungkan beberapa *weak learner* (biasanya Decision Tree kecil) secara berurutan. Pada setiap iterasi, bobot sampel diperbarui sehingga model berikutnya lebih fokus pada data yang sebelumnya salah diklasifikasikan. Dengan cara ini, AdaBoost mampu meningkatkan performa model secara signifikan terutama pada dataset yang kompleks.

**• Metrik Evaluasi Classification**
Dalam *machine learning* khususnya pada model klasifikasi, diperlukan metrik evaluasi untuk menilai seberapa baik model mampu memprediksi label dengan benar. Evaluasi ini dilakukan dengan membandingkan hasil prediksi model terhadap nilai aktual melalui sebuah struktur yang disebut *confusion matrix*.

1. **Confusion Matrix**
   *Confusion matrix* terdiri dari empat komponen penting, antara lain *True Positive* (TP) yaitu prediksi positif yang benar, *True Negative* (TN) yaitu prediksi negatif yang benar, *False Positive* (FP) yaitu prediksi positif yang salah, dan *False Negative* (FN) yaitu prediksi negatif yang salah. Berdasarkan nilai-nilai ini, beberapa metrik evaluasi dapat dihitung untuk memahami performa model secara lebih komprehensif.

2. **Accuracy, Precision, Recall, F1 Score**
   **Accuracy** merupakan metrik paling dasar yang menunjukkan proporsi prediksi yang benar dari seluruh prediksi yang dibuat oleh model. Rumusnya didefinisikan sebagai berikut:
   $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

   **Precision** mengukur tingkat ketepatan model dalam melakukan prediksi positif, yaitu berapa banyak prediksi positif yang benar-benar positif. Precision dihitung dengan rumus berikut:
   $$Precision = \frac{TP}{TP + FP}$$

   Sementara itu, **Recall** mengukur kemampuan model dalam menemukan seluruh sampel yang benar-benar positif. Recall dihitung menggunakan rumus:
   $$Recall = \frac{TP}{TP + FN}$$

   Karena *precision* dan *recall* memiliki fokus yang berbeda, **F1-Score** digunakan sebagai metrik yang menyeimbangkan keduanya. Metrik ini sangat berguna ketika dataset bersifat *imbalanced* dan diperlukan keseimbangan antara kemampuan model menghindari *false positives* serta *false negatives*. F1-score merupakan *harmonic mean* dari *precision* dan *recall*, dengan rumus:
   $$F1\ Score = \frac{2 \times Precision \times Recall}{Precision + Recall}$$

---

### 3.3. ALAT DAN BAHAN
**• Perangkat Keras**
* Komputer/Laptop

**• Perangkat Lunak**
* Python 3
* Google Colaboratory

---

### 3.4. LANGKAH PERCOBAAN

#### 1. Impor Pustaka
> **[Deskripsi Gambar 3.4.1: Impor Pustaka]**
> *Gambar ini menampilkan potongan kode Python (code snippet) yang berisi serangkaian pernyataan `import`. Kode tersebut memuat impor pustaka `pandas` untuk manipulasi dataframe, `numpy` untuk operasi array, `matplotlib.pyplot` dan `seaborn` untuk visualisasi data, serta berbagai modul dari `sklearn` (seperti `train_test_split`, `StandardScaler`, `LabelEncoder`, `SimpleImputer`, dan model-model klasifikasi beserta `classification_report`, `confusion_matrix`, `accuracy_score`, dll).*

#### 2. Load dataset
Dataset dapat diunduh pada link berikut: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

```python
df = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/refs/heads/main/healthcare-dataset-stroke-data.csv')
```

#### 3. Exploratory Data Analysis (EDA)
**a. Melihat lima baris pertama dan terakhir pada dataset dengan `df.head()` dan `df.tail()`.**

> **[Deskripsi Gambar 3.4.2: Lima Baris Pertama Dataset]**
> *Gambar ini menampilkan output tabel dari perintah `df.head()`. Tabel menunjukkan 5 baris teratas dari dataset dengan kolom-kolom: `id`, `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, dan `stroke`.*

> **[Deskripsi Gambar 3.4.3: Lima Baris Terakhir Dataset]**
> *Gambar ini menampilkan output tabel dari perintah `df.tail()`. Tabel menunjukkan 5 baris paling bawah dari dataset yang memiliki struktur kolom yang sama dengan gambar sebelumnya, memberikan gambaran tentang variasi data di akhir kumpulan dataset.*

**b. Melihat deskripsi statistik dengan `df.describe()`**

> **[Deskripsi Gambar 3.4.4: Hasil Deskripsi Statistik Dataset]**
> *Gambar ini menampilkan output tabel statistik deskriptif. Tabel berisi baris metrik (count, mean, std, min, 25%, 50%, 75%, max) dan kolom berupa variabel numerik seperti `id`, `age`, `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`, dan `stroke`. Nilai mean dan standar deviasi memberikan gambaran sebaran data numerik.*

**c. Melihat presentase target kelas**

> **[Deskripsi Gambar 3.4.5: Presentase Target Kelas]**
> *Gambar ini menampilkan sebuah Pie Chart (diagram lingkaran) yang memvisualisasikan distribusi variabel target `stroke`. Terdapat dua irisan utama: irisan terbesar berwarna dominan menunjukkan kelas 0 (tidak mengalami stroke) dengan proporsi sekitar 95.13%, dan irisan yang jauh lebih kecil menunjukkan kelas 1 (mengalami stroke) dengan proporsi 4.87%. Hal ini menandakan bahwa dataset bersifat imbalanced.*

**d. Melihat histogram**

> **[Deskripsi Gambar 3.4.6: Kode untuk Menampilkan Histogram]**
> *Gambar ini menampilkan blok kode Python yang menggunakan perulangan (loop) atau fungsi bawaan pandas/matplotlib untuk memplot histogram bagi setiap kolom bertipe numerik dalam dataset.*

> **[Deskripsi Gambar 3.4.7: Histogram Seluruh Variabel Numerik]**
> *Gambar ini menampilkan grid yang berisi beberapa subplot histogram. Setiap subplot merepresentasikan distribusi frekuensi dari variabel numerik seperti `age`, `avg_glucose_level`, `bmi`, `hypertension`, dan `heart_disease`. Bentuk histogram membantu mengidentifikasi skewness dan outlier pada data.*

**e. Mengecek missing value dengan `df.isnull().sum()`**

> **[Deskripsi Gambar 3.4.8: Pengecekan Missing Value]**
> *Gambar ini menampilkan output teks dari perintah `df.isnull().sum()`. Output tersebut mendaftar seluruh kolom beserta jumlah nilai kosongnya. Hasil menunjukkan bahwa hampir semua kolom memiliki 0 nilai kosong, kecuali kolom `bmi` yang memiliki 201 missing values.*

**f. Mengecek atribut kategorikal**

> **[Deskripsi Gambar 3.4.9: Pengecekan Atribut Kategorikal]**
> *Gambar ini menampilkan output dari perintah `df.select_dtypes(include=['object', 'bool']).columns` atau `df.info()`. Output menyoroti kolom-kolom yang bertipe data objek/kategorikal, yaitu: `gender`, `ever_married`, `work_type`, `Residence_type`, dan `smoking_status`, yang mengindikasikan bahwa kolom-kolom ini perlu dilakukan encoding.*

---

#### 4. Data Preprocessing

> **[Deskripsi Gambar 3.4.10: Kode untuk Melakukan Data Preprocessing]**
> *Gambar ini menampilkan blok kode Python yang panjang dan terstruktur untuk tahap preprocessing. Kode tersebut mencakup:*
> 1. *Pemisahan fitur (`X`) dan target (`y`).*
> 2. *Label encoding untuk variabel target `y`.*
> 3. *Imputasi nilai hilang pada kolom `bmi` menggunakan `SimpleImputer(strategy='median')`.*
> 4. *Categorical encoding menggunakan `LabelEncoder` atau `OneHotEncoder` untuk kolom bertipe objek.*
> 5. *Konversi `X` dan `y` menjadi numpy array bertipe float.*
> 6. *Pembagian data training dan testing (70:30) menggunakan `train_test_split`.*
> 7. *Normalisasi data menggunakan `StandardScaler` yang di-fit hanya pada `X_train` lalu di-transformasikan ke `X_train` dan `X_test`.*

**Pemeriksaan hasil preprocessing:**

**a. Variabel input X**
> **[Deskripsi Gambar 3.4.11: Variabel X Setelah Dikonversi Menjadi Numerik]**
> *Gambar ini menampilkan cuplikan array atau dataframe dari variabel `X`. Seluruh nilai yang sebelumnya berupa teks/kategori (seperti 'Male', 'Yes', 'Private') kini telah berhasil dikonversi menjadi angka-angka numerik (misal: 0, 1, 2).*

**b. Variabel target y**
> **[Deskripsi Gambar 3.4.12: Variabel y Setelah Proses Label Encoding]**
> *Gambar ini menampilkan output array 1-dimensi dari variabel `y`. Nilai di dalamnya hanya terdiri dari angka biner `0` dan `1` yang merepresentasikan kelas target (tidak stroke dan stroke).*

**c. Data training yang telah dinormalisasi (X_train)**
> **[Deskripsi Gambar 3.4.13: Hasil Transformasi pada Data Training]**
> *Gambar ini menampilkan tabel/array dari `X_train`. Nilai-nilai di dalamnya kini berada dalam skala terstandarisasi (angka desimal positif dan negatif yang berpusat di sekitar 0), menunjukkan bahwa `StandardScaler` telah berhasil diterapkan.*

**d. Data testing yang telah dinormalisasi (X_test)**
> **[Deskripsi Gambar 3.4.14: Hasil Transformasi pada Data Testing]**
> *Gambar ini menampilkan tabel/array dari `X_test`. Nilainya juga berupa angka desimal terstandarisasi yang skalanya konsisten dengan `X_train`, membuktikan bahwa scaler yang sama digunakan tanpa terjadi data leakage.*

---

#### 5. Modelling

**a. Logistic Regression**

> **[Deskripsi Gambar 3.4.15: Kode untuk Logistic Regression]**
> *Gambar menampilkan inisialisasi model `LogisticRegression()`, pemanggilan method `.fit(X_train, y_train)` untuk melatih model, dan `.predict(X_test)` untuk menghasilkan prediksi yang disimpan dalam variabel `y_pred`.*

* **Melatih model:** Pada tahap ini, objek `LogisticRegression()` dibuat sebagai model klasifikasi. Model kemudian dilatih menggunakan data training (`X_train` dan `y_train`) agar dapat mempelajari hubungan antara fitur dan label target.
* **Membuat prediksi:** Model yang telah dilatih digunakan untuk memprediksi kelas dari data testing (`X_test`). Hasil prediksi disimpan pada variabel `y_pred`.
* **Mengukur performa model:** Performa model diukur dengan metrik *accuracy*, *precision*, *recall*, dan *confusion matrix*. Penggunaan parameter `average='macro'` memastikan bahwa setiap kelas memiliki kontribusi yang sama.

> **[Deskripsi Gambar 3.4.16: Output Performa Model]**
> *Gambar menampilkan output teks dari `classification_report` atau print statement manual yang memuat nilai numerik untuk Accuracy, Precision, Recall, dan Confusion Matrix (TP, TN, FP, FN) khusus untuk model Logistic Regression.*

> **[Deskripsi Gambar 3.4.17: Confusion Matrix]**
> *Gambar menampilkan visualisasi heatmap dari Confusion Matrix. Sumbu X mewakili prediksi model dan sumbu Y mewakili nilai aktual. Warna sel (misalnya hijau untuk nilai tinggi, merah/kuning untuk nilai rendah) memudahkan interpretasi di mana model sering melakukan kesalahan klasifikasi (FP atau FN).*

* **Menghitung F1 score:** F1-score merupakan kombinasi harmonis dari *precision* dan *recall*.

> **[Deskripsi Gambar 3.4.18: Output F1 Score]**
> *Gambar menampilkan output berupa angka desimal (misal: 0.15 atau sejenisnya) yang merupakan hasil perhitungan F1-Score dari model Logistic Regression.*

* **Model coefficient & intercept:**

> **[Deskripsi Gambar 3.4.19: Pengecekan Nilai Koefisien dari Model Logistic Regression]**
> *Gambar menampilkan output dari `model.coef_`. Berupa array numerik yang merepresentasikan bobot/koefisien untuk setiap fitur. Nilai positif berarti fitur meningkatkan peluang kelas 1, nilai negatif berarti menurunkan.*

> **[Deskripsi Gambar 3.4.20: Pengecekan Nilai Konstanta pada Model Logistic Regression]**
> *Gambar menampilkan output dari `model.intercept_`. Berupa sebuah angka skalar negatif, yang menunjukkan bias awal model ketika semua fitur bernilai nol.*

**b. K-Nearest Neighbors (KNN)**

> **[Deskripsi Gambar 3.4.21: Kode untuk Melatih Model KNN]**
> *Gambar menampilkan kode inisialisasi `KNeighborsClassifier(n_neighbors=10)`, proses `.fit()`, dan `.predict()`.*

* **Melatih model:** Model KNN dibentuk dengan menetapkan jumlah tetangga (*k*) sebanyak 10. Data training digunakan untuk menyimpan representasi data.
* **Membuat prediksi:** Proses prediksi dilakukan dengan menghitung jarak antara sampel testing dan seluruh sampel training, kemudian memilih kelas mayoritas dari 10 tetangga terdekat.

> **[Deskripsi Gambar 3.4.22: Output Performa Model (KNN)]**
> *Gambar menampilkan metrik evaluasi (Accuracy, Precision, Recall) untuk model KNN.*

> **[Deskripsi Gambar 3.4.23: Confusion Matrix (KNN)]**
> *Gambar menampilkan heatmap Confusion Matrix untuk model KNN, menunjukkan distribusi TP, TN, FP, dan FN.*

**c. Decision Tree**

> **[Deskripsi Gambar 3.4.24: Kode untuk Melatih Model Decision Tree]**
> *Gambar menampilkan kode inisialisasi `DecisionTreeClassifier(criterion='entropy')`, proses `.fit()`, dan `.predict()`.*

* **Melatih model:** Model Decision Tree dibentuk menggunakan kriteria pemilihan split *entropy*. Data training digunakan untuk membangun struktur pohon keputusan.
* **Membuat prediksi:** Prediksi dilakukan dengan menelusuri jalur pada pohon keputusan, mulai dari root hingga mencapai node daun.

> **[Deskripsi Gambar 3.4.25: Output Performa Model (Decision Tree)]**
> *Gambar menampilkan metrik evaluasi (Accuracy, Precision, Recall) untuk model Decision Tree.*

> **[Deskripsi Gambar 3.4.26: Confusion Matrix (Decision Tree)]**
> *Gambar menampilkan heatmap Confusion Matrix untuk model Decision Tree.*

**d. Random Forest**

> **[Deskripsi Gambar 3.4.27: Kode untuk Melatih Model Random Forest]**
> *Gambar menampilkan kode inisialisasi `RandomForestClassifier()` dengan parameter default, proses `.fit()`, dan `.predict()`.*

* **Melatih model:** Model Random Forest dibentuk dengan parameter default yang secara otomatis membuat sejumlah pohon keputusan. Algoritma ini bekerja berdasarkan prinsip *bagging*.
* **Membuat prediksi:** Prediksi akhir ditentukan melalui proses voting mayoritas dari seluruh pohon yang ada dalam ensemble.

> **[Deskripsi Gambar 3.4.28: Output Performa Model (Random Forest)]**
> *Gambar menampilkan metrik evaluasi (Accuracy, Precision, Recall) untuk model Random Forest.*

> **[Deskripsi Gambar 3.4.29: Confusion Matrix (Random Forest)]**
> *Gambar menampilkan heatmap Confusion Matrix untuk model Random Forest.*

**e. Ada Boost**

> **[Deskripsi Gambar 3.4.30: Kode untuk Melatih Model Ada Boost]**
> *Gambar menampilkan kode inisialisasi `AdaBoostClassifier()` dengan parameter default, proses `.fit()`, dan `.predict()`.*

* **Melatih model:** Model AdaBoost dibentuk menggunakan parameter default, di mana algoritma akan melatih beberapa *weak learners* secara berurutan. Setiap *weak learner* diberikan bobot berdasarkan tingkat kesalahannya.
* **Melakukan prediksi:** Prediksi akhir merupakan kombinasi berbobot dari seluruh *weak learners* yang telah dilatih sebelumnya.

> **[Deskripsi Gambar 3.4.31: Output Performa Model (AdaBoost)]**
> *Gambar menampilkan metrik evaluasi (Accuracy, Precision, Recall) untuk model AdaBoost.*

> **[Deskripsi Gambar 3.4.32: Confusion Matrix (AdaBoost)]**
> *Gambar menampilkan heatmap Confusion Matrix untuk model AdaBoost.*

---

### 3.5. TUGAS & ANALISIS
**Dataset:** https://www.kaggle.com/datasets/bhavikjikadara/loan-status-prediction

1. Jelaskan apa tujuan penggunaan dataset ini?
2. Definisikan atribut yang menjadi input dan output nya?
3. Silahkan membuat model klasifikasi logistic regression, KNN, Decision Tree, dan Random Forest untuk kasus dataset diatas!
4. Buatlah tabel yang menjelaskan performa dari model machine learning untuk kasus dataset diatas! Kolom pertama "model", kolom selanjutnya "accuracy, precision, recall".
5. Jelaskan dari hasil eksperimen di atas, model mana yang paling baik? Jelaskan alasan anda!

---

### 3.6. REFERENSI
* https://www.python.org/doc/
* https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
* https://numpy.org/
* Han, J., Pei, J., & Tong, H. (2022). *Data Mining: Concepts and Techniques* (4th ed.). Morgan Kaufmann/Elsevier.

---

# Classification Models

Model klasifikasi dalam *machine learning* adalah algoritma yang digunakan untuk mengklasifikasikan data ke dalam kategori atau kelas tertentu berdasarkan fitur-fitur yang diberikan. Tujuan utama dari model klasifikasi adalah untuk mempelajari pola dari data latih sehingga dapat memprediksi kelas atau label dari data baru yang diberikan. Berbagai algoritma klasifikasi seperti *Decision Trees*, *Random Forests*, *Logistic Regression*, *KNN*, *AdaBoost*, dan *Neural Networks* digunakan tergantung pada karakteristik data dan kebutuhan spesifik dari permasalahan yang dihadapi. Evaluasi kinerja model klasifikasi biasanya dilakukan dengan menggunakan metrik seperti akurasi (*accuracy*), presisi (*precision*), *recall*, dan *F1-score*.

---

## Tugas 1
1. Download dataset di https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset dan simpan di Google Drive Anda atau repositori lain, misalnya GitHub Anda.
2. Jelaskan apa tujuan dari penggunaan dataset ini?
3. Definisikan mana input dan output-nya.

---

## Read the Dataset

```python
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/main/healthcare-dataset-stroke-data.csv')
df.head()
df.tail()
```

> **[Deskripsi Gambar 1: Lima Baris Pertama Dataset (`df.head()`)]**
> *Gambar ini menampilkan output tabel yang berisi 5 baris teratas dari dataset. Kolom yang ditampilkan meliputi: `id`, `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, dan `stroke`. Data menunjukkan variasi nilai, termasuk nilai kosong (`NaN`) pada kolom `bmi` di baris kedua.*

> **[Deskripsi Gambar 2: Lima Baris Terakhir Dataset (`df.tail()`)]**
> *Gambar ini menampilkan output tabel yang berisi 5 baris paling bawah dari dataset (indeks 5105 hingga 5109). Struktur kolomnya sama dengan gambar sebelumnya, memberikan gambaran tentang variasi data di akhir kumpulan dataset, termasuk beberapa nilai `NaN` pada kolom `bmi` dan status merokok yang terpotong (*truncated*) seperti "never smo" atau "Unkn".*

---

## Check Descriptive Statistics

```python
# untuk melihat descriptive statistics
df.describe()
```

> **[Deskripsi Gambar 3: Hasil Deskripsi Statistik Dataset (`df.describe()`)]**
> *Gambar ini menampilkan tabel statistik deskriptif untuk variabel numerik. Baris berisi metrik: `count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, dan `max`. Kolom mencakup `id`, `age`, `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`, dan `stroke`. Terlihat bahwa `count` untuk `bmi` adalah 4909.000000 (lebih rendah dari total 5110), mengonfirmasi adanya data yang hilang. Rata-rata usia (`age`) adalah 43.22, dan rata-rata `stroke` adalah 0.048 (sekitar 4.8%).*

---

## Check Percentage of Target Class

```python
import matplotlib.pyplot as plt
data = df['stroke'].value_counts()
data.plot(kind='pie', autopct='%.2f%%')
plt.show()
# apakah distribusi kelas seimbang?
```

> **[Deskripsi Gambar 4: Presentase Target Kelas (Pie Chart)]**
> *Gambar ini menampilkan diagram lingkaran (*pie chart*) yang memvisualisasikan distribusi variabel target `stroke`. Irisan terbesar (sekitar 95.13%) mewakili kelas 0 (tidak mengalami stroke), sedangkan irisan yang sangat kecil (4.87%) mewakili kelas 1 (mengalami stroke). Visualisasi ini dengan jelas menunjukkan bahwa dataset bersifat tidak seimbang (*imbalanced*).*

---

## Check Histogram

```python
# check histogram for continuous columns
df.hist(figsize=(10,10))
plt.show()
```

> **[Deskripsi Gambar 5: Histogram Seluruh Variabel Numerik]**
> *Gambar ini menampilkan grid yang berisi beberapa subplot histogram untuk setiap kolom numerik dalam dataset (`id`, `age`, `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`, `stroke`). Histogram `age` menunjukkan distribusi yang agak normal dengan sedikit skew ke kanan. Histogram `avg_glucose_level` dan `bmi` menunjukkan distribusi yang *right-skewed* (condong ke kanan), mengindikasikan adanya outlier atau nilai ekstrem pada data.*

---

## Check Missing Values

```python
df.isnull().sum()
# kolom bmi ada yg kosong
```

> **[Deskripsi Gambar 6: Pengecekan Missing Value]**
> *Gambar ini menampilkan output teks dari perintah `df.isnull().sum()`. Output ini mendaftar seluruh kolom beserta jumlah nilai kosongnya. Hasilnya menunjukkan bahwa semua kolom memiliki 0 nilai kosong, kecuali kolom `bmi` yang memiliki tepat 201 nilai kosong (*missing values*).*

---

## Check Categorical Attributes

```python
df_X = df.drop(['id', 'stroke'], axis=1)
# definisikan kolom yg jadi input
df_y = df[['stroke']]
# definisikan kolom yg jadi output
cats = df_X.select_dtypes(include=['object', 'bool']).columns
print(cats)
```

> **[Deskripsi Gambar 7: Pengecekan Atribut Kategorikal]**
> *Gambar ini menampilkan output dari variabel `cats`, yaitu sebuah Index yang berisi nama-nama kolom bertipe data objek atau boolean. Kolom-kolom tersebut adalah: `'gender'`, `'ever_married'`, `'work_type'`, `'Residence_type'`, dan `'smoking_status'`. Ini mengidentifikasi kolom-kolom yang memerlukan proses encoding sebelum pemodelan.*

---

## Data Preprocessing

```python
# import library yg dibutuhkan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.metrics import sensitivity_specificity_support
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# data preprocessing dimulai
# membuat X and y. X untuk input variable, y untuk target class
df_X = df.drop(['id', 'stroke'], axis=1)
df_y = df[['stroke']]

# label encoding for y.
# merubah nilai yg ada di y menjadi 0 atau 1.
# sebenarnya ini tidak diperlukan karena nilai y di dataset sudah 0 atau 1
le = LabelEncoder()
df_y = le.fit_transform(df_y['stroke'])

# imputation. kita isi nilai kosong yg di kolom bmi dengan nilai median nya (atau bisa pakai cara lain)
df_X['bmi'].fillna(df_X['bmi'].median(), inplace=True)

# categorical encoding
# merubah categorical value menjadi numerical value
# bisa pakai label encoding, ordinal atau one hot encoding
cats = df_X.select_dtypes(include=['object', 'bool']).columns
cat_features = list(cats.values)
le = LabelEncoder()
for i in cat_features:
    df_X[i] = le.fit_transform(df_X[i])

# menyimpan X dan y menjadi numpy arrays
X = df_X.astype(float).values
y = df_y.astype(float)

# hold-out method, dibagi menjadi training dan testing set. 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scaling
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# data preprocessing selesai
```

> **[Deskripsi Gambar 8: Variabel X Setelah Dikonversi Menjadi Numerik]**
> *Gambar ini menampilkan cuplikan output array NumPy dari variabel `X`. Seluruh nilai yang sebelumnya berupa teks (seperti 'Male', 'Yes', 'Private') kini telah berhasil dikonversi menjadi angka-angka numerik (misalnya: 1.0, 0.0, 67.0, 228.69, 36.6, 1.0).*

> **[Deskripsi Gambar 9: Variabel y Setelah Proses Label Encoding]**
> *Gambar ini menampilkan output array 1-dimensi dari variabel `y`. Nilai di dalamnya hanya terdiri dari angka biner `1.` dan `0.` yang merepresentasikan kelas target (stroke dan tidak stroke).*

> **[Deskripsi Gambar 10: Hasil Transformasi pada Data Training (`X_train`)]**
> *Gambar ini menampilkan cuplikan array dari `X_train`. Nilai-nilai di dalamnya kini berupa angka desimal positif dan negatif (misalnya: 1.184, -1.746, -0.317), menunjukkan bahwa `StandardScaler` telah berhasil diterapkan sehingga data memiliki rata-rata 0 dan standar deviasi 1.*

> **[Deskripsi Gambar 11: Hasil Transformasi pada Data Testing (`X_test`)]**
> *Gambar ini menampilkan cuplikan array dari `X_test`. Nilainya juga berupa angka desimal terstandarisasi yang skalanya konsisten dengan `X_train`, membuktikan bahwa scaler yang sama digunakan tanpa terjadi kebocoran data (*data leakage*).*

---

## Modelling

### a. Logistic Regression

```python
# mulai melakukan modelling. model ML learning/ belajar dari training set
import numpy as np
model = LogisticRegression()
model.fit(X_train, y_train)

# membuat prediksi
y_pred = model.predict(X_test)
# y_pred jika di print out keluar vector prediksi nya

# menghitung performa model, dengan accuracy dll
print('Accuracy ', accuracy_score(y_test, y_pred))
print('Precision ', precision_score(y_test, y_pred, average='macro'))
print('Recall ', recall_score(y_test, y_pred, average='macro'))
print('Confusion matrix ', confusion_matrix(y_test, y_pred))

# plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
# plt.show()
disp.plot(cmap=plt.cm.Blues)

from sklearn.metrics import f1_score
print('F1 ', f1_score(y_test, y_pred, average='macro'))
```

> **[Deskripsi Gambar 12: Output Performa Model Logistic Regression]**
> *Gambar ini menampilkan output teks hasil evaluasi model:*
> - *Accuracy: 0.9419439008480104*
> - *Precision: 0.4709719504240052*
> - *Recall: 0.5*
> - *Confusion matrix: [[1444, 0], [89, 0]]*
> - *F1: 0.485052065838092*
> *Terdapat juga peringatan (`UndefinedMetricWarning`) yang menyatakan bahwa Precision ditetapkan menjadi 0.0 pada label dengan tidak ada sampel yang diprediksi (karena model memprediksi semua sebagai kelas 0).*

> **[Deskripsi Gambar 13: Visualisasi Confusion Matrix Logistic Regression]**
> *Gambar ini menampilkan heatmap Confusion Matrix. Sumbu X adalah predicted label dan sumbu Y adalah true label. Sel (0,0) berwarna biru tua (nilai 1444, True Negative), sel (1,0) berwarna lebih terang (nilai 89, False Negative), sedangkan sel (0,1) dan (1,1) berwarna sangat muda/putih (nilai 0, False Positive dan True Positive), mengonfirmasi model gagal memprediksi kelas positif sama sekali.*

**Melihat coefficients-nya:**
```python
model.coef_
```
> **[Deskripsi Gambar 14: Pengecekan Nilai Koefisien dari Model Logistic Regression]**
> *Gambar menampilkan output array `model.coef_`: `[[-0.01926181, 1.5514967, 0.10348619, 0.07940788, -0.18090425, -0.06958962, 0.05980281, 0.1976799, -0.03397277, 0.02276299]]`. Ini menunjukkan bobot yang dipelajari model untuk setiap fitur.*

**Melihat intercept-nya:**
```python
model.intercept_
```
> **[Deskripsi Gambar 15: Pengecekan Nilai Konstanta pada Model Logistic Regression]**
> *Gambar menampilkan output array `model.intercept_`: `[-4.04479423]`. Nilai konstanta negatif ini menunjukkan bias awal model.*

---

### b. K Nearest Neighbour (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train, y_train)

# membuat prediksi
y_pred = model.predict(X_test)

# menghitung performa model, dengan accuracy dll
print('Accuracy ', accuracy_score(y_test, y_pred))
print('Precision ', precision_score(y_test, y_pred, average='macro'))
print('Recall ', recall_score(y_test, y_pred, average='macro'))
print('Confusion matrix ', confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp.plot(cmap=plt.cm.Blues)
```

> **[Deskripsi Gambar 16: Output Performa Model KNN]**
> *Gambar ini menampilkan output teks hasil evaluasi model KNN:*
> - *Accuracy: 0.9419439008480104*
> - *Precision: 0.4709719504240052*
> - *Recall: 0.5*
> - *Confusion matrix: [[1444, 0], [89, 0]]*
> *Peringatan `UndefinedMetricWarning` yang sama juga muncul, menunjukkan model KNN dengan k=10 juga gagal memprediksi kelas minoritas (stroke).*

> **[Deskripsi Gambar 17: Visualisasi Confusion Matrix KNN]**
> *Gambar ini menampilkan heatmap Confusion Matrix untuk KNN, yang secara visual identik dengan Logistic Regression, menunjukkan 1444 True Negative dan 89 False Negative, tanpa ada prediksi True Positive.*

---

### c. Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)

# membuat prediksi
y_pred = model.predict(X_test)

# menghitung performa model, dengan accuracy dll
print('Accuracy ', accuracy_score(y_test, y_pred))
print('Precision ', precision_score(y_test, y_pred, average='macro'))
print('Recall ', recall_score(y_test, y_pred, average='macro'))
print('Confusion matrix ', confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp.plot(cmap=plt.cm.Blues)
```

> **[Deskripsi Gambar 18: Output Performa Model Decision Tree]**
> *Gambar ini menampilkan output teks hasil evaluasi model Decision Tree:*
> - *Accuracy: 0.9021526418786693*
> - *Precision: 0.5393685085168259*
> - *Recall: 0.5368670048865511*
> - *Confusion matrix: [[1372, 72], [78, 11]]*
> *Model ini mulai mampu memprediksi kelas positif, terlihat dari adanya 11 True Positive dan 72 False Positive.*

> **[Deskripsi Gambar 19: Visualisasi Confusion Matrix Decision Tree]**
> *Gambar ini menampilkan heatmap Confusion Matrix untuk Decision Tree. Sel (0,0) bernilai 1372, sel (0,1) bernilai 72, sel (1,0) bernilai 78, dan sel (1,1) bernilai 11. Warna pada sel (1,1) sedikit lebih gelap dibandingkan model sebelumnya, menandakan adanya deteksi kelas minoritas.*

---

### d. Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# membuat prediksi
y_pred = model.predict(X_test)

# menghitung performa model, dengan accuracy dll
print('Accuracy ', accuracy_score(y_test, y_pred))
print('Precision ', precision_score(y_test, y_pred, average='macro'))
print('Recall ', recall_score(y_test, y_pred, average='macro'))
print('Confusion matrix ', confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp.plot(cmap=plt.cm.Blues)
```

> **[Deskripsi Gambar 20: Output Performa Model Random Forest]**
> *Gambar ini menampilkan output teks hasil evaluasi model Random Forest:*
> - *Accuracy: 0.9412915851272016*
> - *Precision: 0.47095300261096606*
> - *Recall: 0.49965373961218834*
> - *Confusion matrix: [[1443, 1], [89, 0]]*

> **[Deskripsi Gambar 21: Visualisasi Confusion Matrix Random Forest]**
> *Gambar ini menampilkan heatmap Confusion Matrix untuk Random Forest. Model ini memprediksi 1 sampel sebagai True Positive (sel 0,1), namun masih gagal mendeteksi 89 sampel stroke lainnya (False Negative).*

---

### e. AdaBoost

```python
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(X_train, y_train)

# membuat prediksi
y_pred = model.predict(X_test)

# menghitung performa model, dengan accuracy dll
print('Accuracy ', accuracy_score(y_test, y_pred))
print('Precision ', precision_score(y_test, y_pred, average='macro'))
print('Recall ', recall_score(y_test, y_pred, average='macro'))
print('Confusion matrix ', confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp.plot(cmap=plt.cm.Blues)
```

> **[Deskripsi Gambar 22: Output Performa Model AdaBoost]**
> *Gambar ini menampilkan output teks hasil evaluasi model AdaBoost:*
> - *Accuracy: 0.9386823222439661*
> - *Precision: 0.5825678040244969*
> - *Recall: 0.5088121323414984*
> - *Confusion matrix: [[1437, 7], [87, 2]]*

> **[Deskripsi Gambar 23: Visualisasi Confusion Matrix AdaBoost]**
> *Gambar ini menampilkan heatmap Confusion Matrix untuk AdaBoost. Model ini berhasil memprediksi 2 True Positive dan 7 False Positive, menunjukkan performa yang sedikit lebih baik dalam mendeteksi kelas minoritas dibandingkan Random Forest, meskipun akurasinya sedikit lebih rendah.*

---

## Tugas 1 (Tambahan)

4. Dengan menggunakan dataset di bawah ini:  
   https://www.kaggle.com/datasets/bhavikjikadara/loan-status-prediction
5. Jelaskan apa tujuan penggunaan dataset ini?
6. Definisikan atribut yang menjadi input dan output-nya?
7. Silahkan membuat model klasifikasi *Logistic Regression*, *KNN*, *Decision Tree*, dan *Random Forest* untuk kasus dataset di atas!
8. Buatlah tabel yang menjelaskan performa dari model *machine learning* untuk kasus dataset di atas! Kolom pertama "model", kolom selanjutnya "accuracy, precision, recall".
9. Jelaskan dari hasil eksperimen di atas, model mana yang paling baik? Jelaskan alasan Anda!

*** 

# Apa yang Akan Kita Pelajari Hari Ini?

## Table of Content
1. KNN
2. Decision Tree
3. Ensemble Method
4. Logistic Regression
5. Performance metrics for classification
6. Validation techniques

---

## KNN

### Classification and Clustering Techniques
* **Classification**
  * K-Nearest Neighbour
  * Decision Tree
  * Ensemble Methods
* **Clustering**
  * K-medoid
  * K-means
  * DBSCAN

### Nearest Neighbor Classifiers
* **Ide dasar:**
  * Jika dia berjalan seperti kucing, mengeong seperti kucing, maka itu mungkin kucing

> **[Deskripsi Gambar: Alur Kerja Nearest Neighbor Classifiers]**
> *Diagram ini menampilkan alur kerja klasifikasi nearest neighbor. Di sisi kiri terdapat "Training Records" (data latih) yang tersimpan. Di tengah terdapat proses "Hitung jarak (distance)" antara data baru dengan data latih. Di sisi kanan terdapat "Test Record" (data uji) yang akan diklasifikasikan berdasarkan kedekatan jaraknya.*

### Nearest Neighbor Classifiers
**Unknown record**
* **Membutuhkan tiga hal:**
  * Kumpulan data yang tersimpan
  * Jarak Metrik (*distance metric*) untuk menghitung jarak antar data
  * Nilai *k*, jumlah tetangga terdekat (*nearest neighbour*) yang akan diambil
* **Untuk mengklasifikasi data baru:**
  * Hitung jarak terhadap data lain
  * Memilih *k* tetangga terdekat (*nearest neighbour*)
  * Gunakan label kelas dari tetangga terdekat untuk menentukan label kelas dari data baru (misalnya, dengan mengambil suara mayoritas)

### Definition of Nearest Neighbor

> **[Deskripsi Gambar: Definisi Nearest Neighbor]**
> *Gambar ini menampilkan tiga sub-diagram yang mendefinisikan konsep k-nearest neighbor:*
> * *(a) 1-nearest neighbor: Hanya satu titik terdekat yang digunakan untuk klasifikasi.*
> * *(b) 2-nearest neighbor: Dua titik terdekat yang digunakan.*
> * *(c) 3-nearest neighbor: Tiga titik terdekat yang digunakan.*
> * **K-nearest neighbor** dari record *x* adalah *k* data yang memiliki jarak terdekat ke *x*.

### Nearest Neighbor Classification
* **Hitung jarak antara dua titik:**
  * *Euclidean distance*
* **Memilih class dari tetangga terdekat:**
  * Ambil suara mayoritas dari label kelas di antara *k-nearest neighbour*
  * Memberi bobot suara menurut jarak (*distance*)

$$d(p, q) = \sqrt{\sum_{i} (p_i - q_i)^2}$$

$$y = \frac{\sum_{i=1}^{k} w_i \cdot y_i}{\sum w_i}, \quad w_i = \frac{1}{d(x_i, x)}$$

### k-NN - Example
* **Given 14 examples** → map to 4-D space
* **Classify unknown sample**
  * X: (age="<30", income="medium", student="yes", credit="fair") → (0, 0.5, 1, 0)
  * **3-NN:**
    * (0, 0, 1, 0): 1(yes), d= 0.5, w= 1/0.5
    * (0, 0.5, 0, 0): 0(no), d= 1.0, w= 1/1.0
    * (0, 0.5, 1, 1): 1(yes), d= 1.0, w= 1/1.0
  * W = 4
  * y = 2/4 x 1(yes) + 1/4 x 0(no) + 1/4 x 1(yes) = 0.75
  * **Classify X as "Yes"**

> **[Deskripsi Tabel: Dataset 14 Contoh untuk k-NN]**
> *Tabel ini berisi 14 baris data latih dengan kolom: No, age, income, student, credit_rating, dan buys_computer. Nilai-nilai telah dinormalisasi ke dalam rentang 0 hingga 1. Kolom `buys_computer` merupakan label kelas (0 = no, 1 = yes). Data ini digunakan untuk mengklasifikasikan sampel X yang tidak diketahui.*

### Nearest Neighbor Classification
* **Choosing the value of k:**
  * If *k* is too small, sensitive to noise points
  * If *k* is too large, neighborhood may include points from other classes

> **[Deskripsi Gambar: Pemilihan Nilai k]**
> *Diagram ini menampilkan sebuah ruang 2D dengan titik-titik data dari dua kelas berbeda (misalnya, bulatan dan segitiga). Terdapat sebuah titik "X" yang akan diklasifikasikan. Tiga lingkaran dengan radius berbeda digambarkan di sekitar X:*
> * *Lingkaran kecil (k=1): Hanya mencakup satu titik tetangga.*
> * *Lingkaran sedang (k=5): Mencakup lima titik tetangga.*
> * *Lingkaran besar (k=11): Mencakup sebelas titik tetangga, yang mungkin sudah termasuk titik dari kelas lain.*
> * *Ilustrasi ini menunjukkan trade-off antara sensitivitas terhadap noise (k kecil) dan inklusi titik dari kelas lain (k besar).*

### Nearest Neighbor Classification
* **Permasalahan Scaling**
  * Attributes perlu di-*scale*
    * Untuk mencegah jarak (*distance*) didominasi oleh salah satu atribut
  * **Contoh:**
    * tinggi seseorang dapat bervariasi dari 1,5 m hingga 1,8 m
    * berat seseorang dapat bervariasi dari 40kg hingga 150kg
    * pendapatan seseorang dapat bervariasi dari $10K hingga $1M

---

## Decision Tree

> **[Deskripsi Tabel: Training Dataset untuk Decision Tree]**
> *Tabel ini berisi 14 baris data latih dengan kolom: No, age (<=30, 31...40, >40), income (high, medium, low), student (yes, no), credit_rating (fair, excellent), dan buys_computer (yes, no). Data ini digunakan untuk membangun pohon keputusan yang akan memprediksi apakah seseorang akan membeli komputer.*

### Output: A Decision Tree for buys_computer

> **[Deskripsi Gambar: Decision Tree untuk buys_computer]**
> *Diagram pohon keputusan ini memiliki struktur hierarkis:*
> * **Root node:** "age?" dengan tiga cabang: "<=30", "30..40", dan ">40".*
> * **Cabang "<=30":** Menuju node "student?" dengan dua cabang: "no" (leaf: No) dan "yes" (leaf: Yes).*
> * **Cabang "30..40":** Langsung menuju leaf "Yes".*
> * **Cabang ">40":** Menuju node "credit rating?" dengan dua cabang: "fair" (leaf: No) dan "excellent" (leaf: Yes).*
> * *Terdapat juga jalur klasifikasi untuk sampel X: (age="<30", income="medium", student="yes", credit="fair") yang berakhir di leaf "Yes".*

### Algorithm for DT Induction
* **Top-down recursive divide-and-conquer manner**
* Pada awalnya, semua training data ada di root
* Attribute yang di node, dipilih berdasarkan heuristik atau statistik (mis., *information gain*)
* Training data dipartisi secara rekursif berdasarkan atribut yang dipilih
* **Kondisi untuk menghentikan partisi:**
  * Semua sampel termasuk dalam kelas yang sama
  * Tidak ada sampel yang tersisa
  * Tidak ada atribut yang tersisa

> **[Deskripsi Gambar: Algoritma Induksi Decision Tree]**
> *Diagram ini menunjukkan proses pembagian data secara rekursif. Root node berisi seluruh dataset (14 sampel). Data dibagi berdasarkan atribut "age" menjadi tiga subset: "<=30" (5 sampel), "30..40" (4 sampel), dan ">40" (5 sampel). Setiap subset kemudian dibagi lagi berdasarkan atribut berikutnya (student, credit_rating) hingga mencapai leaf node yang homogen.*

### Entropy
* Entropi dapat didefinisikan sebagai ukuran kemurnian sub split.

**Examples:**
* S: {Y(1,2), N(3,4)} for instances 1,2,3,4
  → E(S) = (-1/2)*log(1/2) + (-1/2)*log(1/2) = 1
* S: {Y(1,3,4), N(2)} for instances 1,2,3,4
  → E(S) = (-3/4)*log(3/4) + (-1/4)*log(1/4) = 0.81
* S: {Y(1,2,3,4), N()} for instances 1,2,3,4
  → E(S) = (-4/4)*log(4/4) + (-0/4)*log(0/4) = 0

$$E(S) = -\sum_{i=1}^{m} p_i \log_2(p_i)$$

### Information Gain (ID3/C4.5)
* Entropi yang diharapkan setelah memeriksa nilai atribut A
  = Rata-rata entropy dari S₁, S₂, ... Sₙ setelah partisi S menggunakan atribut A dengan nilai {a₁, a₂, ..., aᵥ}

$$E(A) = \sum_{j=1}^{v} \frac{|S_j|}{|S|} E(S_j)$$

* Hitung *information gain* dari attribute A

$$Gain(A) = E(S) - E(A)$$

* Pilih atribut dengan *Information gain* yang terbesar

### Attribute Selection by IG - Example
* C1: buys_computer = "yes", C2: buys_computer = "no"
* E(D) = E(9,5) = 0.940
* E(age) = 5/14 * E("<=30") + 4/14 * E("30..40") + 5/14 * E(">40") = 0.69
* Gain(age) = E(D) - E(age) = 0.25
* Gain(income) = 0.03, Gain(student) = 0.15

> **[Deskripsi Tabel: Perhitungan Information Gain untuk Atribut Age]**
> *Tabel ini menampilkan perhitungan entropy untuk setiap nilai atribut "age":*
> * *<=30: 2 kelas C1, 3 kelas C2, E(Si) = 0.971*
> * *30...40: 4 kelas C1, 0 kelas C2, E(Si) = 0*
> * *>40: 3 kelas C1, 2 kelas C2, E(Si) = 0.971*
> * *Perhitungan menunjukkan bahwa atribut "age" memiliki information gain sebesar 0.25, yang merupakan nilai tertinggi dibandingkan atribut lainnya.*

### Extracting Classification Rules
* Merepresentasikan pengetahuan dalam bentuk **IF-THEN rules**
  * Satu aturan (*rule*) dibuat untuk setiap jalur dari akar ke daun
  * Node daun menyimpan prediksi kelas
* Rules mudah dipahami manusia

**Example:**
* IF age = "<=30" AND student = "no" THEN buys_computer = "no"
* IF age = "<=30" AND student = "yes" THEN buys_computer = "yes"
* IF age = "31...40" THEN buys_computer = "yes"
* IF age = ">40" AND credit = "excellent" THEN buys_computer = "yes"
* IF age = ">40" AND credit = "fair" THEN buys_computer = "no"

> **[Deskripsi Gambar: Ekstraksi Aturan dari Decision Tree]**
> *Diagram ini menampilkan decision tree yang sama dengan sebelumnya, namun setiap jalur dari root ke leaf diberi label aturan IF-THEN. Misalnya, jalur dari root "age?" → "<=30" → "student?" → "no" → leaf "No" diberi label "IF age = '<=30' AND student = 'no' THEN buys_computer = 'no'".*

### Avoid Overfitting
* Tree yang dibuat mungkin akan *overfit* terhadap training data
  * Terlalu banyak cabang, diakibatkan oleh *outliers* data
  * Hasilnya akurasi yang rendah terhadap data testing
* **Prepruning**
  * Hentikan konstruksi pohon lebih awal — jangan membagi simpul jika ini akan mengakibatkan akurasi jatuh di bawah ambang batas
* **Postpruning**
  * Hapus cabang dari Tree/pohon yang terlalu lebat.
  * Jika memangkas/*pruning* sebuah node menghasilkan tingkat error yang lebih kecil (terhadap test set), silahkan di-*pruning*

### Discussion on DT
* **Kelebihan:**
  * Aturan klasifikasi yang dapat dipahami oleh manusia
  * Kecepatan belajar/klasifikasi yang relatif lebih cepat
* **Kekurangan:**
  * *Sensitive* (not robust/tidak kuat) terhadap *noises*
  * Atribut bernilai kontinu - secara dinamis mempartisi nilai atribut kontinu ke dalam set interval diskrit

---

## Ensemble Methods

### General Idea

> **[Deskripsi Diagram: General Idea Ensemble Methods]**
> *Diagram alur ini menampilkan tiga langkah utama dalam ensemble methods:*
> * **Step 1: Create Multiple Data Sets** - Data training asli (D) dibagi menjadi beberapa subset (D₁, D₂, ..., Dₜ).*
> * **Step 2: Build Multiple Classifiers** - Setiap subset dilatih dengan classifier terpisah (C₁, C₂, ..., Cₜ).*
> * **Step 3: Combine Classifiers** - Hasil prediksi dari semua classifier digabungkan menjadi satu model akhir (C*).*

### Ensemble Methods
* **Ensemble**
  * Use a combination of models to increase accuracy
  * Combine a series of *k* learned models, M₁, M₂, ..., Mₖ, with the aim of creating an improved model M*
* **Popular ensemble methods:**
  * **Bagging:** averaging the prediction over a collection of classifiers
  * **Boosting:** weighted vote with a collection of classifiers

### Bagging
* **Analogy:** Diagnosis based on multiple doctors' majority vote
* **Training:**
  * Given a set D of d tuples, at each iteration i, a training set Dᵢ of d tuples is sampled **with replacement** from D (i.e., *bootstrap*)
  * A classifier model Mᵢ is learned for each training set Dᵢ
* **Classification:**
  * Each classifier Mᵢ returns its class prediction
  * The bagged classifier M* counts the votes and assigns the class with the most votes to an unknown sample X
* **Accuracy:**
  * Often significant better than a single classifier derived from D
  * For noise data: not considerably worse, more robust
  * Proved improved accuracy in prediction

### Bagging
* Sampling with replacement
* Build classifier on each bootstrap sample
* Counts the votes and assigns the class with the most votes to an unknown sample X
* **Example: Random Forest**

> **[Deskripsi Tabel: Contoh Proses Bagging]**
> *Tabel ini menampilkan proses sampling dengan pengembalian (with replacement) untuk tiga ronde bagging:*
> * **Original Data:** 10 sampel (1 hingga 10).*
> * **Bagging (Round 1):** Sampel yang terpilih: 7, 8, 10, 8, 2, 5, 10, 10, 5, 9 (beberapa sampel muncul lebih dari sekali, seperti 8 dan 10).*
> * **Bagging (Round 2):** Sampel yang terpilih: 1, 4, 9, 1, 2, 3, 2, 7, 3, 2.*
> * **Bagging (Round 3):** Sampel yang terpilih: 1, 8, 5, 10, 5, 5, 9, 6, 3, 7.*
> * *Setiap ronde menghasilkan classifier (c₁, c₂, c₃) yang kemudian digabungkan melalui voting.*

### Random Forest
* Merupakan implementasi populer dari teknik Bagging yang menggunakan Decision Tree sebagai base classifier.

### Boosting
* **Analogy:** Consult several doctors, based on a combination of diagnoses. Weight assigned based on the previous diagnosis accuracy.
* **Training:**
  * Weights are assigned to each training tuple
  * A series of *k* classifiers is iteratively learned
  * After a classifier Mᵢ is learned, the weights are updated to allow the subsequent classifier, Mᵢ₊₁, to pay more attention to the training tuples that were misclassified by Mᵢ
* **Classification:**
  * The final M* combines the votes of each individual classifier, where the weight of each classifier's vote is a function of its accuracy
* **Accuracy:**
  * Comparing with bagging: boosting tends to achieve greater accuracy, but it also risks overfitting the model to misclassified data

### Boosting
* Records that are wrongly classified will have their weights increased
* Records that are classified correctly will have their weights decreased
* Example 4 is hard to classify
* Its weight is increased; therefore it is more likely to be chosen again in subsequent rounds
* **Example: AdaBoost**

> **[Deskripsi Tabel: Contoh Proses Boosting]**
> *Tabel ini menampilkan proses boosting dengan penyesuaian bobot:*
> * **Original Data:** 10 sampel (1 hingga 10).*
> * **Boosting (Round 1):** Sampel yang terpilih: 7, 3, 2, 8, 7, 9, 4, 10, 6, 3.*
> * **Boosting (Round 2):** Sampel yang terpilih: 5, 4, 9, 4, 2, 5, 1, 7, 4, 2. (Sampel 4 muncul lebih sering karena sulit diklasifikasikan di ronde sebelumnya).*
> * **Boosting (Round 3):** Sampel yang terpilih: 4, 4, 8, 10, 4, 5, 4, 6, 3, 4. (Sampel 4 terus muncul dengan frekuensi tinggi).*
> * *Setiap ronde menghasilkan classifier (c₁, c₂, c₃) dengan bobot yang berbeda berdasarkan akurasinya.*

---

## Logistic Regression

### Logistic Regression
* Logistic Regression adalah algoritma klasifikasi Machine Learning yang digunakan untuk memprediksi ketika variabel dependen (target) adalah kategoris.
* Target adalah variabel biner yang berisi kelas 1 (untuk kasus benar/ya) atau 0 (untuk kasus salah/tidak).

### Logistic Regression
* Merupakan sebuah kasus khusus regresi linier di mana responsnya adalah 'log of odds'.
* Model Regresi Logistik memprediksi P(Y=1) dengan memasukkan data ke fungsi logit.

> **[Deskripsi Gambar: Contoh Kasus Logistic Regression]**
> *Ilustrasi ini menampilkan contoh kasus prediksi diabetes:*
> * **Q:** Patient with BG (Blood Glucose) 190 mg/dL, is it diagnosed as diabetes?*
> * **A:** Probability diabetes is 0.882*
> * *Grafik menunjukkan kurva sigmoid yang memetakan nilai input (blood glucose level) ke probabilitas antara 0 dan 1. Pada nilai BG 190 mg/dL, probabilitas berada di sekitar 0.882, yang berarti pasien memiliki kemungkinan tinggi terkena diabetes.*

---

## Performance metrics for Classification

### Metrics for Performance Evaluation
* Focus on the predictive capability of a model
* **Confusion Matrix:**
  * a: TP (true positive)
  * b: FN (false negative)
  * c: FP (false positive)
  * d: TN (true negative)

> **[Deskripsi Tabel: Confusion Matrix]**
> *Tabel 2x2 ini menampilkan struktur confusion matrix:*
> * **Sumbu Y (Actual Class):** Class=Yes(1) dan Class=No(0)*
> * **Sumbu X (Predicted Class):** Class=Yes(1) dan Class=No(0)*
> * **Sel (Actual=Yes, Predicted=Yes):** a (TP) - Prediksi positif yang benar*
> * **Sel (Actual=Yes, Predicted=No):** b (FN) - Prediksi negatif yang salah*
> * **Sel (Actual=No, Predicted=Yes):** c (FP) - Prediksi positif yang salah*
> * **Sel (Actual=No, Predicted=No):** d (TN) - Prediksi negatif yang benar*

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN} = \frac{a + d}{a + b + c + d}$$

### Example
* **Example 1:**
  * Accuracy = (5+5)/(5+5+0+0) = 1

> **[Deskripsi Tabel: Confusion Matrix Example 1]**
> *Tabel confusion matrix dengan:*
> * *TP = 5, FN = 0, FP = 0, TN = 5*
> * *Semua prediksi benar, akurasi 100%*

* **Example 2:**
  * Accuracy = (3+4)/(3+4+2+1) = 0.7

> **[Deskripsi Tabel: Confusion Matrix Example 2]**
> *Tabel confusion matrix dengan:*
> * *TP = 4, FN = 1, FP = 2, TN = 3*
> * *Akurasi 70%*

### Limitation of accuracy
* Consider a 2-class problem
  * Number of Class 0 examples = 990
  * Number of Class 1 examples = 10
* If model predicts everything to be class 0, accuracy is 990/1000 = 99%
  * Accuracy is misleading because model does not detect any class 1 example

> **[Deskripsi Tabel: Limitation of Accuracy]**
> *Tabel confusion matrix yang menunjukkan:*
> * *TN = 990, FP = 0, FN = 10, TP = 0*
> * *Akurasi 99%, tetapi model gagal mendeteksi kelas minoritas (class 1) sama sekali. Ini menunjukkan bahwa accuracy bisa menyesatkan pada dataset yang imbalanced.*

### Cost-sensitive measures
* **True positive rate (TPR)** = sensitivity or recall
* **True negative rate (TNR)** = specificity
* **Recall** = How good a model is at detecting the positives
* **Precision** = What proportion of positive identifications was actually correct?
* **F1 score** = conveys the balance between the precision and the recall

> **[Deskripsi Tabel: Struktur Confusion Matrix untuk Cost-Sensitive Measures]**
> *Tabel 2x2 dengan label TP, FN, FP, TN untuk kelas Yes dan No.*

### Example
* **Example:**
  * Accuracy = (TP+TN)/(TP+TN+FP+FN) = (1+6)/(1+6+2+1) = 7/10 = 0.7
  * Precision = TP/(TP+FP) = 1/(1+2) = 1/3 = 0.33
  * Recall = TP/(TP+FN) = 1/(1+1) = 1/2 = 0.5
  * Specificity = TN/(TN+FP) = 6/(6+2) = 6/8 = 0.75
  * F1 = 2*(precision*recall)/(precision+recall) = 2*(0.33*0.5)/(0.33+0.5) = 3.3/0.83 = 0.4

> **[Deskripsi Tabel: Confusion Matrix Example untuk Cost-Sensitive Measures]**
> *Tabel confusion matrix dengan:*
> * *TN = 6, FP = 2, FN = 1, TP = 1*
> * *Perhitungan metrik: Accuracy = 0.7, Precision = 0.33, Recall = 0.5, Specificity = 0.75, F1 = 0.4*

> **[Deskripsi Tabel: Struktur Confusion Matrix dengan Label TP, TN, FP, FN]**
> *Tabel 2x2 dengan:*
> * *Actual Class=0, Predicted Class=0: TN*
> * *Actual Class=0, Predicted Class=1: FP*
> * *Actual Class=1, Predicted Class=0: FN*
> * *Actual Class=1, Predicted Class=1: TP*

---

## Validation techniques

### Validation techniques
* **Performance of a model may depend on other factors besides the learning algorithm:**
  * Class distribution
  * Cost of misclassification
  * Size of training and test sets
* **Techniques:**
  * Holdout
  * Cross validation

### Holdout
* **Holdout:**
  * The data is split into two different datasets as a training and a testing dataset.
  * This can be a 60/40 or 70/30 or 80/20 split.

> **[Deskripsi Diagram: Holdout Method]**
> *Diagram ini menampilkan dataset asli yang dibagi menjadi dua bagian:*
> * **Training set:** 60%, 70%, atau 80% dari data*
> * **Testing set:** 40%, 30%, atau 20% dari data*
> * *Model dilatih pada training set dan dievaluasi pada testing set yang tidak pernah dilihat selama pelatihan.*

### Cross validation
* **Cross validation:**
  * Partition data into *k* disjoint subsets
  * *k-fold:* train on k-1 partitions, test on the remaining one

> **[Deskripsi Diagram: k-Fold Cross Validation]**
> *Diagram ini menampilkan dataset yang dibagi menjadi k subset (misalnya k=5). Proses iterasi dilakukan k kali:*
> * **Iterasi 1:** Fold 1 sebagai test set, Fold 2-5 sebagai train set*
> * **Iterasi 2:** Fold 2 sebagai test set, Fold 1, 3-5 sebagai train set*
> * **Iterasi 3:** Fold 3 sebagai test set, Fold 1-2, 4-5 sebagai train set*
> * **Iterasi 4:** Fold 4 sebagai test set, Fold 1-3, 5 sebagai train set*
> * **Iterasi 5:** Fold 5 sebagai test set, Fold 1-4 sebagai train set*
> * *Hasil akhir adalah rata-rata performa dari k iterasi.*

### Stratified cross validation
* **Stratified cross validation:**
  * Partition data into *k* disjoint subsets
  * In stratified *k-fold* cross-validation, each subset is stratified so that they contain approximately the same proportion of class labels as the original dataset.
  * *k-fold:* train on k-1 partitions, test on the remaining one

> **[Deskripsi Diagram: Stratified k-Fold Cross Validation]**
> *Diagram ini mirip dengan k-fold cross validation, namun setiap fold mempertahankan proporsi kelas yang sama dengan dataset asli. Misalnya, jika dataset asli memiliki 20% kelas positif dan 80% kelas negatif, maka setiap fold juga akan memiliki proporsi yang sama. Ini penting untuk dataset yang imbalanced agar evaluasi model lebih representatif.*

---