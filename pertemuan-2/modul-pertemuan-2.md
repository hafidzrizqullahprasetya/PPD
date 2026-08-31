# Modul Praktikum Penambangan Data
## Teknologi Rekayasa Perangkat Lunak – 2025
### PERTEMUAN II: Exploratory Data Analysis dan Data Preprocessing

---

## 2.1. TUJUAN PEMBELAJARAN

A. Mahasiswa mampu memahami konsep dasar Artificial Intelligence (AI)
B. Mahasiswa mengetahui dasar-dasar Machine Learning (ML)
C. Mahasiswa dapat menerapkan langkah awal analisis data
D. Mahasiswa mampu mengetahui dan menerapkan teknik-teknik dasar dalam data preprocessing

---

## 2.2. DASAR TEORI

### Artificial Intelligence
Artificial Intelligence (AI) merupakan bidang ilmu komputer yang berupaya membuat mesin mampu melakukan tugas-tugas yang biasanya membutuhkan kecerdasan manusia, seperti berpikir, belajar, mengenali pola, dan mengambil keputusan. AI tidak hanya terbatas pada sistem cerdas yang meniru manusia, tetapi juga mencakup algoritma dan pendekatan komputasional yang memungkinkan suatu sistem bekerja secara otonom dan adaptif. Perkembangan komputer modern, besarnya volume data, dan meningkatnya kemampuan komputasi membuat AI berkembang pesat di berbagai sektor seperti kesehatan, finansial, transportasi, hingga hiburan. Google AI Guides menjelaskan bahwa AI modern sangat bertumpu pada data, karena kemampuan kecerdasan sebuah sistem sangat bergantung pada kualitas data yang dipelajari dan pola-pola yang berhasil dipahami selama proses pemodelan.

### Machine Learning
Machine Learning adalah salah satu cabang AI yang menjadi fondasi bagi banyak inovasi teknologi saat ini. Machine Learning merupakan sebuah metode yang memungkinkan mesin meningkatkan performanya pada suatu tugas berdasarkan data. Inti dari ML adalah kemampuan sistem untuk mengenali pola dan membuat prediksi tanpa harus diprogram dengan aturan eksplisit. Secara umum, ML terbagi menjadi tiga, yaitu supervised learning, unsupervised learning, dan reinforcement learning.

Supervised learning menggunakan data berlabel untuk melakukan prediksi atau klasifikasi, misalnya memprediksi harga rumah atau mendeteksi sentimen teks. Unsupervised learning berfungsi menemukan struktur tersembunyi dalam data tanpa label, seperti clustering pelanggan berdasarkan kebiasaan belanja. Sementara itu, reinforcement learning memungkinkan model belajar dari feedback berupa reward dan penalty, yang sering digunakan dalam robotika dan game AI. Dalam semua metode ini, data harus dipersiapkan dengan baik agar algoritma dapat belajar secara optimal. Kualitas model ML tidak hanya ditentukan oleh algoritma yang digunakan, tetapi juga oleh kualitas data dan preprocessing yang mendasarinya.

### Exploratory Data Analysis
Sebelum data dapat digunakan untuk membangun model ML, perlu dilakukan Exploratory Data Analysis (EDA). EDA merupakan proses penggalian informasi awal dari dataset untuk memahami karakteristik dan kondisi data secara menyeluruh. Gagasan EDA pertama kali diperkenalkan oleh John W. Tukey, yang menekankan pentingnya "mendengarkan apa yang data katakan" sebelum terburu-buru membuat asumsi atau membangun model. Dalam praktik modern, seperti yang dijelaskan dalam Kaggle Learn dan IBM Data Analytics Guides, EDA melibatkan berbagai aktivitas seperti melihat distribusi variabel melalui histogram atau boxplot, mencari korelasi antar fitur menggunakan heatmap, mendeteksi outliers, memeriksa tipe data, serta mengidentifikasi masalah seperti missing values atau duplikasi. Tahapan ini sangat penting karena memberikan gambaran awal tentang kualitas dataset, potensi masalah, dan tindakan apa saja yang perlu dilakukan selama preprocessing. Tanpa EDA, proses pemodelan bisa tidak akurat atau bias akibat ketidaktahuan terhadap struktur dan perilaku data.

### Data Preprocessing
Data preprocessing adalah proses sistematis untuk membersihkan, mengubah, dan menyiapkan data agar dapat digunakan secara optimal dalam algoritma machine learning. Preprocessing disebut sebagai tahapan kritis yang secara langsung memengaruhi performa dan generalisasi model.

Tahapan pertama biasanya dimulai dari data cleaning, yaitu penanganan missing values menggunakan metode imputasi seperti mean, median, atau mode, serta menghapus atau memperbaiki data duplikat yang dapat mengganggu konsistensi analisis. Tahap berikutnya adalah data transformation, yang bertujuan mengubah data ke bentuk yang lebih representatif. Fitur kategorikal perlu di-encode menggunakan label encoding atau one-hot encoding, sebagaimana dianjurkan dalam Google Machine Learning Crash Course. Selanjutnya, dilakukan proses feature scaling seperti normalisasi (min-max scaling) atau standardisasi (z-score scaling) untuk menyamakan rentang nilai antar fitur, terutama pada algoritma sensitif terhadap jarak seperti KNN dan SVM. Langkah penting lainnya adalah melakukan dataset splitting seperti train–test split (misalnya 80:20) untuk memastikan model dapat diuji secara objektif dan tidak overfitting. Seluruh proses ini umumnya dilakukan menggunakan pustaka numerik dan analitik seperti NumPy dan Pandas.

---

## 2.3. ALAT DAN BAHAN

**Perangkat Keras:**
- Komputer/Laptop

**Perangkat Lunak:**
- Python 3
- Google Colaboratory

---

## 2.4. LANGKAH PERCOBAAN

### A. Exploratory Data Analysis (EDA)

#### 1. Impor Pustaka
**Penjelasan Gambar 2.4.1 (Import Library):** Gambar ini menampilkan kode Python untuk mengimpor pustaka yang akan digunakan selama praktikum, yaitu `pandas` untuk manipulasi data, `numpy` untuk komputasi numerik, dan `matplotlib.pyplot` untuk visualisasi data. Contoh kode yang ditampilkan:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

#### 2. Load Dataset
Link dataset: https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers

Kode yang digunakan:

```python
df_churning = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/main/BankChurners.csv', delimiter=',')
```

#### 3. Melihat lima baris pertama dengan df.head()
**Penjelasan Gambar 2.4.2 (Lima Baris Pertama Dataset):** Gambar ini menampilkan output dari perintah `df_churning.head()` yang menunjukkan lima baris pertama dari dataset Bank Churners. Tabel menampilkan kolom-kolom seperti CLIENTNUM, Attrition_Flag, Customer_Age, Gender, Dependent_count, Education_Level, Marital_Status, Income_Category, Card_Category, Months_on_book, dan berbagai atribut lainnya.

```python
df_churning.head()
```

#### 4. Melihat lima baris terakhir dengan df.tail()
**Penjelasan Gambar 2.4.3 (Lima Baris Terakhir Dataset):** Gambar ini menampilkan output dari perintah `df_churning.tail()` yang menunjukkan lima baris terakhir dari dataset, dengan struktur kolom yang sama seperti pada head().

```python
df_churning.tail()
```

#### 5. Menyimpan nilai dari class menjadi 2 dataframe yang berbeda
**Penjelasan Gambar 2.4.4 (Pemisahan Dataset Bank Churner):** Gambar menampilkan kode untuk memisahkan dataset berdasarkan nilai kolom Attrition_Flag menjadi dua subset.

```python
existing_data = df_churning[df_churning['Attrition_Flag'] == 'Existing Customer']
attrited_data = df_churning[df_churning['Attrition_Flag'] == 'Attrited Customer']
```

Kode tersebut merupakan kode untuk memisahkan dataset berdasarkan nilai kolom Attrition_Flag, sehingga terbentuk dua subset yaitu existing_data yang berisi pelanggan yang masih aktif (Existing Customer) dan attrited_data yang berisi pelanggan yang berhenti atau churn (Attrited Customer).

#### 6. Line Chart
**Penjelasan Gambar 2.4.5 (Pembuatan Line Chart dengan Data Dummy):** Gambar menampilkan kode pembuatan line chart menggunakan data dummy.

```python
x = [1, 2, 3, 4, 5]
y1 = [2, 3, 5, 7, 11]
y2 = [1, 4, 9, 16, 25]

plt.plot(x, y1, marker='o')
plt.plot(x, y2, marker='X')
plt.show()
```

Pada kode di atas, x dipakai sebagai sumbu horizontal, y1 dan y2 sebagai dua garis berbeda di sumbu vertikal. Parameter marker='o' dan 'X' digunakan untuk memberi tanda di setiap titik data. plt.show() digunakan untuk menampilkan plot ke layar. Tujuan pemanggilan fungsi ini adalah menunjukkan cara membuat line chart untuk melihat tren atau perubahan nilai dari dua seri data sekaligus.

**Penjelasan Gambar 2.4.6 (Line Chart dari Data Dummy):** Gambar menampilkan hasil visualisasi line chart dengan dua garis berbeda yang menghubungkan titik-titik data dengan marker lingkaran (o) dan silang (X), menunjukkan tren kenaikan kedua seri data.

Setelah memahami konsep line chart menggunakan data dummy, langkah berikutnya adalah menerapkannya pada dataset IoT untuk menampilkan perubahan nilai sensor terhadap waktu.

**Penjelasan Gambar 2.4.7 (Load Dataset IoT):** Gambar menampilkan kode untuk memuat dataset IoT dari sumber data.

```python
df_iot = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/main/IoT.csv', delimiter=',')
```

**Penjelasan Gambar 2.4.8 (Sampel Data IoT):** Gambar menampilkan output `df_iot.head()` yang memperlihatkan beberapa baris awal data IoT dengan kolom seperti date, Temperature, Humidity, dan atribut sensor lainnya sebagai verifikasi bahwa data telah terbaca dengan benar.

Langkah berikutnya adalah menampilkan line chart untuk memvisualisasikan perubahan nilai sensor terhadap waktu.

**Penjelasan Gambar 2.4.9 (Pembuatan Line Chart dengan Data IoT):** Gambar menampilkan kode untuk membuat line chart dari dataset IoT.

```python
plt.figure(figsize=(15, 3))
plt.plot(df_iot['date'], df_iot['Humidity'])
plt.show()
```

Kode plt.figure(figsize=(15, 3)) berfungsi untuk mengatur ukuran grafik agar panjang secara horizontal. Selanjutnya digunakan plt.plot(df_iot['date'], df_iot['Humidity']) untuk memetakan kolom date sebagai sumbu-x dan nilai Humidity sebagai sumbu-y. Grafik kemudian ditampilkan dengan plt.show().

**Penjelasan Gambar 2.4.10 (Line Chart dengan Data IoT):** Gambar menampilkan hasil line chart yang menunjukkan perubahan nilai kelembaban (Humidity) terhadap waktu (date), dengan garis yang naik-turun mengikuti perubahan nilai sensor.

#### 7. Pie Chart
**Penjelasan Gambar 2.4.11 (Pembuatan Pie Chart dengan Data Dummy):** Gambar menampilkan kode pembuatan pie chart dengan data dummy.

```python
y = [35, 25, 25, 15]
mylabels = ["Web Programmer", "Data Scientist", "DB Admin", "Manager"]

plt.pie(y, labels=mylabels, autopct='%.2f%%')
plt.show()
```

Pembuatan pie chart dimulai dengan mendefinisikan array y yang berisi nilai-nilai numerik mewakili frekuensi tiap kategori, yaitu [35, 25, 25, 15]. Kemudian dibuat daftar mylabels yang merupakan nama kategori: "Web Programmer", "Data Scientist", "DB Admin", dan "Manager". Fungsi plt.pie() digunakan untuk menghasilkan pie chart, dengan parameter labels=mylabels untuk memberi nama pada setiap irisan grafik. Sedangkan autopct='%.2f%%' digunakan untuk menampilkan persentase masing-masing irisan dengan format dua angka di belakang koma. Terakhir, plt.show() memunculkan visualisasi ke layar.

**Penjelasan Gambar 2.4.12 (Pie Chart dengan Data Dummy):** Gambar menampilkan hasil pie chart dengan empat irisan berwarna berbeda yang mewakili proporsi masing-masing profesi, lengkap dengan label dan persentase dua angka di belakang koma.

Selanjutnya, pembuatan pie chart dilakukan menggunakan dataset sebelumnya yaitu Bank Churner. Visualisasi ini bertujuan untuk menampilkan proporsi kategori pada variabel Marital_Status.

**Penjelasan Gambar 2.4.13 (Pembuatan Pie Chart dengan Data Bank Churner):** Gambar menampilkan kode pembuatan pie chart dari data Bank Churner.

```python
data = df_churning['Marital_Status'].value_counts()
label = data.index

plt.pie(data, labels=label, autopct='%.2f%%')
plt.title('Marital Status')
plt.show()
```

Kode dimulai dengan memanggil df_churning['Marital_Status'].value_counts() untuk menghitung jumlah kemunculan setiap kategori dalam kolom tersebut. Nilai frekuensi yang dihasilkan kemudian disimpan dalam variabel data, sementara data.index digunakan untuk mengambil nama kategorinya dan disimpan ke dalam variabel label. Fungsi plt.pie(data, labels=label, autopct='%.2f%%') digunakan untuk menggambar pie chart dengan persentase setiap kategori ditampilkan secara otomatis. Selanjutnya, plt.title('Marital Status') digunakan untuk menambahkan judul pada grafik.

**Penjelasan Gambar 2.4.14 (Pie Chart Marital Status):** Gambar menampilkan hasil pie chart distribusi status pernikahan nasabah Bank Churner dengan judul "Marital Status", lengkap dengan persentase tiap kategori (Married, Single, Divorced, Unknown).

#### 8. Bar Plot
**Penjelasan Gambar 2.4.15 (Pembuatan Bar Plot dengan Data Dummy):** Gambar menampilkan kode pembuatan bar plot dengan data dummy.

```python
y = [35, 25, 25, 15]
x = ["Web Programmer", "Data Scientist", "DB Admin", "Manager"]

plt.bar(x, y)
plt.show()
```

Pembuatan bar plot dimulai dengan mendefinisikan array y yang berisi nilai jumlah pekerja pada beberapa jenis profesi, dan daftar x yang berisi nama kategori profesi seperti "Web Programmer", "Data Scientist", "DB Admin", dan "Manager". Selanjutnya, fungsi plt.bar(x, y) digunakan untuk menggambar batang pada sumbu x berdasarkan kategori, dengan tinggi batang mengikuti nilai yang diberikan pada variabel y.

**Penjelasan Gambar 2.4.16 (Bar Plot dengan Data Dummy):** Gambar menampilkan hasil bar plot dengan empat batang vertikal yang tingginya sesuai dengan nilai masing-masing kategori profesi.

Langkah selanjutnya adalah membuat bar plot dengan dataset Bank Churner untuk membandingkan distribusi kategori Marital_Status antara dua kelompok nasabah, yaitu existing customers dan attrited customers.

**Penjelasan Gambar 2.4.17 (Pembuatan Bar Plot Marital Status):** Gambar menampilkan kode pembuatan bar plot perbandingan Marital Status.

```python
y1 = existing_data['Marital_Status'].value_counts()
y2 = attrited_data['Marital_Status'].value_counts()

x1 = np.arange(len(y1.index))
x2 = np.arange(len(y2.index))
bar_width = 0.4

plt.bar(x1, y1, width=bar_width)
plt.bar(x2 + bar_width, y2, width=bar_width)
plt.title("Marital Status")
plt.ylabel("Count")
plt.xticks(x2 + bar_width / 2, labels=df_churning.Marital_Status.unique())
plt.show()
```

Langkah pertama adalah menghitung jumlah masing-masing kategori marital status pada kedua subset menggunakan value_counts(), yang disimpan dalam variabel y1 dan y2. Lalu membuat indeks posisi awal masing-masing kategori dengan np.arange(len(y1.index)) dan np.arange(len(y2.index)). Variabel bar_width=0.4 digunakan untuk menentukan lebar batang. Fungsi plt.bar(x1, y1, width=bar_width) menggambar batang pertama untuk existing customers, sementara plt.bar(x2 + bar_width, y2, width=bar_width) menggambar batang kedua untuk attrited customers. Selanjutnya, judul grafik ditetapkan dengan plt.title("Marital Status"), dan sumbu-y diberi label "Count". plt.xticks(x2 + bar_width / 2, labels=df_churning.Marital_Status.unique()) digunakan untuk memastikan label kategori muncul tepat di tengah pasangan batang untuk setiap kelompok marital status.

**Penjelasan Gambar 2.4.18 (Bar Plot Marital Status):** Gambar menampilkan hasil bar plot dengan batang berpasangan untuk setiap kategori marital status, membandingkan jumlah existing customers dan attrited customers.

#### 9. Stacked Bar Plot
**Penjelasan Gambar 2.4.19 (Pembuatan Stacked Bar Plot Marital Status):** Gambar menampilkan kode pembuatan stacked bar plot.

```python
data1 = existing_data['Marital_Status'].value_counts()
data2 = attrited_data['Marital_Status'].value_counts()

df_new = pd.concat([data1, data2], keys=['existing', 'attrited'], axis=1)
df_new.plot(kind='bar', stacked=True)
plt.show()
```

Langkah selanjutnya yaitu membuat stacked bar plot untuk menganalisis komposisi kategori Marital_Status pada dua kelompok nasabah dalam dataset Bank Churner. Jumlah masing-masing kategori marital status dihitung menggunakan value_counts(), yang disimpan dalam variabel data1 untuk existing customers dan data2 untuk attrited customers, kemudian digabungkan menjadi satu DataFrame menggunakan pd.concat() dengan parameter keys=['existing', 'attrited'] dan axis=1, sehingga muncul dua kolom baru yang merepresentasikan kedua kelompok tersebut. Visualisasi dibuat menggunakan df_new.plot(kind='bar', stacked=True), yang menghasilkan diagram batang bertumpuk.

**Penjelasan Gambar 2.4.20 (Stacked Bar Plot Marital Status):** Gambar menampilkan hasil stacked bar plot dengan batang bertumpuk yang menunjukkan komposisi existing dan attrited customers untuk setiap kategori marital status.

#### 10. Histogram
**Penjelasan Gambar 2.4.21 (Pembuatan Histogram dengan Data Dummy):** Gambar menampilkan kode pembuatan histogram dengan data dummy.

```python
data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
bins = [1, 2, 3, 4, 5]

plt.hist(data, bins=bins)
plt.show()
```

Pembuatan histogram dimulai dengan pembuatan data dummy yang disimpan dalam array. Variabel bins didefinisikan digunakan untuk membagi data ke dalam empat interval atau kelas. Fungsi plt.hist(data, bins=bins) digunakan untuk menggambar histogram, di mana setiap batang menunjukkan jumlah data yang jatuh ke dalam masing-masing interval.

**Penjelasan Gambar 2.4.22 (Histogram dengan Data Dummy):** Gambar menampilkan hasil histogram dengan empat batang yang menunjukkan frekuensi data dalam setiap interval bins.

Langkah selanjutnya adalah membuat histogram dengan dataset Bank Churner, khususnya pada variabel Customer_Age.

**Penjelasan Gambar 2.4.23 (Pembuatan Histogram Customer Age):** Gambar menampilkan kode pembuatan histogram Customer Age.

```python
data1 = existing_data['Customer_Age']
data2 = attrited_data['Customer_Age']

plt.hist(data1, label='existing')
plt.hist(data2, label='attrited')
plt.show()
```

Dua subset data existing_data dan attrited_data digunakan untuk memisahkan nasabah yang masih aktif dan nasabah yang sudah berhenti. Nilai usia dari masing-masing kelompok diambil menggunakan existing_data['Customer_Age'] dan attrited_data['Customer_Age'], lalu disimpan dalam data1 dan data2. Visualisasi dilakukan dengan memanggil plt.hist(data1, label='existing') dan plt.hist(data2, label='attrited').

**Penjelasan Gambar 2.4.24 (Histogram Customer Age):** Gambar menampilkan hasil histogram distribusi usia pelanggan dengan dua kelompok data (existing dan attrited) yang ditumpuk dalam satu grafik, menunjukkan bahwa sebagian besar nasabah berada pada rentang usia produktif (sekitar 40-55 tahun).

#### 11. Scatter Plot
**Penjelasan Gambar 2.4.25 (Pembuatan Scatter Plot dengan plt.scatter):** Gambar menampilkan kode pembuatan scatter plot menggunakan plt.scatter.

```python
x1 = attrited_data['Customer_Age']
y1 = attrited_data['Months_on_book']
x2 = existing_data['Customer_Age']
y2 = existing_data['Months_on_book']

plt.scatter(x2, y2)
plt.scatter(x1, y1)
plt.show()
```

Pembuatan scatter plot dimulai dengan memisahkan kedua variabel untuk dua kelompok yaitu attrited customers (x1 untuk usia dan y1 untuk lama berlangganan) serta existing customers (x2 dan y2). Dengan menggunakan plt.scatter(x2, y2) dan plt.scatter(x1, y1), kedua kelompok digambarkan dalam satu grafik scatter.

**Penjelasan Gambar 2.4.26 (Scatter Plot Customer Age vs Months on Book):** Gambar menampilkan hasil scatter plot yang menunjukkan hubungan antara usia pelanggan (Customer_Age) dan lama berlangganan (Months_on_book) untuk kedua kelompok nasabah.

Versi lain pembuatan scatter plot menggunakan fungsi .plot.scatter(). Cara ini lebih ringkas dan terstruktur dibandingkan pemanggilan plt.scatter() secara manual.

**Penjelasan Gambar 2.4.27 (Pembuatan Scatter Plot dengan fungsi plot.scatter):** Gambar menampilkan kode alternatif pembuatan scatter plot.

```python
ax = existing_data.plot.scatter(x='Customer_Age', y='Months_on_book', c='red')
attrited_data.plot.scatter(x='Customer_Age', y='Months_on_book', c='blue', ax=ax)
plt.show()
```

Visualisasi dimulai dengan memanggil existing_data.plot.scatter(x='Customer_Age', y='Months_on_book', c='red'), yang menghasilkan scatter plot berwarna merah untuk nasabah yang masih aktif. Hasil plot ini disimpan dalam variabel ax, yang berfungsi sebagai axes reference untuk menumpuk plot berikutnya pada grafik yang sama. Baris selanjutnya memanggil attrited_data.plot.scatter(x='Customer_Age', y='Months_on_book', c='blue', ax=ax), yang menggambar titik-titik untuk nasabah churn dengan warna biru pada axis yang sama.

**Penjelasan Gambar 2.4.28 (Scatter Plot Customer Age vs Months on Book):** Gambar menampilkan hasil scatter plot dengan titik merah untuk existing customers dan titik biru untuk attrited customers dalam satu grafik yang sama.

#### 12. Box Plot
**Penjelasan Gambar 2.4.29 (Pembuatan Box Plot dengan plt.boxplot):** Gambar menampilkan kode pembuatan box plot.

```python
plt.boxplot(df_churning['Customer_Age'])
plt.show()
```

Pembuatan boxplot dilakukan dengan memanggil plt.boxplot(df_churning['Customer_Age']). Visualisasi yang dihasilkan memperlihatkan nilai median, sebaran kuartil, serta potensi outlier dalam satu kelompok data.

**Penjelasan Gambar 2.4.30 (Box Plot Customer Age):** Gambar menampilkan hasil box plot distribusi usia pelanggan yang memperlihatkan median (garis dalam kotak), kuartil 1 dan 3 (batas kotak), whisker, serta titik-titik outlier jika ada.

**Penjelasan Gambar 2.4.31 (Pembuatan Box Plot dengan fungsi plot.box):** Gambar menampilkan kode alternatif pembuatan box plot menggunakan method DataFrame.

```python
df_churning.boxplot(column='Customer_Age')
plt.show()
```

**Penjelasan Gambar 2.4.32 (Box Plot Customer Age):** Gambar menampilkan hasil box plot Customer Age menggunakan fungsi plot.box dari objek dataframe.

#### 13. Violin Plot
**Penjelasan Gambar 2.4.33 (Pembuatan Violin Plot):** Gambar menampilkan kode pembuatan violin plot.

```python
data_existing = existing_data['Customer_Age']
data_attrited = attrited_data['Customer_Age']

data_dict = {'existing': data_existing, 'attrited': data_attrited}

plt.violinplot(data_dict.values())
plt.xticks([1, 2], data_dict.keys())
plt.show()
```

Pembuatan violin plot dimulai dengan mengambil nilai Customer_Age dari kedua kelompok pelanggan, lalu menyusunnya dalam dictionary. Data tersebut kemudian divisualisasikan menggunakan fungsi utama plt.violinplot(), yang menampilkan distribusi dan kepadatan usia untuk existing dan attrited customers dalam satu grafik.

**Penjelasan Gambar 2.4.34 (Violin Plot Customer Age):** Gambar menampilkan hasil violin plot dengan dua bentuk violin yang menunjukkan distribusi dan kepadatan usia untuk kelompok existing dan attrited customers.

#### 14. Sub Plot
**Penjelasan Gambar 2.4.35 (Pembuatan Sub Plot):** Gambar menampilkan kode pembuatan subplot dengan empat visualisasi dalam satu figure.

```python
plt.subplot(2, 2, 1)
plt.boxplot(existing_data['Customer_Age'])
plt.title('Box Plot Existing')

plt.subplot(2, 2, 2)
plt.boxplot(attrited_data['Customer_Age'])
plt.title('Box Plot Attrited')

plt.subplot(2, 2, 3)
plt.hist(existing_data['Customer_Age'])
plt.title('Histogram Existing')

plt.subplot(2, 2, 4)
plt.hist(attrited_data['Customer_Age'])
plt.title('Histogram Attrited')

plt.show()
```

Kode di atas membuat empat visualisasi sekaligus dalam satu figure menggunakan plt.subplot(), yaitu dua boxplot (untuk existing dan attrited customers) serta dua histogram usia untuk kedua kelompok tersebut. Tujuan penggunaan subplot adalah menampilkan perbandingan distribusi usia antar kelompok dalam satu tampilan sehingga memudahkan analisis visual secara menyeluruh.

**Penjelasan Gambar 2.4.36 (Sub Plot):** Gambar menampilkan hasil subplot dalam grid 2x2 dengan empat grafik: dua box plot dan dua histogram untuk masing-masing kelompok nasabah.

#### 15. Annotation
**Penjelasan Gambar 2.4.37 (Pembuatan Annotation):** Gambar menampilkan kode pembuatan annotation pada histogram.

```python
plt.hist(df_churning['Customer_Age'])
plt.text(x_posisi, y_posisi, f'Median Age: {median_age}')
plt.text(x_posisi, y_posisi, f'Mean Age: {mean_age}')
plt.show()
```

Kode di atas menampilkan histogram usia pelanggan (Customer_Age) dan menambahkan informasi statistik langsung pada grafik menggunakan plt.text(). Dua anotasi yang ditampilkan yaitu Median Age dan Mean Age.

**Penjelasan Gambar 2.4.38 (Annotation pada Histogram):** Gambar menampilkan hasil histogram Customer Age dengan anotasi teks yang menunjukkan nilai median dan mean usia pelanggan di dalam grafik.

#### 16. Axis
**Penjelasan Gambar 2.4.39 (Pengaturan Axis):** Gambar menampilkan kode pengaturan axis pada line chart.

```python
plt.plot(hari, suhu1, marker='o', label='Hari 1')
plt.plot(hari, suhu2, marker='X', label='Hari 2')
plt.legend()
plt.ylabel('Suhu')
plt.xlabel('Hari')
plt.ylim([-20, 30])
plt.show()
```

Kode di atas menampilkan dua line chart yang menunjukkan data suhu pada beberapa hari. Fungsi plt.plot() digunakan untuk menggambar dua garis berbeda dengan marker serta label yang dibedakan. Kemudian plt.legend() ditambahkan untuk menampilkan keterangan setiap garis. Setelah itu, plt.ylabel() dan plt.xlabel() digunakan untuk memberikan label pada sumbu-y dan sumbu-x agar grafik lebih informatif. plt.ylim([-20, 30]) digunakan untuk mengatur rentang nilai pada sumbu-y, agar tampilan grafik menjadi lebih mudah dibaca.

**Penjelasan Gambar 2.4.40 (Line Chart):** Gambar menampilkan hasil line chart dengan dua garis suhu yang dilengkapi label sumbu, legend, dan rentang sumbu-y dari -20 hingga 30.

#### 17. Legend
**Penjelasan Gambar 2.4.41 (Pembuatan Legend):** Gambar menampilkan kode pembuatan legend pada histogram.

```python
plt.hist(existing_data['Customer_Age'], label='existing')
plt.hist(attrited_data['Customer_Age'], label='attrited')
plt.legend()
plt.show()
```

Kode di atas menampilkan dua histogram usia (Customer_Age) untuk existing dan attrited customers dalam satu grafik. Label untuk masing-masing kelompok ditentukan terlebih dahulu melalui parameter label pada fungsi plt.hist(). Setelah kedua histogram didefinisikan, plt.legend() dipanggil untuk menampilkan kotak legend yang menunjukkan warna dari masing-masing kelompok.

**Penjelasan Gambar 2.4.42 (Histogram dengan Legend):** Gambar menampilkan hasil histogram dengan kotak legend yang menunjukkan warna untuk kelompok existing dan attrited.

---

### B. Data Preprocessing

#### 1. Impor Pustaka
**Penjelasan Gambar 2.4.43 (Import Library):** Gambar menampilkan kode untuk mengimpor pustaka yang diperlukan untuk preprocessing, seperti pandas, numpy, matplotlib, dan sklearn.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

#### 2. Load Dataset
Link: https://www.kaggle.com/datasets/yasserh/titanic-dataset

```python
df = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/main/train.csv')
```

#### 3. Melihat lima baris data pertama dengan df.head()
**Penjelasan Gambar 2.4.44 (Lima Baris Pertama Dataset):** Gambar menampilkan output `df.head()` yang memperlihatkan lima baris pertama dataset Titanic dengan kolom PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, dan Embarked.

#### 4. Melihat lima baris data terakhir dengan df.tail()
**Penjelasan Gambar 2.4.45 (Lima Baris Terakhir Dataset):** Gambar menampilkan output `df.tail()` yang memperlihatkan lima baris terakhir dataset Titanic.

#### 5. Melihat bentuk dataset dengan df.shape
**Penjelasan Gambar 2.4.46 (Melihat Bentuk Dataset dengan df.shape):** Gambar menampilkan output dari perintah `df.shape` yang menghasilkan tuple (891, 12).

```python
df.shape
```

Output (891, 12) menunjukkan bahwa dataset memiliki 891 baris data dan 12 kolom fitur.

#### 6. Visualisasi distribusi data
**Penjelasan Gambar 2.4.47 (Membuat Pie Chart):** Gambar menampilkan kode untuk membuat pie chart distribusi Survived.

```python
df['Survived'].value_counts().plot.pie(autopct='%.2f%%')
plt.show()
```

Kode di atas membuat pie chart untuk menampilkan proporsi penumpang yang selamat dan tidak selamat berdasarkan nilai Survived dalam dataset.

**Penjelasan Gambar 2.4.48 (Pie Chart Distribusi Data):** Gambar menampilkan hasil pie chart yang menunjukkan proporsi penumpang yang selamat (Survived=1) dan tidak selamat (Survived=0), dengan sebagian besar penumpang tidak selamat.

**Penjelasan Gambar 2.4.49 (Membuat Histogram):** Gambar menampilkan kode untuk membuat histogram seluruh kolom numerik.

```python
df.hist(figsize=(10, 8))
plt.show()
```

Kode di atas menampilkan histogram untuk seluruh kolom numerik dalam dataset guna melihat distribusi datanya.

**Penjelasan Gambar 2.4.50 (Histogram Kolom Numerik):** Gambar menampilkan hasil histogram dalam grid untuk semua kolom numerik (PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare), menunjukkan distribusi masing-masing variabel.

#### 7. Analisis korelasi antar variabel
**Penjelasan Gambar 2.4.51 (Korelasi Antar Variabel):** Gambar menampilkan kode dan hasil perhitungan korelasi antar variabel numerik.

```python
df.corr()
```

Kode di atas menghitung korelasi antar variabel numerik dalam dataset, sehingga kita bisa melihat hubungan mana yang paling kuat atau lemah antar fitur. Hasilnya menunjukkan bahwa Pclass memiliki korelasi negatif cukup kuat dengan Survived (–0.33), artinya penumpang kelas lebih rendah cenderung memiliki tingkat keselamatan lebih rendah. Fare berkorelasi positif dengan Survived (0.25), menunjukkan bahwa penumpang dengan tiket lebih mahal lebih mungkin selamat. Sementara variabel lain seperti Age, SibSp, dan Parch memiliki korelasi sangat lemah terhadap Survived.

#### 8. Descriptive Statistics
**Penjelasan Gambar 2.4.52 (Deskripsi Statistik):** Gambar menampilkan output dari `df.describe()`.

```python
df.describe()
```

Hasil df.describe() menampilkan ringkasan statistik untuk fitur numerik. Dari output di atas, terlihat bahwa rata-rata usia penumpang adalah 29.7 tahun, dengan rentang dari 0.42 hingga 80 tahun. Harga tiket (Fare) memiliki variasi besar, mulai dari 0 hingga 512, menunjukkan ketimpangan biaya perjalanan. Variabel Pclass dan Survived juga terlihat sebagai fitur kategorikal numerik dengan nilai 0–1 dan 1–3. Statistik ini membantu kita untuk memahami sebaran, pusat data, serta potensi outlier sebelum dianalisis lebih lanjut.

#### 9. Mengecek missing values dan duplicates
**Penjelasan Gambar 2.4.53 (Cek Missing Value):** Gambar menampilkan kode dan hasil pengecekan missing values.

```python
df.isnull().sum()
```

Pengecekan missing values di atas menunjukkan bahwa sebagian besar kolom tidak memiliki data kosong, namun terdapat 177 missing pada kolom Age, 687 missing pada Cabin, dan 2 missing pada Embarked. Kolom Cabin, Age dan Embarked perlu di-impute agar dataset tetap dapat digunakan.

**Penjelasan Gambar 2.4.54 (Cek Data Duplikat):** Gambar menampilkan kode pengecekan dan penghapusan duplikat.

```python
df.duplicated().sum()
df.drop_duplicates(inplace=True)
```

Kode di atas digunakan untuk mengecek apakah terdapat baris duplikat dalam dataset. Jika ditemukan duplikasi, df.drop_duplicates(inplace=True) akan menghapus baris-baris tersebut.

#### 10. Imputation
**Penjelasan Gambar 2.4.55 (Imputasi Missing Value):** Gambar menampilkan kode imputasi missing values.

```python
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Cabin'].fillna(df['Cabin'].mode()[0], inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
```

Kolom numerik seperti Age diisi menggunakan nilai median agar distribusinya tetap stabil, sedangkan kolom kategorikal seperti Cabin dan Embarked diisi menggunakan modus. Setelah dilakukan imputation, kita perlu memeriksa apakah masih ada missing values pada dataset.

**Penjelasan Gambar 2.4.56 (Missing Value):** Gambar menampilkan hasil pengecekan ulang missing values setelah imputasi, yang menunjukkan bahwa semua kolom sudah tidak memiliki missing value (semua bernilai 0).

#### 11. Memeriksa categorical attribute
**Penjelasan Gambar 2.4.57 (Memeriksa Atribut Kategorikal):** Gambar menampilkan kode pemisahan fitur dan target serta pengecekan atribut kategorikal.

```python
X = df.drop(['PassengerId', 'Name', 'Survived', 'Cabin', 'Ticket'], axis=1)
y = df['Survived']

df_X.select_dtypes(include=['object', 'bool']).columns
```

Kode di atas memisahkan fitur (X) dan target (y) dengan menghapus kolom yang tidak diperlukan dari X, seperti PassengerId, Name, Survived, Cabin, dan Ticket. Selanjutnya, df_X.select_dtypes(include=['object','bool']).columns digunakan untuk mengecek atribut kategorikal dalam X.

**Penjelasan Gambar 2.4.58 (Atribut Kategorikal):** Gambar menampilkan output yang menunjukkan kolom-kolom dengan tipe data object.

Output tersebut menunjukkan bahwa kolom kategorikal dalam fitur X adalah Sex dan Embarked, sehingga kedua kolom ini perlu dilakukan encoding sebelum modelling.

#### 12. One Hot Encoding
**Penjelasan Gambar 2.4.59 (One Hot Encoding):** Gambar menampilkan kode one-hot encoding menggunakan pd.get_dummies().

```python
X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)
```

Kode di atas digunakan untuk one-hot encoding pada kolom kategorikal Sex dan Embarked menggunakan pd.get_dummies(). Parameter drop_first=True digunakan untuk menghindari dummy trap dengan menghapus salah satu kategori. Hasilnya, fitur kategorikal diubah menjadi kolom biner.

**Penjelasan Gambar 2.4.60 (Hasil One Hot Encoding):** Gambar menampilkan hasil dataframe setelah one-hot encoding, dengan kolom baru seperti Sex_male, Embarked_Q, Embarked_S yang berisi nilai biner (0/1).

**Penjelasan Gambar 2.4.61 (One Hot Encoding dengan OneHotEncoder):** Gambar menampilkan kode one-hot encoding menggunakan sklearn.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop_first=True)
encoded_data = encoder.fit_transform(X[['Sex', 'Embarked']])
df_ohe = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['Sex', 'Embarked']))
```

Kode di atas menggunakan OneHotEncoder dari sklearn untuk melakukan one-hot encoding pada kolom Sex dan Embarked. Encoder di-fit pada data kategorikal, lalu ditransformasi menjadi representasi numerik biner. Hasil df_ohe berisi matriks encoded yang siap digabungkan ke fitur model.

**Penjelasan Gambar 2.4.62 (Hasil One Hot Encoding):** Gambar menampilkan hasil one-hot encoding menggunakan OneHotEncoder dalam bentuk dataframe dengan kolom-kolom biner.

#### 13. Label Encoding
**Penjelasan Gambar 2.4.63 (Label Encoding):** Gambar menampilkan kode label encoding untuk semua kolom kategorikal dalam X.

```python
from sklearn.preprocessing import LabelEncoder

for col in X.select_dtypes(include=['object', 'bool']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])
```

Kode di atas digunakan untuk melakukan label encoding semua kolom kategorikal dalam X. Setiap kolom diubah menjadi nilai numerik menggunakan LabelEncoder, sehingga fitur kategorikal dapat digunakan oleh algoritma machine learning yang hanya menerima input numerik.

**Penjelasan Gambar 2.4.64 (Hasil Label Encoding):** Gambar menampilkan hasil dataframe setelah label encoding, dengan kolom Sex dan Embarked yang sudah berubah menjadi nilai numerik.

**Penjelasan Gambar 2.4.65 (Label Encoding untuk y):** Gambar menampilkan kode label encoding untuk target y.

```python
y = LabelEncoder().fit_transform(y)
```

Kode di atas melakukan label encoding pada target y, mengubah nilai kategorikal seperti 0/1 atau label teks menjadi angka yang dapat digunakan oleh model machine learning.

**Penjelasan Gambar 2.4.66 (Hasil Label Encoding):** Gambar menampilkan hasil label encoding pada target y dalam bentuk array numerik.

#### 14. Mengecek korelasi atribut
**Penjelasan Gambar 2.4.67 (Cek Korelasi Atribut):** Gambar menampilkan kode penggabungan fitur dan target serta perhitungan korelasi.

```python
df_combined = pd.concat([X, y], axis=1)
df_combined.corr()
```

Kode di atas menggabungkan kembali fitur (X) dan target (y) menjadi satu DataFrame lalu menghitung korelasi antar semua variabel untuk melihat hubungan fitur terhadap Survived.

**Penjelasan Gambar 2.4.68 (Korelasi Antar Variabel):** Gambar menampilkan hasil matriks korelasi antar semua variabel setelah encoding, menunjukkan hubungan setiap fitur terhadap Survived.

#### 15. Membagi dataset
**Penjelasan Gambar 2.4.69 (Pembagian Dataset):** Gambar menampilkan kode pembagian dataset menggunakan train_test_split.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Kode di atas membagi dataset menjadi training set (70%) dan testing set (30%) menggunakan train_test_split. Training dipakai untuk melatih model, sedangkan testing untuk mengevaluasi performanya agar tidak overfitting.

**Penjelasan Gambar 2.4.70 (Sampel Data Training):** Gambar menampilkan output `X_train.head()` yang memperlihatkan beberapa baris pertama dari data training.

#### 16. Scaling
**Penjelasan Gambar 2.4.71 (Scaling dengan StandardScaler):** Gambar menampilkan kode scaling menggunakan StandardScaler.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Kode di atas merupakan kode untuk melakukan standardization menggunakan StandardScaler.

**Penjelasan Gambar 2.4.72 (Sampel Data Training Setelah Scaling):** Gambar menampilkan output data training setelah dilakukan standardization, dengan nilai-nilai yang sudah terstandarisasi (mean=0, std=1).

**Penjelasan Gambar 2.4.73 (Scaling dengan MinMaxScaler):** Gambar menampilkan kode scaling menggunakan MinMaxScaler.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Kode di atas merupakan kode untuk melakukan normalization menggunakan MinMaxScaler.

**Penjelasan Gambar 2.4.74 (Sampel Data Training Setelah Scaling):** Gambar menampilkan output data training setelah dilakukan normalization, dengan nilai-nilai dalam rentang 0 hingga 1.

---

## 2.5. TUGAS & ANALISIS

### Exploratory Data Analysis
**Dataset:** https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers

1. Silahkan membaca dataset di atas menggunakan dataframe!
2. Jelaskan apa tujuan dari penggunaan dataset ini!
3. Definisikan atribut mana yang menjadi input dan atribut mana yang menjadi output/class/label!
4. Berikan penjelasan singkat untuk setiap atribut/variabel dari dataset tersebut

### Data Preprocessing
**Dataset:** https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data

1. Silahkan membaca dataset di atas menggunakan dataframe!
2. Buatlah visualisasi histogram untuk beberapa variabel numerik, lalu jelaskan pola distribusi yang terlihat!
3. Hitung dan visualisasikan korelasi antar variabel, kemudian jelaskan hubungan yang ditemukan!
4. Lakukan exploratory data analysis (EDA) dengan membuat descriptive statistics serta melakukan pengecekan missing value dan duplikasi. Jelaskan tindakan apa yang perlu dilakukan jika keduanya ditemukan!
5. Lakukan preprocessing lanjutan berupa encoding variabel kategorikal (one-hot & label encoding), melakukan train–test split (80:20), serta menerapkan standardization atau normalization pada fitur numerik. Jelaskan alasan penggunaan masing-masing teknik!

---

## 2.6. REFERENSI

- https://www.python.org/doc/
- https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
- https://numpy.org/
- Han, J., Pei, J., & Tong, H. (2022). *Data Mining: Concepts and Techniques* (4th ed.). Morgan Kaufmann/Elsevier.