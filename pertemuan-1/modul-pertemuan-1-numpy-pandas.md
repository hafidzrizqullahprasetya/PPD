# MODUL PRAKTIKUM PENAMBANGAN DATA
## PERTEMUAN I: NUMPY DAN PANDAS
**Program Studi:** Teknologi Rekayasa Perangkat Lunak – 2025

---

## 1.1. TUJUAN PEMBELAJARAN

A. Mahasiswa mampu memahami konsep dasar NumPy dan penggunaan array n-dimensi serta berbagai operasi manipulasi array.
B. Mahasiswa mengetahui berbagai manipulasi dan transformasi data pada pandas, termasuk pembuatan DataFrame.
C. Mahasiswa dapat menerapkan NumPy dan pandas untuk mengolah dan menganalisis data.

---

## 1.2. DASAR TEORI

### • NumPy
**NumPy (Numerical Python)** adalah pustaka Python yang digunakan untuk membangun dan mengelola array sebagai dasar komputasi numerik. NumPy dikembangkan pada tahun 2005 oleh Travis Oliphant sebagai proyek open-source. NumPy menawarkan kecepatan pemrosesan yang lebih tinggi, penggunaan memori yang efisien, serta dukungan terhadap struktur array berdimensi banyak. Kemampuan ini menjadikan NumPy penting dalam analisis numerik dan komputasi ilmiah.

Struktur utama NumPy adalah **ndarray**, yaitu array yang dapat berbentuk satu atau lebih dimensi, mulai dari 1D hingga 3D dan seterusnya. Array dapat dibentuk dari berbagai struktur sekuensial di Python maupun melalui fungsi internal untuk menghasilkan nilai tertentu.

Akses elemen pada array dilakukan melalui **indexing**, yaitu penunjukan posisi berdasarkan indeks yang dimulai dari nol. NumPy juga mendukung **slicing**, yakni pengambilan sebagian elemen berdasarkan rentang indeks tertentu, termasuk pada array multidimensi. Mekanisme ini memungkinkan manipulasi data secara efisien tanpa menyalin seluruh array.

### • Pandas
**Pandas** adalah pustaka Python yang dirancang untuk mempermudah pengolahan dan analisis data terstruktur. Pustaka ini menyediakan dua struktur data utama, yaitu **Series** (data satu dimensi) dan **DataFrame** (data dua dimensi berbentuk tabel). Dengan indeks yang fleksibel, Pandas memungkinkan pengguna melakukan manipulasi data secara efisien, mencakup pembersihan, transformasi, penggabungan, hingga analisis ringkas terhadap dataset.

Sebagai library utama dalam data analisis, Pandas menyediakan berbagai fungsi untuk memodifikasi struktur DataFrame, seperti menambah atau menghapus kolom, melakukan pengurutan, memfilter baris berdasarkan kondisi, serta melakukan grouping dan aggregation untuk merangkum data. Selain itu, Pandas juga mendukung operasi lanjutan seperti merging, joining, concatenation, dan reshaping melalui pivoting atau melting, yang menjadikannya sangat fleksibel untuk mengelola dataset dalam berbagai format.

Pandas dirancang dengan integrasi yang baik bersama NumPy, sehingga operasi numerik dan manipulasi array dapat dilakukan secara konsisten pada struktur DataFrame. Dengan performa yang efisien dan sintaks yang mudah dibaca, Pandas menjadi salah satu pustaka paling penting dalam ekosistem data science untuk penanganan data tabular.

---

## 1.3. ALAT DAN BAHAN

### • Perangkat Keras
- Komputer/Laptop

### • Perangkat Lunak
- Python3
- Google Colaboratory

---

## 1.4. LANGKAH PERCOBAAN

### A. NUMPY

#### 1) Install NumPy
NumPy telah terinstal di Google Colaboratory secara default. Jika belum terinstal, instalasi numpy dapat dilakukan dengan menggunakan PIP. Pada Jupyter Notebook termasuk Google Colab, instalasi NumPy dapat dilakukan seperti pada **Gambar 1.4.1**.

> **Penjelasan Gambar 1.4.1 – Instalasi NumPy:**
> Gambar menunjukkan perintah instalasi NumPy menggunakan PIP di Jupyter Notebook/Google Colab.

#### 2) Impor pustaka NumPy
Impor pustaka dengan syntax `import <pustaka>`. Nama alias juga dapat diberikan untuk mempersingkat pengetikan nama pemanggilan pustaka. Secara umum alias `np` digunakan untuk menyingkat numpy. Melakukan impor dan membuat alias dapat dilakukan seperti **Gambar 1.4.2**. Pada waktu pembuatan modul ini, pustaka NumPy yang terinstal di Google Colaboratory adalah pustaka NumPy versi 2.0.2.

> **Penjelasan Gambar 1.4.2 – Impor pustaka NumPy:**
> Gambar menunjukkan cara mengimpor library NumPy dengan alias `np` menggunakan perintah `import numpy as np`.

#### 3) Membuat array 1 dimensi
Membuat array dapat dilakukan dengan memanggil fungsi `array` seperti **Gambar 1.4.3**. Tipe data numpy berupa `ndarray` yang memiliki arti N-dimensional array.

> **Penjelasan Gambar 1.4.3 – Membuat array 1 dimensi:**
> Gambar menampilkan contoh pembuatan array 1 dimensi menggunakan `np.array()` dengan list Python sebagai input.

#### 4) Melihat shape atau bentuk array
Untuk melihat bentuk array yang telah dibuat, dapat dilakukan dengan memanggil atribut `shape` pada objek ndarray. Lakukan seperti **Gambar 1.4.4**, hasil tersebut menginformasikan bahwa array terdiri dari 1 dimensi dan terdiri dari 5 elemen.

> **Penjelasan Gambar 1.4.4 – Melihat bentuk array:**
> Gambar menampilkan penggunaan atribut `.shape` yang menghasilkan output `(5,)`, menunjukkan array 1 dimensi dengan 5 elemen.

#### 5) Membuat array 2 dimensi
Membuat array 2 dimensi dapat dilakukan pada fungsi yang sama. Untuk membuat array 2 dimensi, masukkan objek list seperti pembuatan dimensi 1 ke dalam list sehingga membentuk nested list seperti pada **Gambar 1.4.5**. Jika di cek bentuk array-nya maka akan menampilkan `(2, 3)`. Angka dua ini menunjukkan banyak elemen di sumbu 0 atau di list terluar, sementara itu angka tiga menunjukkan banyak elemen di sumbu 1 atau list bagian dalam.

> **Penjelasan Gambar 1.4.5 – Membuat array 2 dimensi dan mengecek bentuk array:**
> Gambar menampilkan pembuatan array 2 dimensi menggunakan nested list dan output shape `(2, 3)`.

#### 6) Membuat array 3 dimensi
Serupa dengan konsep pembuatan array 2 dimensi, array 3 dimensi akan memuat elemen 2 dimensi. Contoh pembuatan array 3 dimensi dapat dilihat di **Gambar 1.4.6**.

> **Penjelasan Gambar 1.4.6 – Membuat array 3 dimensi:**
> Gambar menampilkan pembuatan array 3 dimensi menggunakan nested list yang lebih dalam.

#### 7) Membuat array dengan String sebagai member

> **Penjelasan Gambar 1.4.7 – Array String:**
> Gambar menampilkan contoh pembuatan array dengan elemen bertipe data String.

#### 8) Melakukan indexing di array 1 dimensi
Indexing tidak jauh berbeda dengan fungsi native indexing untuk tipe data list. Mengakses elemen dengan indeks ke-n dapat dilakukan dengan `objek_ndarray[n]` seperti **Gambar 1.4.8** baris ke 3. Perlu diingat bahwa indeks selalu mulai dari 0.

> **Penjelasan Gambar 1.4.8 – Indexing array 1 dimensi:**
> Gambar menampilkan cara mengakses elemen array 1 dimensi menggunakan indeks `[n]`.

#### 9) Melakukan indexing di array 2 dimensi

> **Penjelasan Gambar 1.4.9 – Indexing array 2 dimensi:**
> Gambar menampilkan cara mengakses elemen array 2 dimensi dengan format `[baris, kolom]`.

#### 10) Melakukan indexing di array 3 dimensi

> **Penjelasan Gambar 1.4.10 – Indexing array 3 dimensi:**
> Gambar menampilkan cara mengakses elemen array 3 dimensi dengan format `[sumbu0, sumbu1, sumbu2]`.

#### 11) Membuat array dari contoh
Diberikan sebuah array dengan bentuk seperti **Gambar 1.4.11**. Bagian yang tidak terlihat dianggap memiliki nilai 0.

> **Penjelasan Gambar 1.4.11 – Contoh array 3 dimensi:**
> Gambar menampilkan visualisasi bentuk array 3 dimensi dalam struktur kubus.

Jika Anda diminta untuk membuatkan dalam bentuk kode, maka akan terlihat seperti **Gambar 1.4.12**.

> **Penjelasan Gambar 1.4.12 – Implementasi Array:**
> Gambar menampilkan kode implementasi pembuatan array 3 dimensi sesuai visualisasi sebelumnya.

#### 12) Array slicing
NumPy menerapkan cara yang sama dengan cara slicing tipe data List untuk melakukan slicing tipe data ndarray. Sebagai contoh Anda ingin untuk mengambil **Gambar 1.4.13** yang diberi tanda merah (3 dan 0), maka Anda harus bisa menentukan indeks lokasi target tersebut.

> **Penjelasan Gambar 1.4.13 – Target slicing:**
> Gambar menampilkan array 3 dimensi dengan target elemen yang ditandai merah (nilai 3 dan 0).

Di sumbu 0, target tersebut berada di indeks ke 2. Di sumbu 1, target berada di indeks ke 1 dan 2 atau dapat disebut indeks ke 1 dan selanjutnya. Di sumbu 2, target hanya di indeks ke 0. Dengan demikian, penulisan kode slicing dapat dilakukan seperti **Gambar 1.4.14**.

> **Penjelasan Gambar 1.4.14 – Contoh penerapan Slicing:**
> Gambar menampilkan hasil kode slicing dengan notasi `[2, 1:, 0]` untuk mengambil target elemen.

#### 13) Mengonversi tipe data
Melakukan konversi tipe data dapat dilakukan dengan fungsi `astype` dengan memasukkan target kelas.

> **Penjelasan Gambar 1.4.15 – Konversi tipe data di NumPy:**
> Gambar menampilkan penggunaan fungsi `.astype()` untuk mengubah tipe data array (misalnya dari int ke float).

#### 14) Reshape ke dimensi lebih tinggi
Untuk mengubah bentuk array ke dimensi lebih tinggi, pastikan bahwa banyak elemen tetap harus sama. Sebagai contoh **Gambar 1.4.16**, banyak elemen array asli adalah 12, ketika ingin mengubah bentuk ke 2 dimensi, total banyak elemen tetap harus 12. Sebagai contoh perhitungannya, reshaping ke dimensi 2 dilakukan dengan banyak elemen di sumbu 0 adalah 3 dan di sumbu 1 adalah 4, total dari banyak elemen tersebut adalah 12 (cara perhitungan: 3 x 4 = 12). Hasil tersebut tetap sama dengan banyak elemen di array asli.

> **Penjelasan Gambar 1.4.16 – Contoh Reshape ke dimensi lebih tinggi:**
> Gambar menampilkan penggunaan `.reshape(3, 4)` untuk mengubah array 1D menjadi 2D.

#### 15) Reshape ke dimensi 1
Untuk melakukan reshape ke dimensi yang lebih rendah, Anda perlu melakukan reshape ke dimensi 1 terlebih dahulu. Untuk mengubah bentuk ke dimensi 1 dapat dilakukan seperti **Gambar 1.4.17**. Setelah melakukan pengubahan bentuk ke dimensi 1, perubahan ke dimensi ke lebih tinggi dapat dilakukan.

> **Penjelasan Gambar 1.4.17 – Reshape ke dimensi 1:**
> Gambar menampilkan penggunaan `.reshape(-1)` atau `.flatten()` untuk mengubah array menjadi 1D.

#### 16) Perbedaan fungsi len dan atribut shape
Telah kita mengetahui bahwa `shape` akan menampilkan bentuk array, sementara itu, `len` hanya mampu menghitung banyak elemen yang ada di sumbu 0. Masih menggunakan **Gambar 1.4.11**, berikut akan membandingkan hasil perbedaan len dan shape. Di hasil **Gambar 1.4.18**, terlihat bahwa `len` hanya memunculkan angka 4 karena di sumbu 0 hanya ada 4 elemen.

> **Penjelasan Gambar 1.4.18 – Len vs Shape:**
> Gambar menampilkan perbandingan output `len()` yang hanya menampilkan jumlah elemen sumbu 0, dan `shape` yang menampilkan bentuk lengkap array.

#### 17) Iterasi ndarray

> **Penjelasan Gambar 1.4.19 – Iterasi langsung:**
> Gambar menampilkan iterasi langsung pada ndarray menggunakan perulangan `for`.

> **Penjelasan Gambar 1.4.20 – Iterasi dengan indeks:**
> Gambar menampilkan iterasi ndarray menggunakan indeks dengan bantuan `range()` dan `len()`.

#### 18) Melakukan join pada array dimensi 1
Menggabungkan array dapat dilakukan dengan fungsi `concatenate`. Untuk menggabungkan array berdimensi 1 maka sumbu (axis) di set ke 0 seperti **Gambar 1.4.21**.

> **Penjelasan Gambar 1.4.21 – Concatenate 2 array:**
> Gambar menampilkan penggunaan `np.concatenate()` untuk menggabungkan dua array 1D dengan `axis=0`.

#### 19) Melakukan join pada array dimensi 2
Di dimensi 2, Anda bisa memilih untuk menggabungkan array ke sumbu 1 atau 0. Penggabungan array ke sumbu 1 akan terlihat seperti **Gambar 1.4.22**. Penggabungan array ke sumbu 0 akan terlihat seperti **Gambar 1.4.23**.

> **Penjelasan Gambar 1.4.22 – Menambahkan Array di sumbu 1:**
> Gambar menampilkan penggabungan array 2D secara horizontal (axis=1).

> **Penjelasan Gambar 1.4.23 – Menambahkan Array di sumbu 0:**
> Gambar menampilkan penggabungan array 2D secara vertikal (axis=0).

#### 20) Array Split
Melakukan split pada objek ndarray dapat dilakukan dengan fungsi `array_split` seperti **Gambar 1.4.24**.

> **Penjelasan Gambar 1.4.24 – Array Split:**
> Gambar menampilkan penggunaan `np.array_split()` untuk memecah array menjadi beberapa bagian.

#### 21) Array Search
Melakukan pencarian dapat dilakukan dengan fungsi `where` dengan memasukkan kondisi seperti **Gambar 1.4.25**.

> **Penjelasan Gambar 1.4.25 – Mencari elemen dalam array:**
> Gambar menampilkan penggunaan `np.where()` untuk mencari indeks elemen yang memenuhi kondisi tertentu.

#### 22) Sorting
Melakukan sorting dapat dilakukan dengan fungsi `sort`. Terdapat beberapa parameter yang bisa digunakan. Penjelasan lengkap mengenai parameter sorting dapat dibaca di dokumentasi NumPy. Sebagai contoh **Gambar 1.4.26**, sorting dilakukan secara ascending.

> **Penjelasan Gambar 1.4.26 – Sorting NumPy:**
> Gambar menampilkan penggunaan `np.sort()` untuk mengurutkan elemen array secara ascending.

#### 23) NumPy Random
Impor package `random` dari pustaka numpy untuk menggunakan numpy random. Gunakan fungsi `rand` untuk membuat nilai random dengan shape n. Untuk membuat nilai random dengan shape dimensi 1 dan 2 elemen, maka akan terlihat seperti **Gambar 1.4.27**.

> **Penjelasan Gambar 1.4.27 – Numpy Random:**
> Gambar menampilkan penggunaan `np.random.rand()` untuk menghasilkan nilai acak.

#### 24) Operasi penjumlahan
NumPy memiliki operasi penjumlahan array. Operasi tersebut dapat diakses dengan fungsi `add` seperti **Gambar 1.4.28**.

> **Penjelasan Gambar 1.4.28 – Penjumlahan Array:**
> Gambar menampilkan penggunaan `np.add()` untuk menjumlahkan dua array.

#### 25) Operasi Summation (Total jumlah)
NumPy juga memberikan operasi untuk melakukan kalkulasi total jumlah semua elemen.

> **Penjelasan Gambar 1.4.29 – Summation:**
> Gambar menampilkan penggunaan `np.sum()` untuk menghitung total seluruh elemen array.

#### 26) Product of elements
Untuk menghitung perkalian semua elemen, dapat dilakukan dengan menggunakan fungsi `prod`.

> **Penjelasan Gambar 1.4.30 – Product of elements NumPy:**
> Gambar menampilkan penggunaan `np.prod()` untuk menghitung hasil perkalian seluruh elemen array.

---

### B. DATAFRAME PANDAS

#### 1) Instal Pandas
Pandas telah terinstal di Google Colaboratory secara default. Jika belum terinstal, instalasi pandas dapat dilakukan dengan menggunakan PIP. Pada Jupyter Notebook termasuk Google Colaboratory, instalasi Pandas dapat dilakukan seperti **Gambar 1.4.31**.

> **Penjelasan Gambar 1.4.31 – Instalasi pandas:**
> Gambar menampilkan perintah instalasi pandas menggunakan PIP.

#### 2) Import Library
Import library dengan syntax `import <pustaka>`. Nama alias juga dapat diberikan untuk mempersingkat pengetikan nama pemanggilan pustaka. Secara umum alias `pd` digunakan untuk menyingkat pandas. Pada percobaan ini, kita memerlukan library pandas dan numpy, sehingga kita akan mengimpor kedua library tersebut.

> **Penjelasan Gambar 1.4.32 – Import Library pandas:**
> Gambar menampilkan perintah `import pandas as pd` dan `import numpy as np`.

#### 3) Membuat DataFrame
DataFrame dibuat dengan memberikan sebuah list Python sebagai input ke fungsi `pd.DataFrame()`. Setiap elemen dalam list otomatis menjadi satu baris dalam DataFrame, sementara indeks dihasilkan secara default mulai dari 0.

> **Penjelasan Gambar 1.4.33 – Membuat DataFrame:**
> Gambar menampilkan pembuatan DataFrame sederhana dari list Python.

Kita juga dapat menambahkan parameter `columns` untuk menetapkan nama kolom dan parameter `index` untuk memberikan label indeks custom. Hasilnya, DataFrame tampil dalam format tabular seperti pada **Gambar 1.4.34**.

> **Penjelasan Gambar 1.4.34 – Membuat DataFrame:**
> Gambar menampilkan pembuatan DataFrame dengan parameter `columns` dan `index` custom.

#### 4) Menambahkan Kolom
Penambahan kolom pada dataframe dapat dilakukan dengan menambahkan `df['Nama Kolom']`.

> **Penjelasan Gambar 1.4.35 – Menambahkan Kolom:**
> Gambar menampilkan penambahan kolom baru dengan notasi `df['NamaKolom'] = nilai`.

Penambahan kolom juga dapat dilakukan dengan fungsi `df.insert` seperti pada **Gambar 1.4.36**.

> **Penjelasan Gambar 1.4.36 – Menambahkan Kolom:**
> Gambar menampilkan penggunaan fungsi `df.insert()` untuk menambahkan kolom.

Kolom juga dapat ditambahkan pada posisi tertentu dengan menambahkan parameter `loc`, `column`, `value` pada fungsi `df.insert`.

> **Penjelasan Gambar 1.4.37 – Menambahkan Kolom:**
> Gambar menampilkan penggunaan `df.insert(loc, column, value)` untuk menyisipkan kolom pada posisi tertentu.

#### 5) Menambahkan Multi Kolom
Penambahan multi kolom dapat dilakukan dengan menggunakan fungsi `assign()`, yaitu dengan memberikan beberapa pasangan nama kolom dan nilai sekaligus.

> **Penjelasan Gambar 1.4.38 – Menambahkan Multi Kolom:**
> Gambar menampilkan penggunaan `df.assign()` untuk menambahkan beberapa kolom sekaligus.

Penambahan multi kolom juga dapat dilakukan dengan membuat sebuah dictionary yang berisi pasangan nama kolom dan nilai, kemudian memasukkannya ke dalam DataFrame secara bersamaan.

> **Penjelasan Gambar 1.4.39 – Menambahkan Multi Kolom:**
> Gambar menampilkan penambahan multi kolom menggunakan dictionary.

#### 6) Menghapus Kolom
Penghapusan kolom dilakukan menggunakan fungsi `drop()` seperti pada **Gambar 1.4.40**. Parameter nama kolom dan axis digunakan untuk menentukan kolom yang akan dihapus. Parameter `inplace=True` membuat perubahan diterapkan langsung pada DataFrame tanpa membuat salinan baru.

> **Penjelasan Gambar 1.4.40 – Menghapus Kolom:**
> Gambar menampilkan penggunaan `df.drop()` dengan parameter `axis=1` dan `inplace=True`.

Penghapusan kolom juga dapat dilakukan dengan fungsi `df.pop()` (**Gambar 1.4.41**) dan juga perintah `del` (**Gambar 1.4.42**).

> **Penjelasan Gambar 1.4.41 – Menghapus Kolom dengan Fungsi pop:**
> Gambar menampilkan penggunaan `df.pop()` untuk menghapus dan mengembalikan kolom.

> **Penjelasan Gambar 1.4.42 – Menghapus Kolom dengan Perintah del:**
> Gambar menampilkan penggunaan perintah `del df['NamaKolom']`.

#### 7) Menghapus Multi Kolom
Penghapusan multi kolom dapat dilakukan dengan fungsi `df.drop()` seperti gambar berikut.

> **Penjelasan Gambar 1.4.43 – Menghapus Multi Kolom:**
> Gambar menampilkan penghapusan beberapa kolom sekaligus menggunakan `df.drop()` dengan list nama kolom.

#### 8) Sorting
Mengurutkan data berdasarkan indeks pada dataframe dapat dilakukan dengan fungsi `df.sort_index()` seperti pada **Gambar 1.4.45**.

> **Penjelasan Gambar 1.4.44 – Membuat DataFrame:**
> Gambar menampilkan DataFrame awal yang akan diurutkan.

> **Penjelasan Gambar 1.4.45 – Sorting Data:**
> Gambar menampilkan penggunaan `df.sort_index()` untuk mengurutkan berdasarkan indeks baris.

Parameter `axis=1` digunakan untuk mengurutkan kolom berdasarkan indeks kolom.

> **Penjelasan Gambar 1.4.46 – Sorting Data:**
> Gambar menampilkan penggunaan `df.sort_index(axis=1)` untuk mengurutkan kolom.

#### 9) Sorting Value
Untuk mengurutkan data berdasarkan nilai pada kolom tertentu, digunakan fungsi `sort_values` dengan parameter nama kolom seperti pada **Gambar 1.4.47** dan **Gambar 1.4.48**.

> **Penjelasan Gambar 1.4.47 – Sorting Data Berdasarkan Kolom Tertentu:**
> Gambar menampilkan penggunaan `df.sort_values()` berdasarkan kolom tertentu.

> **Penjelasan Gambar 1.4.48 – Sorting Data Berdasarkan Kolom Tertentu:**
> Gambar menampilkan variasi sorting dengan parameter tambahan seperti `ascending=False`.

#### 10) Filtering
Filtering dataset dilakukan dengan memberikan kondisi logika pada DataFrame seperti pada **Gambar 1.4.49**.

> **Penjelasan Gambar 1.4.49 – Filtering Data dengan Kondisi:**
> Gambar menampilkan filtering dengan kondisi logika tunggal.

Filtering juga dapat dilakukan menggunakan lebih dari satu kondisi seperti pada **Gambar 1.4.50**.

> **Penjelasan Gambar 1.4.50 – Filtering Data dengan Kondisi:**
> Gambar menampilkan filtering dengan multiple conditions menggunakan operator `&` atau `|`.

Filtering juga dapat dilakukan menggunakan fungsi `isin()`, yang mengecek apakah nilai pada kolom termasuk dalam sebuah list tertentu.

> **Penjelasan Gambar 1.4.51 – Filtering Data dengan Fungsi isin():**
> Gambar menampilkan penggunaan `df['kolom'].isin([list nilai])`.

Filtering dapat dilakukan menggunakan fungsi `query()`, yang memungkinkan penulisan kondisi logis dalam bentuk ekspresi string.

> **Penjelasan Gambar 1.4.52 – Filtering Data dengan Query:**
> Gambar menampilkan penggunaan `df.query()` dengan ekspresi string.

> **Penjelasan Gambar 1.4.53 – Filtering Data dengan Query:**
> Gambar menampilkan variasi penggunaan `query()` dengan kondisi lebih kompleks.

Filtering juga dapat dilakukan menggunakan fungsi `str.contains()`.

> **Penjelasan Gambar 1.4.54 – Filtering Data dengan fungsi str.contains():**
> Gambar menampilkan filtering berdasarkan substring dalam kolom string.

#### 11) Grouping

> **Penjelasan Gambar 1.4.55 – Membuat DataFrame:**
> Gambar menampilkan DataFrame awal untuk operasi grouping.

> **Penjelasan Gambar 1.4.56 – Menggabungkan DataFrame:**
> Gambar menampilkan proses penggabungan DataFrame sebelum grouping.

Fungsi `groupby()` digunakan untuk mengelompokkan data berdasarkan nilai pada kolom Pekerjaan. Atribut `.groups` menampilkan indeks baris yang termasuk dalam setiap kelompok, sehingga terlihat pembagian data sesuai kategori pekerjaan.

> **Penjelasan Gambar 1.4.57 – Menampilkan Data Menurut Pekerjaan:**
> Gambar menampilkan hasil `groupby()` dengan atribut `.groups` yang menunjukkan pengelompokan data.

Objek hasil `groupby()` diiterasi untuk menampilkan setiap kelompok data secara terpisah. Variabel pekerjaan berisi nama kelompok, sedangkan group berisi DataFrame kecil yang memuat baris-baris sesuai kategori tersebut.

> **Penjelasan Gambar 1.4.58 – Menampilkan Satu Kategori Tertentu:**
> Gambar menampilkan penggunaan `get_group()` untuk mengambil satu kelompok tertentu.

#### 12) Aggregation
Operasi agregate dilakukan dengan menerapkan fungsi seperti mean, sum, min, atau max pada data yang telah dikelompokkan menggunakan `groupby()`.

> **Penjelasan Gambar 1.4.59 – Aggregation:**
> Gambar menampilkan operasi agregasi dasar pada hasil groupby.

Operasi aggregate dilakukan dengan mengelompokkan data berdasarkan kolom Pekerjaan menggunakan `groupby()`, kemudian menerapkan fungsi `mean()` untuk menghitung rata-rata pada kolom numerik seperti pada **Gambar 1.4.60**.

> **Penjelasan Gambar 1.4.60 – Aggregation:**
> Gambar menampilkan penggunaan `groupby().mean()` untuk menghitung rata-rata per kelompok.

Contoh operasi aggregate lain dengan menerapkan beberapa fungsi secara sekaligus melalui `agg()`, setelah data dikelompokkan berdasarkan kolom Pekerjaan. Pada contoh ini, fungsi mean, sum, dan std digunakan untuk menghitung rata-rata, total, dan standar deviasi pada kolom Umur dan Gaji untuk setiap kelompok pekerjaan.

> **Penjelasan Gambar 1.4.61 – Aggregation:**
> Gambar menampilkan penggunaan `agg()` dengan dictionary fungsi untuk multiple aggregations.

#### 13) Merge
Penggabungan dataframe dilakukan menggunakan fungsi `merge()`. Parameter `on='Id'` menentukan key column yang digunakan, sedangkan `sort=True` mengurutkan hasil berdasarkan key value tersebut.

> **Penjelasan Gambar 1.4.62 – Merging DataFrame:**
> Gambar menampilkan penggunaan `pd.merge()` dengan parameter `on` dan `sort`.

> **Penjelasan Gambar 1.4.63 – Merging DataFrame:**
> Gambar menampilkan hasil penggabungan DataFrame berdasarkan key column.

Penggabungan juga dapat dilakukan menggunakan `merge()` dengan parameter `how='left'`, sehingga semua baris dari DataFrame karyawan tetap ditampilkan, sementara data dari DataFrame pekerjaan hanya ditambahkan jika nilai Id Pekerjaan cocok.

> **Penjelasan Gambar 1.4.64 – Merging DataFrame:**
> Gambar menampilkan left join menggunakan parameter `how='left'`.

#### 14) Join
Join dapat dilakukan dengan menggunakan `merge()` pada key column yang sama, di mana parameter `how='left'` memastikan seluruh data dari tabel kiri ditampilkan, sedangkan `suffixes` digunakan untuk membedakan nama kolom yang berasal dari kedua DataFrame.

> **Penjelasan Gambar 1.4.65 – Joining DataFrame:**
> Gambar menampilkan merge dengan parameter `suffixes` untuk membedakan kolom duplikat.

Join juga dapat dilakukan dengan fungsi `join()`. Join bekerja berdasarkan indeks, sehingga hasilnya bisa berbeda dari `merge()` yang menggunakan key column. Parameter `how='left'` menjaga seluruh baris DataFrame kiri tetap tampil, sementara `rsuffix` digunakan untuk membedakan kolom yang duplikat dari DataFrame kanan.

> **Penjelasan Gambar 1.4.66 – Joining DataFrame:**
> Gambar menampilkan penggunaan `df.join()` dengan parameter `how='left'` dan `rsuffix`.

#### 15) Concatenate
Concatenate dilakukan dengan menggunakan `pd.concat()` pada `axis=0`, sehingga DataFrame kedua ditambahkan sebagai baris baru di bawah DataFrame pertama. Parameter `ignore_index=True` digunakan agar indeks diurutkan ulang secara otomatis.

> **Penjelasan Gambar 1.4.67 – Concatenate:**
> Gambar menampilkan `pd.concat()` dengan `axis=0` untuk penggabungan vertikal.

Dengan menambahkan parameter `axis=0` dan parameter `join='outer'` dapat menggabungkan semua kolom dari kedua DataFrame. Kolom yang tidak dimiliki oleh salah satu DataFrame akan berisi nilai NaN, sementara `ignore_index=True` mengatur ulang indeks agar berurutan.

> **Penjelasan Gambar 1.4.68 – Concatenate:**
> Gambar menampilkan `pd.concat()` dengan `join='outer'` yang menghasilkan NaN untuk kolom yang tidak ada.

Jika menggunakan `axis=1`, penggabungan dilakukan secara horizontal sehingga kolom baru ditambahkan di samping DataFrame awal.

> **Penjelasan Gambar 1.4.69 – Concatenate:**
> Gambar menampilkan `pd.concat()` dengan `axis=1` untuk penggabungan horizontal.

#### 16) Reset Indeks

> **Penjelasan Gambar 1.4.70 – Reset Index:**
> Gambar menampilkan reset index menggunakan `pd.concat()` dengan `ignore_index=True`.

Reset index dilakukan dengan menggunakan `pd.concat()` dan menetapkan `ignore_index=True`, sehingga indeks lama dari kedua DataFrame diabaikan dan diganti dengan indeks baru yang berurutan setelah proses penggabungan.

> **Penjelasan Gambar 1.4.71 – Reset Index:**
> Gambar menampilkan penggunaan `reset_index(drop=True)` untuk mereset indeks DataFrame.

Reset indeks juga dapat dilakukan dengan fungsi `reset_index(drop=True)`, yang menghapus indeks lama dan menggantinya dengan indeks baru yang berurutan. Parameter `drop=True` memastikan indeks sebelumnya tidak ikut disimpan sebagai kolom baru.

#### 17) Set Indeks

> **Penjelasan Gambar 1.4.72 – Set Index:**
> Gambar menampilkan DataFrame sebelum perubahan indeks.

> **Penjelasan Gambar 1.4.73 – Set Index:**
> Gambar menampilkan DataFrame setelah menggunakan `set_index()`.

Set indeks dapat dilakukan dengan memilih baris tertentu sebagai referensi indeks baru. Pada **Gambar 1.4.73**, nilai pada baris ke-2 ditampilkan karena DataFrame sebelumnya telah atau akan diubah indeksnya menggunakan nilai kolom tertentu melalui `set_index()`, sehingga baris tersebut ditampilkan sesuai indeks barunya.

Set indeks juga dapat dilakukan dengan fungsi `set_index()` yang menetapkan kolom ID sebagai indeks baru DataFrame.

> **Penjelasan Gambar 1.4.74 – Set Index:**
> Gambar menampilkan penggunaan `set_index('ID')` dan akses data menggunakan `.loc['003']`.

`kiri.loc['003']` menampilkan data dengan indeks bernilai '003', sesuai indeks baru yang telah ditetapkan menggunakan kolom ID.

#### 18) Pivot Tabel

> **Penjelasan Gambar 1.4.75 – Membuat DataFrame:**
> Gambar menampilkan DataFrame awal untuk operasi pivot.

Pivot tabel dibuat menggunakan `pd.pivot_table()` dengan menetapkan kolom Gender dan Pekerjaan sebagai pembentuk kolom baru. Agregasi dilakukan melalui parameter `aggfunc`, untuk menjumlahkan nilai Gaji dan menghitung jumlah data berdasarkan Pekerjaan seperti pada **Gambar 1.4.76**.

> **Penjelasan Gambar 1.4.76 – Pivot Table:**
> Gambar menampilkan penggunaan `pd.pivot_table()` dengan parameter `values`, `index`, `columns`, dan `aggfunc`.

Menghitung sum, mean, median, dan lain-lain dapat dilakukan dengan `np.sum` seperti pada **Gambar 1.4.77**.

> **Penjelasan Gambar 1.4.77 – Pivot Table:**
> Gambar menampilkan pivot table dengan agregasi menggunakan `np.sum`.

#### 19) Melt
Melt digunakan untuk mengubah DataFrame dari format lebar (wide) menjadi format panjang (long).

> **Penjelasan Gambar 1.4.78 – Membuat DataFrame:**
> Gambar menampilkan DataFrame dalam format wide.

> **Penjelasan Gambar 1.4.79 – Melt Table:**
> Gambar menampilkan hasil `pd.melt()` yang mengubah format wide menjadi long.

Kolom yang ditentukan pada `id_vars` tetap dipertahankan, sedangkan kolom pada `value_vars` diubah menjadi dua kolom baru, yaitu `variable` dan `value` sehingga tiap nilai mata pelajaran ditampilkan sebagai baris terpisah.

#### 20) Lambda Function
Berikut penggunaan fungsi lambda dengan dua parameter untuk melakukan operasi penjumlahan.

> **Penjelasan Gambar 1.4.80 – Lambda Function:**
> Gambar menampilkan lambda function dengan dua parameter untuk penjumlahan.

Berikut penggunaan fungsi lambda dengan satu parameter untuk melakukan operasi perkalian.

> **Penjelasan Gambar 1.4.81 – Lambda Function:**
> Gambar menampilkan lambda function dengan satu parameter untuk perkalian.

> **Penjelasan Gambar 1.4.82 – Lambda Function:**
> Gambar menampilkan variasi lambda function.

Berikut penggunaan fungsi lambda dengan kondisi bertingkat untuk menentukan nilai berdasarkan skor Matematika pada DataFrame.

> **Penjelasan Gambar 1.4.83 – Lambda Function:**
> Gambar menampilkan lambda function dengan kondisi bertingkat menggunakan `apply()`.

#### 21) Membaca File
Membaca file dari Google Drive dapat dilakukan dengan menggunakan `drive.mount()`.

> **Penjelasan Gambar 1.4.84 – Menghubungkan Google Colaboratory dengan Google Drive:**
> Gambar menampilkan penggunaan `drive.mount('/content/drive')` untuk mounting Google Drive.

Lalu untuk membaca file dataset dari Google Drive, dapat dilakukan dengan menggunakan `pd.read_csv()` seperti pada **Gambar 1.4.85**.

> **Penjelasan Gambar 1.4.85 – Membaca File dari Google Drive:**
> Gambar menampilkan penggunaan `pd.read_csv()` untuk membaca file dataset dari Google Drive.

---

## 1.5. TUGAS & ANALISIS

1. Buatlah array NumPy 2D berukuran 4×4 berisi angka acak (0–50), kemudian ubah bentuknya menjadi array 2×8 dan tampilkan hanya elemen-elemen yang bernilai lebih dari 25.

2. Buatlah dataframe untuk data karyawan yang berisi kolom Nama, Pekerjaan, dan Gaji. Lalu lakukan operasi `groupby()` pada kolom Pekerjaan dan hitung nilai maksimum, minimum, serta standar deviasi untuk kolom 'Gaji' menggunakan `agg()`.

3. Buat dua DataFrame untuk data karyawan dengan kolom Emp_id, Nama, dan Divisi, serta DataFrame gaji dengan kolom ID, Gaji, dan Status (masing-masing minimal tiga baris). Setelah itu, gabungkan kedua DataFrame tersebut menggunakan `merge()` dengan `left_on='Emp_id'` dan `right_on='ID'`, terapkan left join, dan gunakan `suffixes` untuk membedakan kolom yang sama.

---

## 1.6. REFERENSI

- https://numpy.org/
- https://pandas.pydata.org/
- https://www.python.org/doc/

---

*Catatan: Penjelasan gambar disusun berdasarkan deskripsi kontekstual yang terdapat dalam teks modul, karena gambar asli tidak dapat divisualisasikan secara langsung.*