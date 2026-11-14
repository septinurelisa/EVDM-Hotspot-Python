[README.md](https://github.com/user-attachments/files/34844391/README.md)
# EVDM-Hotspot (Python / Streamlit)

Eksplorasi dan visualisasi pola hotspot kebakaran hutan dan lahan di Indonesia tahun 2020–2025, dilengkapi analisis regresi linier berganda dan dashboard interaktif **Streamlit**. Proyek UTS mata kuliah Eksplorasi dan Visualisasi Data Multivariat (EVDM).

**Penulis:** Septi Nurelisa, Program Studi Statistika, Universitas AKPRIND Indonesia

## Tujuan
1. Mendeskripsikan karakteristik hotspot (univariat).
2. Mengeksplorasi hubungan antarvariabel (bivariat dan multivariat).
3. Membangun dan mengevaluasi model regresi linier berganda untuk FRP, lengkap dengan interval konfidensi 95%.
4. Menyajikan hasilnya dalam dashboard interaktif.

## Sumber data
NASA FIRMS (https://firms.modaps.eosdis.nasa.gov/), wilayah Indonesia, 2020–2025.

| Berkas | Isi | Dipakai untuk |
|---|---|---|
| `fire_archive_M-C61_43157.csv` | MODIS Collection 6.1 (Terra dan Aqua) | **data utama analisis** (disertakan di `data/raw`) |
| `fire_archive_J1V-C2_43158.csv` | VIIRS NOAA-20 | perbandingan jumlah per tahun (tidak disertakan karena besar) |
| `fire_archive_J2V-C2_43159.csv` | VIIRS NOAA-21 | perbandingan jumlah per tahun (tidak disertakan karena besar) |

Hasil perbandingan VIIRS sudah disimpan di `data/processed/ringkasan_sensor_tahun.csv`, sehingga aplikasi tetap jalan tanpa berkas VIIRS. Data analisis: hotspot bertipe kebakaran vegetasi (`type = 0`), **117.070 observasi, 14 variabel**. Data FIRMS milik NASA dan bebas digunakan dengan mencantumkan sumber.

> Hotspot adalah deteksi anomali termal satelit, bukan kebakaran yang terverifikasi. FRP bukan luas area terbakar.

## Struktur folder
```
EVDM-Hotspot-Python/
├── app.py                 # aplikasi Streamlit
├── evdm.py                # fungsi bersama (data, statistik, regresi)
├── requirements.txt
├── README.md
├── .streamlit/config.toml # tema
├── data/
│   ├── raw/               # CSV asli MODIS
│   └── processed/         # hasil 01_cleaning.py
├── scripts/
│   ├── 01_cleaning.py
│   ├── 02_univariate.py
│   ├── 03_bivariate.py
│   ├── 04_multivariate.py
│   ├── 05_regression.py
│   └── _plot.py           # pengaturan grafik
├── output/
│   ├── plots/             # grafik (png)
│   ├── tables/            # tabel hasil (csv)
│   └── logs/              # keluaran lengkap tiap script
└── docs/
    └── presentasi.md
```

## Instalasi
Butuh Python 3.10 atau lebih baru.
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Menjalankan
1. (Opsional, hasil sudah disertakan) jalankan analisis dari folder proyek:
```bash
python scripts/01_cleaning.py
python scripts/02_univariate.py
python scripts/03_bivariate.py
python scripts/04_multivariate.py
python scripts/05_regression.py
```
2. Jalankan dashboard:
```bash
streamlit run app.py
```
Browser akan terbuka di http://localhost:8501.

## Menggunakan dashboard
| Menu | Fungsi |
|---|---|
| Beranda | Ringkasan penelitian dan statistik utama |
| Dataset | Tabel interaktif, filter, informasi variabel, unduh CSV |
| Univariat | Pilih variabel dan jenis grafik; statistik dan interpretasi otomatis |
| Bivariat | Pilih dua variabel; grafik menyesuaikan tipe variabel |
| Multivariat | Heatmap korelasi dan scatterplot interaktif (warna dan ukuran titik) |
| Regresi | Pilih respons dan minimal 2 prediktor; koefisien, IK 95%, uji t dan F, R², diagnostik |
| Peta Hotspot | Peta titik dengan filter tahun, confidence, dan waktu (butuh internet untuk peta dasar) |
| Tentang | Metode dan keterbatasan |

## Metode
Statistik deskriptif, korelasi Pearson/Spearman, uji Mann-Whitney (Wilcoxon), Kruskal-Wallis, Chi-kuadrat, regresi linier berganda (OLS, `statsmodels`), interval konfidensi 95% koefisien (dihitung manual dan dicocokkan dengan `conf_int()`), uji F dan uji t, VIF, Breusch-Pagan, Shapiro-Wilk, Durbin-Watson, Cook's distance, dan galat baku robust HC3.

## Hasil utama
| Model | Spesifikasi | Adjusted R² |
|---|---|---|
| A | FRP ~ brightness + bright_t31 | 0,4999 |
| B | ln(FRP) ~ brightness + bright_t31 | 0,5584 |
| **C (final)** | ln(FRP) ~ brightness + bright_t31 + scan | **0,9043** |

Persamaan model final: `ln(FRP) = −10,2800 + 0,05807·brightness − 0,02227·bright_t31 + 0,71233·scan`. Nilai AIC dari `statsmodels` berbeda sekitar 2 poin dari `AIC()` di R karena perbedaan cara menghitung jumlah parameter; koefisien dan R² identik.

## Keterbatasan
Data tidak independen (ruang-waktu), asumsi homoskedastisitas dan normalitas residual tidak terpenuhi sempurna, FRP bukan luas terbakar, tidak ada variabel lingkungan, dan hanya MODIS yang dimodelkan.

## Mengunggah ke GitHub (GitHub Desktop)
1. Buat akun di github.com, pasang **GitHub Desktop**, lalu login.
2. Ekstrak ZIP, lalu **File → Add local repository** dan pilih folder `EVDM-Hotspot-Python`. Jika muncul pesan "bukan repository", klik **create a repository**.
3. Isi *Summary*, misalnya `Commit pertama`, lalu klik **Commit to main**.
4. Klik **Publish repository**, beri nama `EVDM-Hotspot`, pilih Public, lalu **Publish**.
5. Setelah ada perubahan: commit lagi, lalu **Push origin**.

Tanpa GitHub Desktop: di github.com klik **New repository**, lalu **uploading an existing file** dan seret isi folder (bukan foldernya). Berkas terbesar proyek ini sekitar 11 MB sehingga masih di bawah batas 25 MB unggah lewat browser.

## Menayangkan dashboard online (opsional, gratis)
1. Buka https://share.streamlit.io dan login dengan akun GitHub.
2. Klik **Create app**, pilih repository tadi, branch `main`, berkas utama `app.py`, lalu **Deploy**.
3. Tautan aplikasi dapat dicantumkan pada laporan.
