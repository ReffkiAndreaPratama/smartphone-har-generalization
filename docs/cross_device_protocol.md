# Protokol Eksperimen C — Generalisasi Lintas Perangkat

## Status pada repository ini

**Eksperimen C tidak dijalankan pada dataset UCI HAR.** Alasannya adalah faktual:
UCI HAR dikumpulkan dengan satu perangkat saja (Samsung Galaxy S II, dipasang di
pinggang). Tanpa variasi perangkat, tidak ada eksperimen lintas perangkat yang
dapat dievaluasi, dan tidak ada klaim generalisasi lintas perangkat yang dapat
dibuat dari dataset ini.

Berkas `reports/results/cross_device.json` selalu berisi status eksplisit
`not_performed` beserta alasannya - bukan angka hasil yang dipalsukan.

Modul `src/cross_device.py` sudah berisi jalur eksekusi penuh
(*leave-one-device-out*) sehingga data mandiri multi-perangkat dapat langsung
dipakai tanpa mengubah kode pipeline.

## Kebutuhan data

1. Minimal dua perangkat smartphone berbeda dengan sensor Accelerometer dan
   Gyroscope (misalnya Samsung Galaxy S II, Xiaomi Redmi, OPPO Reno).
2. Protokol aktivitas identik: 6 aktivitas (`WALKING`, `WALKING_UPSTAIRS`,
   `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`), urutan dan durasi
   sama untuk semua subjek dan semua perangkat.
3. Posisi pemasangan (pinggang/kantong/tangan) dan orientasi dicatat per sesi.
4. Setiap perangkat Idealnya digunakan oleh minimal 5 subjek, sehingga
   generalize lintas perangkat tidak identik dengan generalize lintas subjek.
5. Metadata perangkat: model, versi OS, vendor sensor, sampling rate, dan
   resolusi.
6. Sinyal mentah harus dapat disimpan sebagai window 128 sampel x 9 kanal
   (50 Hz, overlap 50%) agar fitur diekstrak dengan definisi yang sama dengan
   `src/feature_extraction.py`.

## Format manifest

Letakkan file berikut di `data/cross_device/`:

```text
data/cross_device/
├── manifest.csv
├── windows/
│   ├── s01_phoneA_walking_000.npy
│   └── ...
```

`manifest.csv` (kolom wajib):

| Kolom | Tipe | Keterangan |
| --- | --- | --- |
| `file_path` | teks | Path relatif ke folder manifest (atau absolut) |
| `subject_id` | teks | ID subjek, mis. `S01` |
| `device_id` | teks | ID perangkat, mis. `phoneA` |
| `activity` | teks | Nama aktivitas persis seperti daftar di atas (huruf besar) |
| `sampling_rate_hz` | angka (opsional) | Default 50 |

Contoh:

```csv
file_path,subject_id,device_id,activity,sampling_rate_hz
windows/s01_phoneA_walking_000.npy,S01,phoneA,WALKING,50
windows/s02_phoneB_sitting_000.npy,S02,phoneB,SITTING,50
```

Isi file `.npy` adalah array NumPy berbentuk `(128, 9)` dengan urutan kanal:

```text
body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z,
total_acc_x, total_acc_y, total_acc_z
```

Format `.csv` dengan header-less dan 9 kolom juga didukung.

## Menjalankan eksperimen

```bash
python scripts/run_experiments.py --stages cross-device
```

Hasil ditulis ke `reports/results/cross_device.json` dengan protokol
`leave_one_device_out`: untuk setiap perangkat, model dilatih pada semua
perangkat lain dan diuji pada perangkat yang ditahan.

Perhatikan bahwa fitur yang dipakai pada data mandiri **bukan** 561 fitur UCI,
melainkan fitur eksplisit yang dihitung oleh
`src/feature_extraction.extract_window_features` (mean, std, min, max, median,
rms, iqr, energy, skewness, kurtosis, zero-crossings, diff std, dan tiga fitur
spektral per kanal). Karena itu angka pada data mandiri tidak boleh dibandingkan
langsung dengan angka model yang dilatih pada 561 fitur UCI.

## Catatan pelaporan

* Hasil data mandiri harus diberi label "data mandiri" dan tidak boleh
  dicampur dengan tabel hasil UCI HAR.
* Pengambilan data sendiri memerlukan persetujuan etis/wawancara sesuai
  ketentuan universitas masing-masing.
