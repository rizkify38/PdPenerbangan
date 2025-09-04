import os
import pickle
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from flask import Flask, render_template, request

app = Flask(__name__)

# ==========================
# Paths
# ==========================
DATA_PATHS = ["data/Hasil_Penerbangan_Final.csv"]
MODEL_PATHS = ["model/best_random_forest_model.pkl"]
ENCODER_PATHS = ["model/label_encoders.pkl"]

def first_existing(path_list):
    for p in path_list:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Tidak menemukan file di salah satu path: {path_list}")

DATA_FILE = first_existing(DATA_PATHS)
MODEL_FILE = first_existing(MODEL_PATHS)
ENCODER_FILE = first_existing(ENCODER_PATHS)

# ==========================
# Load dataset
# ==========================
df_raw = pd.read_csv(DATA_FILE)

# === Derive Delay jika tidak ada di CSV ===
if "Delay" not in df_raw.columns:
    # Pastikan kolom yang dibutuhkan ada
    NEED = {"Jadwal_Keberangkatan","Jadwal_Kedatangan","Durasi_Penerbangan"}
    if NEED.issubset(df_raw.columns):
        # 1) Coerce numeric untuk durasi rencana
        df_raw["Durasi_Penerbangan"] = pd.to_numeric(df_raw["Durasi_Penerbangan"], errors="coerce")

        # 2) Parse jam robust: coba beberapa pola umum
        def parse_time(s):
            s = str(s)
            for fmt in ("%H:%M", "%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
                try:
                    return pd.to_datetime(s, format=fmt)
                except Exception:
                    continue
            # fallback: biarkan pandas tebak (bisa NaT kalau aneh)
            return pd.to_datetime(s, errors="coerce")

        dep = df_raw["Jadwal_Keberangkatan"].map(parse_time)
        arr = df_raw["Jadwal_Kedatangan"].map(parse_time)

        # 3) Hitung durasi aktual (menit). Jika lewat tengah malam → tambah 24 jam
        dur_akt = (arr - dep).dt.total_seconds() / 60
        cross_midnight = dur_akt < 0
        dur_akt = dur_akt.where(~cross_midnight, dur_akt + 24*60)

        # 4) Delay = durasi aktual - durasi rencana (negatif → 0 = tidak terlambat/lebih cepat)
        delay = dur_akt - df_raw["Durasi_Penerbangan"]
        delay = delay.where(delay > 0, 0)

        # 5) Finalize
        valid_rows = delay.notna().sum()
        total_rows = len(df_raw)
        print(f"[Delay derive] valid_rows={valid_rows}/{total_rows}, "
              f"avg={delay.dropna().mean() if valid_rows else 0:.2f}")

        df_raw["Delay"] = delay.fillna(0)
    else:
        print("[Delay derive] Kolom wajib tidak lengkap, set Delay=0")
        df_raw["Delay"] = 0


# === Lookup Bandara_Tujuan -> Kota/Deskripsi (GLOBAL & AMAN) ===
def build_lookups(df):
    city = {}
    desc = {}
    if {"Bandara_Tujuan", "Kota_tujuan"}.issubset(df.columns):
        city = (
            df.dropna(subset=["Bandara_Tujuan", "Kota_tujuan"])
              .drop_duplicates("Bandara_Tujuan")
              .set_index("Bandara_Tujuan")["Kota_tujuan"]
              .to_dict()
        )
    if {"Bandara_Tujuan", "Deskripsi_tujuan"}.issubset(df.columns):
        desc = (
            df.dropna(subset=["Bandara_Tujuan", "Deskripsi_tujuan"])
              .drop_duplicates("Bandara_Tujuan")
              .set_index("Bandara_Tujuan")["Deskripsi_tujuan"]
              .to_dict()
        )
    return city, desc

city_by_bandara, desc_by_bandara = build_lookups(df_raw)
print("[Lookup] sizes -> city:", len(city_by_bandara), "| desc:", len(desc_by_bandara))

# ==========================
# Load model & encoders
# ==========================
model = joblib.load(MODEL_FILE)
if hasattr(model, "feature_names_in_"):
    print("[Model features] ->", list(model.feature_names_in_))
else:
    print("[Model features] -> (model tidak simpan nama fitur)")

with open(ENCODER_FILE, "rb") as f:
    label_encoders = pickle.load(f)  # dict {kolom: LabelEncoder}

# Fitur yang dipakai model
EXPECTED_FEATURES = [
    "Kota_tujuan", "Deskripsi_tujuan",
    "Suhu_Celcius_tujuan", "Kelembaban_%_tujuan",
    "Tekanan_hPa_tujuan", "Kecepatan_Angin_m/s_tujuan",
]

# ==========================
# Kolom untuk encode/analisis (boleh lebih banyak)
# ==========================
CATEGORICAL_CANDIDATES = [
    "Tanggal_Penerbangan", "Nomor_Penerbangan", "Maskapai", "Hari",
    "Bandara_Asal", "Bandara_Tujuan",
    "Jadwal_Keberangkatan", "Jadwal_Kedatangan",
    "Kode_IATA_Asal", "Kode_IATA_Tujuan",
    "Kota_tujuan", "Deskripsi_tujuan",
    "Cuaca_tujuan",
]
CATEGORICAL_COLS = [c for c in CATEGORICAL_CANDIDATES if c in df_raw.columns]

NUMERIC_CANDIDATES = [
    "Suhu_Celcius_tujuan", "Kelembaban_%_tujuan",
    "Tekanan_hPa_tujuan", "Kecepatan_Angin_m/s_tujuan",
    "Durasi_Penerbangan",
]
NUMERIC_COLS = [c for c in NUMERIC_CANDIDATES if c in df_raw.columns]

# ==========================
# Helpers
# ==========================
def prettify_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.replace("_", " ") for c in out.columns]
    return out

def safe_transform(le, value: str):
    value = str(value)
    if value in le.classes_:
        return le.transform([value])[0]
    le.classes_ = np.append(le.classes_, value)
    return le.transform([value])[0]

def encode_input(form_data: dict) -> pd.DataFrame:
    row = {}
    for col in CATEGORICAL_COLS:
        val = str(form_data.get(col, ""))
        le = label_encoders.get(col, None)
        row[col] = safe_transform(le, val) if le is not None else 0
    for col in NUMERIC_COLS:
        row[col] = float(form_data.get(col, 0) or 0)
    return pd.DataFrame([row])

def decode_df(df_encoded: pd.DataFrame) -> pd.DataFrame:
    out = df_encoded.copy()
    for col in CATEGORICAL_COLS:
        le = label_encoders.get(col, None)
        if le is not None:
            out[col] = le.inverse_transform(out[col].astype(int))
    return out

def minutes_to_hhmm(m):
    total = int(round(float(m)))
    h = total // 60
    mm = total % 60
    return f"{h}:{mm:02d}"

def try_minutes(hhmm: str) -> float:
    try:
        t = datetime.strptime(hhmm, "%H:%M")
        return t.hour * 60 + t.minute
    except Exception:
        return 0.0

def align_features(X: pd.DataFrame, model):
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        missing = [c for c in expected if c not in X.columns]
        extra   = [c for c in X.columns if c not in expected]
        print("[Feature Align] missing:", missing, "| extra:", extra)
        for c in missing:
            X[c] = 0
        X = X.reindex(columns=expected, fill_value=0)
    else:
        X = X.reindex(columns=EXPECTED_FEATURES, fill_value=0)
    return X

def encode_frame_for_model(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    # isi fitur sesuai EXPECTED_FEATURES
    for col in EXPECTED_FEATURES:
        if col in ["Kota_tujuan","Deskripsi_tujuan"]:
            ser = df.get(col, "").astype(str).fillna("")
            le  = label_encoders.get(col)
            if le is not None:
                known = set(le.classes_)
                out = np.empty(len(ser), dtype=np.int64)
                mk = ser.isin(known)
                out[mk.values] = le.transform(ser[mk].values)
                if (~mk).any():
                    new_vals = ser[~mk].unique()
                    le.classes_ = np.append(le.classes_, new_vals)
                    out[~mk.values] = le.transform(ser[~mk].values)
                feats[col] = out
            else:
                feats[col] = 0
        else:
            feats[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)

    if hasattr(model, "feature_names_in_"):
        feats = feats.reindex(columns=list(model.feature_names_in_), fill_value=0)
    else:
        feats = feats.reindex(columns=EXPECTED_FEATURES, fill_value=0)
    return feats

def compute_avg_predicted_delay(df_src: pd.DataFrame) -> float:
    # prediksi durasi
    X = encode_frame_for_model(df_src)
    try:
        dur_pred = model.predict(X)
    except ValueError:
        cols = list(model.feature_names_in_) if hasattr(model,"feature_names_in_") else EXPECTED_FEATURES
        dur_pred = model.predict(X[cols].to_numpy())

    # durasi rencana dari jadwal atau kolom Durasi_Penerbangan
    def _parse(s):
        for fmt in ("%H:%M","%H:%M:%S","%Y-%m-%d %H:%M","%Y-%m-%d %H:%M:%S"):
            try: return pd.to_datetime(s, format=fmt)
            except: pass
        return pd.to_datetime(s, errors="coerce")

    if {"Jadwal_Keberangkatan","Jadwal_Kedatangan"}.issubset(df_src.columns):
        dep = df_src["Jadwal_Keberangkatan"].map(_parse)
        arr = df_src["Jadwal_Kedatangan"].map(_parse)
        plan = (arr - dep).dt.total_seconds()/60
        plan = plan.where(plan >= 0, plan + 24*60)
    else:
        plan = pd.to_numeric(df_src.get("Durasi_Penerbangan", 0), errors="coerce")

    plan = plan.fillna(0).to_numpy()
    delay_pred = np.maximum(0.0, dur_pred - plan)
    return float(np.nanmean(delay_pred))



# ==========================
# Routes
# ==========================
@app.route("/")
def index():
    total_flight = len(df_raw)
    total_airline = df_raw["Maskapai"].nunique() if "Maskapai" in df_raw.columns else 0
    total_origin = df_raw["Bandara_Asal"].nunique() if "Bandara_Asal" in df_raw.columns else 0
    total_dest = df_raw["Bandara_Tujuan"].nunique() if "Bandara_Tujuan" in df_raw.columns else 0
    # 1) coba pakai delay aktual (kalau ada & > 0)
    # pakai Delay aktual bila ada & > 0, kalau tidak -> rata-rata keterlambatan PREDIKSI
    avg_delay_actual = df_raw["Delay"].mean() if "Delay" in df_raw.columns else np.nan
    avg_delay = compute_avg_predicted_delay(df_raw) if (pd.isna(avg_delay_actual) or avg_delay_actual <= 0) else float(avg_delay_actual)


    sample_df = df_raw.head(5)
    sample_data = prettify_columns(sample_df).to_dict(orient="records")

    return render_template(
        "index.html",
        total_flight=total_flight,
        total_airline=total_airline,
        total_origin=total_origin,
        total_dest=total_dest,
        avg_delay=avg_delay,
        sample_data=sample_data
    )

@app.route("/analisis")
def analisis():
    chart_dir = "static/charts"
    os.makedirs(chart_dir, exist_ok=True)

    df_encoded = df_raw.copy()
    for col in CATEGORICAL_COLS:
        le = label_encoders.get(col, None)
        if le is not None:
            df_encoded[col] = df_raw[col].astype(str).map(lambda v: safe_transform(le, v))

    df_decoded = decode_df(df_encoded)

    if "Delay" in df_raw.columns:
        plt.figure(figsize=(8,4))
        sns.histplot(df_raw["Delay"], bins=30, kde=True)
        plt.title("Distribusi Keterlambatan Penerbangan")
        plt.xlabel("Delay (menit)"); plt.ylabel("Jumlah Penerbangan"); plt.tight_layout()
        plt.savefig(f"{chart_dir}/distribusi.png"); plt.close()

    if "Maskapai" in df_decoded.columns:
        plt.figure(figsize=(8,4))
        df_decoded["Maskapai"].value_counts().sort_values(ascending=False).plot(kind="bar")
        plt.title("Jumlah Penerbangan per Maskapai")
        plt.xlabel("Maskapai"); plt.ylabel("Jumlah Penerbangan"); plt.tight_layout()
        plt.savefig(f"{chart_dir}/jumlah_maskapai.png"); plt.close()

    if {"Maskapai","Delay"}.issubset(df_decoded.columns):
        plt.figure(figsize=(8,4))
        df_decoded.groupby("Maskapai")["Delay"].mean().sort_values().plot(kind="bar")
        plt.title("Rata-rata Delay per Maskapai")
        plt.xlabel("Maskapai"); plt.ylabel("Delay Rata-rata (menit)"); plt.tight_layout()
        plt.savefig(f"{chart_dir}/delay_maskapai.png"); plt.close()

    if {"Bandara_Asal","Delay"}.issubset(df_decoded.columns):
        plt.figure(figsize=(8,4))
        df_decoded.groupby("Bandara_Asal")["Delay"].mean().sort_values().plot(kind="bar")
        plt.title("Rata-rata Delay per Bandara Asal")
        plt.xlabel("Bandara Asal"); plt.ylabel("Delay Rata-rata (menit)"); plt.tight_layout()
        plt.savefig(f"{chart_dir}/delay_asal.png"); plt.close()

    if {"Bandara_Tujuan","Delay"}.issubset(df_decoded.columns):
        plt.figure(figsize=(8,4))
        df_decoded.groupby("Bandara_Tujuan")["Delay"].mean().sort_values().plot(kind="bar")
        plt.title("Rata-rata Delay per Bandara Tujuan")
        plt.xlabel("Bandara Tujuan"); plt.ylabel("Delay Rata-rata (menit)"); plt.tight_layout()
        plt.savefig(f"{chart_dir}/delay_tujuan.png"); plt.close()

    if {"Cuaca_tujuan","Delay"}.issubset(df_decoded.columns):
        plt.figure(figsize=(8,4))
        df_decoded.groupby("Cuaca_tujuan")["Delay"].mean().sort_values().plot(kind="bar")
        plt.title("Pengaruh Cuaca terhadap Delay")
        plt.xlabel("Cuaca Tujuan"); plt.ylabel("Delay Rata-rata (menit)"); plt.tight_layout()
        plt.savefig(f"{chart_dir}/delay_cuaca.png"); plt.close()

    return render_template("analisis.html")

# ===== util format
def mm_to_hhmm_local(m):
    m = int(round(float(m)))
    return f"{m//60}:{m%60:02d}"

@app.route("/prediksi", methods=["GET", "POST"])
def prediksi():
    hasil = None
    if request.method == "POST":
        form = request.form

        # Jaga-jaga: fallback lookup jika tidak ada di global (hindari NameError)
        cb = city_by_bandara if isinstance(city_by_bandara, dict) else {}
        db = desc_by_bandara if isinstance(desc_by_bandara, dict) else {}

        # Ambil nilai form
        kota  = (form.get("Kota_tujuan") or "").strip()
        desk  = (form.get("Deskripsi_tujuan") or "").strip()
        bandt = (form.get("Bandara_Tujuan") or "").strip()

        if not kota and bandt in cb:
            kota = cb.get(bandt, "")
        if not desk and bandt in db:
            desk = db.get(bandt, "")

        raw_row = {
            "Kota_tujuan": kota,
            "Deskripsi_tujuan": desk,
            "Suhu_Celcius_tujuan": form.get("Suhu_Celcius_tujuan", "0"),
            "Kelembaban_%_tujuan": form.get("Kelembaban_%_tujuan", "0"),
            "Tekanan_hPa_tujuan": form.get("Tekanan_hPa_tujuan", "0"),
            "Kecepatan_Angin_m/s_tujuan": form.get("Kecepatan_Angin_m/s_tujuan", "0"),
        }

        # Encode sesuai encoder
        row_enc = {}
        for col in EXPECTED_FEATURES:
            if col in ["Kota_tujuan", "Deskripsi_tujuan"]:
                le = label_encoders.get(col)
                val = str(raw_row.get(col, ""))
                row_enc[col] = safe_transform(le, val) if le is not None else 0
            else:
                row_enc[col] = float(raw_row.get(col, 0) or 0)

        input_df = pd.DataFrame([row_enc])
        input_df = align_features(input_df, model)
        print("[Input to predict] ->", list(input_df.columns), input_df.shape)

        try:
            y_pred = float(model.predict(input_df)[0])
        except ValueError:
            cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else EXPECTED_FEATURES
            arr  = input_df[cols].to_numpy()
            y_pred = float(model.predict(arr)[0])

        # ===== interpretasi: model memprediksi DURASI (menit), bukan delay
        durasi_pred = float(y_pred)

        # Durasi rencana dari jadwal (HH:MM) atau fallback Durasi_Penerbangan
        def _to_minutes(hhmm):
            try:
                t = datetime.strptime(hhmm, "%H:%M")
                return t.hour*60 + t.minute
            except:
                return 0

        durasi_rencana = _to_minutes(request.form.get("Jadwal_Kedatangan","")) - _to_minutes(request.form.get("Jadwal_Keberangkatan",""))
        if durasi_rencana <= 0:
            durasi_rencana = float(request.form.get("Durasi_Penerbangan","0") or 0)

        # Keterlambatan prediksi = pred - rencana (negatif -> 0)
        delay_pred = max(0.0, durasi_pred - (durasi_rencana or 0))
        selisih    = durasi_pred - (durasi_rencana or 0)
        status     = "Lebih Cepat" if selisih < 0 else ("Tepat Waktu" if selisih == 0 else "Lebih Lambat")

        def _mm(m): 
            m = int(round(float(m))); return f"{m//60}:{m%60:02d}"

        tgl   = request.form.get("Tanggal_Penerbangan","")
        hari  = request.form.get("Hari","")
        mask  = request.form.get("Maskapai","")
        kota  = (request.form.get("Kota_tujuan") or request.form.get("Bandara_Tujuan") or "").strip()
        cuaca = request.form.get("Cuaca_tujuan","")
        brkt  = request.form.get("Jadwal_Keberangkatan","")
        tiba  = request.form.get("Jadwal_Kedatangan","")

        hasil = f"""
Tanggal Penerbangan : {tgl} ({hari})
Maskapai            : {mask}
Kota Tujuan         : {kota}
Kondisi Cuaca       : {cuaca}
Jadwal Keberangkatan: {brkt}
Jadwal Kedatangan   : {tiba}
Durasi Rencana      : {durasi_rencana:.0f} menit ({_mm(durasi_rencana)})
Durasi Prediksi     : {durasi_pred:.2f} menit ({_mm(durasi_pred)})
Selisih Durasi      : {selisih:.2f} menit → {status}
Perkiraan Keterlambatan (prediksi): {delay_pred:.2f} menit

{mask or 'Maskapai'} diprediksi mengalami keterlambatan sekitar {round(delay_pred)} menit (~{_mm(delay_pred)}).
""".strip()

    return render_template("prediksi.html", hasil=hasil)

# ==========================
# Run app
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)
