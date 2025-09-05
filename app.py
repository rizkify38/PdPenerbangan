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
    NEED = {"Jadwal_Keberangkatan","Jadwal_Kedatangan","Durasi_Penerbangan"}
    if NEED.issubset(df_raw.columns):
        df_raw["Durasi_Penerbangan"] = pd.to_numeric(df_raw["Durasi_Penerbangan"], errors="coerce")

        def parse_time(s):
            s = str(s)
            for fmt in ("%H:%M", "%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
                try:
                    return pd.to_datetime(s, format=fmt)
                except Exception:
                    continue
            return pd.to_datetime(s, errors="coerce")

        dep = df_raw["Jadwal_Keberangkatan"].map(parse_time)
        arr = df_raw["Jadwal_Kedatangan"].map(parse_time)
        dur_akt = (arr - dep).dt.total_seconds() / 60
        cross_midnight = dur_akt < 0
        dur_akt = dur_akt.where(~cross_midnight, dur_akt + 24*60)

        delay = dur_akt - df_raw["Durasi_Penerbangan"]
        delay = delay.where(delay > 0, 0)
        df_raw["Delay"] = delay.fillna(0)
    else:
        df_raw["Delay"] = 0

# === Lookup Bandara_Tujuan -> Kota/Deskripsi ===
def build_lookups(df):
    city, desc = {}, {}
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

# ==========================
# Load model & encoders
# ==========================
model = joblib.load(MODEL_FILE)
with open(ENCODER_FILE, "rb") as f:
    label_encoders = pickle.load(f)

EXPECTED_FEATURES = [
    "Kota_tujuan", "Deskripsi_tujuan",
    "Suhu_Celcius_tujuan", "Kelembaban_%_tujuan",
    "Tekanan_hPa_tujuan", "Kecepatan_Angin_m/s_tujuan",
]

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

# kondisi ideal → delay 0 (diset strict: hanya "cerah" + rentang variabel cuaca normal)
def is_weather_ideal(deskripsi: str, suhu: float, kelembaban: float, tekanan: float, angin: float) -> bool:
    def _norm(x): return str(x or "").strip().lower()
    d = _norm(deskripsi)
    try:
        suhu = float(suhu); kelembaban = float(kelembaban)
        tekanan = float(tekanan); angin = float(angin)
    except Exception:
        return False

    label_ideal = (d == "cerah")  # kalau mau longgar, tambahkan: or d == "berawan"
    return (
        label_ideal and
        21 <= suhu <= 31 and
        65 <= kelembaban <= 78 and
        1005 <= tekanan <= 1014 and
        1 <= angin <= 6
    )

# interpretasi cuaca (human-friendly) sesuai label datasetmu
def interpret_weather(deskripsi: str) -> str:
    d = str(deskripsi or "").strip().lower()

    if d == "cerah":
        return "Kondisi cuaca cerah (ideal), tidak ada prediksi keterlambatan."
    if d == "berawan":
        return "Cuaca berawan: umumnya aman dan tidak mengganggu jadwal."
    if d == "gerimis":
        return "Cuaca gerimis: potensi penundaan kecil bisa terjadi, namun biasanya minim."
    if d == "hujan ringan":
        return "Hujan ringan; ada peluang keterlambatan kecil pada proses taxi/takeoff."
    if d == "berkabut":
        return "Cuaca berkabut: visibilitas menurun, kewaspadaan ekstra diperlukan."
    if d == "badai petir":
        return "Badai petir: risiko keterlambatan tinggi karena keselamatan prioritas."

    return "Kondisi cuaca tidak dikenali; prediksi tetap menggunakan model."


def align_features(X: pd.DataFrame, model):
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        for c in expected:
            if c not in X.columns:
                X[c] = 0
        X = X.reindex(columns=expected, fill_value=0)
    else:
        X = X.reindex(columns=EXPECTED_FEATURES, fill_value=0)
    return X

def encode_frame_for_model(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
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
    X = encode_frame_for_model(df_src)
    dur_pred = model.predict(X)
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
    avg_delay_actual = df_raw["Delay"].mean() if "Delay" in df_raw.columns else np.nan
    avg_delay = compute_avg_predicted_delay(df_raw) if (pd.isna(avg_delay_actual) or avg_delay_actual <= 0) else float(avg_delay_actual)
    sample_df = df_raw.head(5)
    sample_data = prettify_columns(sample_df).to_dict(orient="records")
    return render_template("index.html",
        total_flight=total_flight,
        total_airline=total_airline,
        total_origin=total_origin,
        total_dest=total_dest,
        avg_delay=avg_delay,
        sample_data=sample_data)

@app.route("/analisis")
def analisis():
    # direktori output chart
    chart_dir = "static/charts"
    os.makedirs(chart_dir, exist_ok=True)

    # --- 1) Distribusi Delay (kalau ada kolomnya)
    if "Delay" in df_raw.columns and df_raw["Delay"].notna().any():
        plt.figure(figsize=(8, 4))
        sns.histplot(df_raw["Delay"], bins=30, kde=True)
        plt.title("Distribusi Keterlambatan Penerbangan")
        plt.xlabel("Delay (menit)")
        plt.ylabel("Jumlah Penerbangan")
        plt.tight_layout()
        plt.savefig(f"{chart_dir}/distribusi.png")
        plt.close()

    # --- 2) Jumlah penerbangan per Maskapai
    if "Maskapai" in df_raw.columns:
        plt.figure(figsize=(8, 4))
        (df_raw["Maskapai"]
            .value_counts()
            .sort_values(ascending=False)
            .plot(kind="bar"))
        plt.title("Jumlah Penerbangan per Maskapai")
        plt.xlabel("Maskapai")
        plt.ylabel("Jumlah Penerbangan")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{chart_dir}/jumlah_maskapai.png")
        plt.close()

    # --- 3) Rata-rata Delay per Maskapai
    if {"Maskapai","Delay"}.issubset(df_raw.columns):
        plt.figure(figsize=(8, 4))
        (df_raw.groupby("Maskapai")["Delay"]
              .mean()
              .sort_values(ascending=False)
              .plot(kind="bar"))
        plt.title("Rata-rata Delay per Maskapai")
        plt.xlabel("Maskapai")
        plt.ylabel("Delay Rata-rata (menit)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{chart_dir}/delay_maskapai.png")
        plt.close()

    # --- 4) Rata-rata Delay per Bandara Asal
    if {"Bandara_Asal","Delay"}.issubset(df_raw.columns):
        plt.figure(figsize=(8, 4))
        (df_raw.groupby("Bandara_Asal")["Delay"]
              .mean()
              .sort_values(ascending=False)
              .plot(kind="bar"))
        plt.title("Rata-rata Delay per Bandara Asal")
        plt.xlabel("Bandara Asal")
        plt.ylabel("Delay Rata-rata (menit)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{chart_dir}/delay_asal.png")
        plt.close()

    # --- 5) Rata-rata Delay per Bandara Tujuan
    if {"Bandara_Tujuan","Delay"}.issubset(df_raw.columns):
        plt.figure(figsize=(8, 4))
        (df_raw.groupby("Bandara_Tujuan")["Delay"]
              .mean()
              .sort_values(ascending=False)
              .plot(kind="bar"))
        plt.title("Rata-rata Delay per Bandara Tujuan")
        plt.xlabel("Bandara Tujuan")
        plt.ylabel("Delay Rata-rata (menit)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{chart_dir}/delay_tujuan.png")
        plt.close()

    # --- 6) Pengaruh Cuaca terhadap Delay (cuaca sesuai datasetmu)
    # label cuaca: badai petir, berawan, berkabut, cerah, gerimis, hujan ringan
    if {"Cuaca_tujuan","Delay"}.issubset(df_raw.columns):
        # normalisasi teks agar konsisten
        cuaca_series = df_raw["Cuaca_tujuan"].astype(str).str.strip().str.lower()
        df_tmp = df_raw.copy()
        df_tmp["Cuaca_norm"] = cuaca_series
        plt.figure(figsize=(8, 4))
        (df_tmp.groupby("Cuaca_norm")["Delay"]
              .mean()
              .reindex(["cerah", "berawan", "gerimis", "hujan ringan", "berkabut", "badai petir"])
              .dropna()
              .plot(kind="bar"))
        plt.title("Pengaruh Cuaca terhadap Delay")
        plt.xlabel("Cuaca Tujuan")
        plt.ylabel("Delay Rata-rata (menit)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{chart_dir}/delay_cuaca.png")
        plt.close()

    # render halaman analisis (pastikan ada templates/analisis.html)
    return render_template("analisis.html")

@app.route("/prediksi", methods=["GET", "POST"])
def prediksi():
    hasil = None
    if request.method == "POST":
        form = request.form
        # Lookup fallback
        cb = city_by_bandara if isinstance(city_by_bandara, dict) else {}
        db = desc_by_bandara if isinstance(desc_by_bandara, dict) else {}
        kota  = (form.get("Kota_tujuan") or "").strip()
        desk  = (form.get("Deskripsi_tujuan") or "").strip()
        bandt = (form.get("Bandara_Tujuan") or "").strip()
        if not kota and bandt in cb: kota = cb.get(bandt, "")
        if not desk and bandt in db: desk = db.get(bandt, "")

        # Input untuk model
        raw_row = {
            "Kota_tujuan": kota,
            "Deskripsi_tujuan": desk,
            "Suhu_Celcius_tujuan": form.get("Suhu_Celcius_tujuan", "0"),
            "Kelembaban_%_tujuan": form.get("Kelembaban_%_tujuan", "0"),
            "Tekanan_hPa_tujuan": form.get("Tekanan_hPa_tujuan", "0"),
            "Kecepatan_Angin_m/s_tujuan": form.get("Kecepatan_Angin_m/s_tujuan", "0"),
        }

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
        y_pred = float(model.predict(input_df)[0])  # model memprediksi durasi (menit)
        durasi_pred = float(y_pred)

        # Versi aman hitung durasi rencana
        def _to_minutes_safe(val):
            if val is None: return np.nan
            v = pd.to_numeric(val, errors="coerce")
            if pd.notna(v): return float(v)
            try:
                t = datetime.strptime(str(val).strip(), "%H:%M")
                return float(t.hour * 60 + t.minute)
            except Exception:
                return np.nan

        dep_s  = request.form.get("Jadwal_Keberangkatan")
        arr_s  = request.form.get("Jadwal_Kedatangan")
        plan_from_time = _to_minutes_safe(arr_s) - _to_minutes_safe(dep_s)
        if not np.isfinite(plan_from_time) or plan_from_time <= 0:
            plan_from_time = pd.to_numeric(request.form.get("Durasi_Penerbangan"), errors="coerce")
        durasi_rencana = float(plan_from_time) if pd.notna(plan_from_time) else 0.0

        # ===== Cuaca & alasan =====
        deskripsi_cuaca = desk or form.get("Cuaca_tujuan","")
        alasan_cuaca = interpret_weather(deskripsi_cuaca)

        # ===== Logika cuaca ideal (strict) =====
        if is_weather_ideal(deskripsi_cuaca, raw_row["Suhu_Celcius_tujuan"], raw_row["Kelembaban_%_tujuan"],
                            raw_row["Tekanan_hPa_tujuan"], raw_row["Kecepatan_Angin_m/s_tujuan"]):
            delay_pred = 0.0
            durasi_pred = float(durasi_rencana or durasi_pred)
        else:
            delay_pred = max(0.0, durasi_pred - (durasi_rencana or 0))

        status = "Tepat Waktu" if delay_pred == 0 else ("Lebih Cepat" if (durasi_pred - (durasi_rencana or 0)) < 0 else "Lebih Lambat")

        # Formatter H:MM
        def _mm(m): 
            m = int(round(float(m))); return f"{m//60}:{m%60:02d}"

        tgl   = form.get("Tanggal_Penerbangan","")
        hari  = form.get("Hari","")
        mask  = form.get("Maskapai","")
        kota_out  = (form.get("Kota_tujuan") or form.get("Bandara_Tujuan") or "").strip()
        brkt  = form.get("Jadwal_Keberangkatan","")
        tiba  = form.get("Jadwal_Kedatangan","")

        # Kalimat ringkas maskapai (2 variasi)
        if delay_pred > 0:
            line_maskapai = f"{mask or 'Maskapai'} diprediksi mengalami keterlambatan sekitar {round(delay_pred)} menit (~{_mm(delay_pred)})."
        else:
            line_maskapai = f"{mask or 'Maskapai'} tidak ada prediksi keterlambatan."

        # === TEMPLATE: jika TIDAK ADA keterlambatan => seperti Gambar 1
        if delay_pred == 0:
            estimasi = durasi_rencana if durasi_rencana > 0 else durasi_pred
            hasil = f"""
Tanggal Penerbangan : {tgl} ({hari})
Maskapai            : {mask}
Kota Tujuan         : {kota_out}
Kondisi Cuaca       : {deskripsi_cuaca or '-'}
estimasi            : {estimasi:.0f} menit ({_mm(estimasi)})
Durasi Prediksi     : {durasi_pred:.2f} menit ({_mm(durasi_pred)})
Perkiraan Delay     : {delay_pred:.2f} menit → {status}

{line_maskapai}

{alasan_cuaca}
""".strip()
        # === TEMPLATE normal (ada keterlambatan / berbeda dari rencana)
        else:
            selisih = durasi_pred - (durasi_rencana or 0)
            hasil = f"""
Tanggal Penerbangan : {tgl} ({hari})
Maskapai            : {mask}
Kota Tujuan         : {kota_out}
Kondisi Cuaca       : {deskripsi_cuaca or '-'}
Jadwal Keberangkatan: {brkt}
Jadwal Kedatangan   : {tiba}
Durasi Rencana      : {durasi_rencana:.0f} menit ({_mm(durasi_rencana)})
Durasi Prediksi     : {durasi_pred:.2f} menit ({_mm(durasi_pred)})
Selisih Durasi      : {selisih:.2f} menit → {status}
Perkiraan Keterlambatan (prediksi): {delay_pred:.2f} menit

{line_maskapai}

{alasan_cuaca}
""".strip()

    return render_template("prediksi.html", hasil=hasil)

# ==========================
# Run app
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)
