from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# ==========================
# Load dataset & model
# ==========================
df = pd.read_csv("data/Hasil_Penerbangan.csv")
model = joblib.load("model/model_prediksi_delay.pkl")

# Label encoder simpan
label_encoders = {}
for col in ["Tanggal_Penerbangan","Nomor_Penerbangan","Maskapai","Hari",
            "Bandara_Asal","Bandara_Tujuan","Cuaca_tujuan",
            "Jadwal_Keberangkatan","Jadwal_Kedatangan"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ==========================
# Route Halaman Utama (Informasi Penerbangan)
# ==========================
@app.route("/")
def index():
    total_flight = len(df)
    total_airline = df["Maskapai"].nunique()
    total_origin = df["Bandara_Asal"].nunique()
    total_dest = df["Bandara_Tujuan"].nunique()
    avg_delay = df["Delay"].mean()

    # Ambil 5 data teratas sebagai contoh informasi penerbangan
    sample_df = pd.read_csv("data/Hasil_Penerbangan.csv").head(5)

    # Hilangkan underscore di nama kolom
    sample_df.columns = [col.replace("_", " ") for col in sample_df.columns]

    sample_data = sample_df.to_dict(orient="records")

    return render_template(
        "index.html",
        total_flight=total_flight,
        total_airline=total_airline,
        total_origin=total_origin,
        total_dest=total_dest,
        avg_delay=avg_delay,
        sample_data=sample_data
    )
# ==========================
# Fungsi untuk balikin angka → nama asli
# ==========================
def decode_dataframe(df_encoded):
    df_decoded = df_encoded.copy()
    for col, le in label_encoders.items():
        df_decoded[col] = le.inverse_transform(df_encoded[col])
    return df_decoded

# ==========================
# Route Analisis
# ==========================
@app.route("/analisis")
def analisis():
    chart_dir = "static/charts"
    os.makedirs(chart_dir, exist_ok=True)

    # Gunakan df yang sudah di-encode
    df_encoded = df.copy()

    # Balikin angka → nama asli
    df_decoded = decode_dataframe(df_encoded)

    # 1. Distribusi Delay
    plt.figure(figsize=(6,4))
    sns.histplot(df_encoded["Delay"], bins=30, kde=True, color="skyblue")
    plt.title("Distribusi Keterlambatan Penerbangan")
    plt.xlabel("Delay (menit)")
    plt.ylabel("Jumlah Penerbangan")
    plt.savefig(f"{chart_dir}/distribusi.png")
    plt.close()

    # 2. Jumlah penerbangan per Maskapai
    plt.figure(figsize=(8,4))
    df_encoded["Maskapai"].value_counts().plot(kind="bar", color="orange")
    plt.title("Jumlah Penerbangan per Maskapai")
    plt.xlabel("Maskapai")
    plt.ylabel("Jumlah Penerbangan")
    plt.savefig(f"{chart_dir}/jumlah_maskapai.png")
    plt.close()

    # 3. Rata-rata delay per Maskapai
    plt.figure(figsize=(8,4))
    df_encoded.groupby("Maskapai")["Delay"].mean().sort_values().plot(kind="bar", color="red")
    plt.title("Rata-rata Delay per Maskapai")
    plt.xlabel("Maskapai")
    plt.ylabel("Delay Rata-rata (menit)")
    plt.savefig(f"{chart_dir}/delay_maskapai.png")
    plt.close()

    # 4. Rata-rata delay per Bandara Asal
    plt.figure(figsize=(8,4))
    df_encoded.groupby("Bandara_Asal")["Delay"].mean().sort_values().plot(kind="bar", color="green")
    plt.title("Rata-rata Delay per Bandara Asal")
    plt.xlabel("Bandara Asal")
    plt.ylabel("Delay Rata-rata (menit)")
    plt.savefig(f"{chart_dir}/delay_asal.png")
    plt.close()

    # 5. Rata-rata delay per Bandara Tujuan
    plt.figure(figsize=(8,4))
    df_encoded.groupby("Bandara_Tujuan")["Delay"].mean().sort_values().plot(kind="bar", color="purple")
    plt.title("Rata-rata Delay per Bandara Tujuan")
    plt.xlabel("Bandara Tujuan")
    plt.ylabel("Delay Rata-rata (menit)")
    plt.savefig(f"{chart_dir}/delay_tujuan.png")
    plt.close()

    # 6. Pengaruh Cuaca terhadap Delay
    plt.figure(figsize=(8,4))
    df_encoded.groupby("Cuaca_tujuan")["Delay"].mean().sort_values().plot(kind="bar", color="brown")
    plt.title("Pengaruh Cuaca terhadap Delay")
    plt.xlabel("Cuaca Tujuan")
    plt.ylabel("Delay Rata-rata (menit)")
    plt.savefig(f"{chart_dir}/delay_cuaca.png")
    plt.close()

    return render_template("analisis.html")

# ==========================
# Route Prediksi
# ==========================
@app.route("/prediksi", methods=["GET","POST"])
def prediksi():
    hasil = None
    if request.method == "POST":
        # Ambil input dari form
        tanggal = request.form["Tanggal_Penerbangan"]
        nomor = request.form["Nomor_Penerbangan"]
        maskapai = request.form["Maskapai"]
        hari = request.form["Hari"]
        asal = request.form["Bandara_Asal"]
        tujuan = request.form["Bandara_Tujuan"]
        cuaca = request.form["Cuaca_tujuan"]
        berangkat = request.form["Jadwal_Keberangkatan"]
        tiba = request.form["Jadwal_Kedatangan"]
        suhu = float(request.form["Suhu_Celcius_tujuan"])
        kelembaban = int(request.form["Kelembaban_%_tujuan"])
        tekanan = int(request.form["Tekanan_hPa_tujuan"])
        angin = float(request.form["Kecepatan_Angin_m/s_tujuan"])

        # Encode input
        input_data = {
            "Tanggal_Penerbangan": label_encoders["Tanggal_Penerbangan"].transform([tanggal])[0] if tanggal in label_encoders["Tanggal_Penerbangan"].classes_ else 0,
            "Nomor_Penerbangan": label_encoders["Nomor_Penerbangan"].transform([nomor])[0] if nomor in label_encoders["Nomor_Penerbangan"].classes_ else 0,
            "Maskapai": label_encoders["Maskapai"].transform([maskapai])[0] if maskapai in label_encoders["Maskapai"].classes_ else 0,
            "Hari": label_encoders["Hari"].transform([hari])[0] if hari in label_encoders["Hari"].classes_ else 0,
            "Bandara_Asal": label_encoders["Bandara_Asal"].transform([asal])[0] if asal in label_encoders["Bandara_Asal"].classes_ else 0,
            "Bandara_Tujuan": label_encoders["Bandara_Tujuan"].transform([tujuan])[0] if tujuan in label_encoders["Bandara_Tujuan"].classes_ else 0,
            "Cuaca_tujuan": label_encoders["Cuaca_tujuan"].transform([cuaca])[0] if cuaca in label_encoders["Cuaca_tujuan"].classes_ else 0,
            "Jadwal_Keberangkatan": label_encoders["Jadwal_Keberangkatan"].transform([berangkat])[0] if berangkat in label_encoders["Jadwal_Keberangkatan"].classes_ else 0,
            "Jadwal_Kedatangan": label_encoders["Jadwal_Kedatangan"].transform([tiba])[0] if tiba in label_encoders["Jadwal_Kedatangan"].classes_ else 0,
            "Suhu_Celcius_tujuan": suhu,
            "Kelembaban_%_tujuan": kelembaban,
            "Tekanan_hPa_tujuan": tekanan,
            "Kecepatan_Angin_m/s_tujuan": angin
        }

        input_df = pd.DataFrame([input_data])
        prediksi = model.predict(input_df)[0]
        hasil = f"Maskapai penerbangan mengalami keterlambatan sekitar {round(prediksi)} menit"

    return render_template("prediksi.html", hasil=hasil)

# ==========================
# Run app
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)