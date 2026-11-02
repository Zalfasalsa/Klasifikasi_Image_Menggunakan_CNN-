import os
import json

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image, ImageOps

# ==========================================
# 1. KONFIGURASI
# ==========================================
st.set_page_config(
    page_title="OceanicLife CNN Dashboard",
    page_icon="🌊",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")

MODEL_PATH = os.path.join(OUT_DIR, "final_oceanic_model.keras")
BEST_MODEL_PATH = os.path.join(OUT_DIR, "oceanic_best.keras")  # cadangan

METRICS_PATH = os.path.join(OUT_DIR, "metrics.json")
HISTORY_PATH = os.path.join(OUT_DIR, "history.json")
EDA_PATH = os.path.join(OUT_DIR, "eda_summary.json")
CONFIG_PATH = os.path.join(OUT_DIR, "config.json")
CLASS_NAMES_PATH = os.path.join(OUT_DIR, "class_names.json")
CM_PATH = os.path.join(OUT_DIR, "confusion_matrix.npy")
METRICS_TABLE_PATH = os.path.join(OUT_DIR, "metrics_table.csv")
SPLIT_COUNTS_PATH = os.path.join(OUT_DIR, "split_counts.csv")
CLASS_COUNTS_PATH = os.path.join(OUT_DIR, "class_counts.csv")

IMG_SIZE = (160, 160)  # sama dengan notebook


# ==========================================
# 2. FUNGSI BANTU
# ==========================================
@st.cache_resource(show_spinner="Memuat model...")
def load_model(path):
    return tf.keras.models.load_model(path)


def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {} if default is None else default


def load_csv(path, **kwargs):
    if os.path.exists(path):
        return pd.read_csv(path, **kwargs)
    return None


def show_image_if_exists(filename, caption=None):
    path = os.path.join(OUT_DIR, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"File {filename} tidak ditemukan di folder output.")


def build_analysis(metrics, table):
    """Analisis otomatis dari hasil evaluasi (pengganti analysis.json)."""
    lines = []
    acc = metrics.get("accuracy")
    macro = metrics.get("macro", {})
    weighted = metrics.get("weighted", {})

    if acc is not None:
        level = (
            "sangat baik" if acc >= 0.9
            else "baik" if acc >= 0.8
            else "cukup" if acc >= 0.65
            else "kurang baik"
        )
        lines.append(f"Akurasi pada data uji sebesar **{acc * 100:.2f}%**, "
                     f"artinya model tergolong **{level}** dalam mengenali kelas.")

    if macro.get("f1") is not None and weighted.get("f1") is not None:
        gap = abs(weighted["f1"] - macro["f1"])
        if gap < 0.03:
            lines.append("Selisih F1 macro dan weighted kecil, sehingga kinerja "
                         "relatif merata antar kelas.")
        else:
            lines.append("Selisih F1 macro dan weighted cukup besar, sehingga kelas "
                         "dengan data sedikit kinerjanya tertinggal dari kelas besar.")

    if table is not None and "f1-score" in table.columns:
        per_class = table.drop(index=["macro avg", "weighted avg"], errors="ignore")
        per_class = per_class[per_class["support"] > 0]
        if len(per_class):
            best = per_class["f1-score"].idxmax()
            worst = per_class["f1-score"].idxmin()
            lines.append(f"Kelas paling mudah dikenali: **{best}** "
                         f"(F1 = {per_class.loc[best, 'f1-score']:.3f}).")
            lines.append(f"Kelas paling sulit dikenali: **{worst}** "
                         f"(F1 = {per_class.loc[worst, 'f1-score']:.3f}).")
    return lines


# ==========================================
# 3. MEMUAT MODEL & DATA
# ==========================================
model_file = MODEL_PATH if os.path.exists(MODEL_PATH) else (
    BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else None
)

if model_file is None:
    st.error(
        "Model tidak ditemukan!\n\n"
        f"Folder yang dicari: `{OUT_DIR}`\n\n"
        "Pastikan ada file `final_oceanic_model.keras` di folder `output`, "
        "sejajar dengan `app.py`."
    )
    st.write("Isi folder output:",
             os.listdir(OUT_DIR) if os.path.exists(OUT_DIR) else "folder tidak ada")
    st.stop()

model = load_model(model_file)

metrics = load_json(METRICS_PATH)
history = load_json(HISTORY_PATH)
eda = load_json(EDA_PATH)
config = load_json(CONFIG_PATH)

class_names = load_json(CLASS_NAMES_PATH, default=[])
if isinstance(class_names, dict):
    class_names = list(class_names.values())
if not class_names:
    class_names = list(metrics.get("per_class", {}).keys())
if not class_names:
    class_names = [f"Kelas {i}" for i in range(model.output_shape[-1])]

metrics_table = load_csv(METRICS_TABLE_PATH, index_col=0)
split_counts = load_csv(SPLIT_COUNTS_PATH, index_col=0)
class_counts = load_csv(CLASS_COUNTS_PATH, index_col=0)

n_train = int(split_counts["train"].sum()) if split_counts is not None else 0
n_val = int(split_counts["validation"].sum()) if split_counts is not None else 0
n_test = int(split_counts["test"].sum()) if split_counts is not None else 0

# ==========================================
# 4. JUDUL & MENU
# ==========================================
st.title("🌊 OceanicLife CNN Dashboard")
st.markdown(
    "Dashboard klasifikasi organisme laut menggunakan **Convolutional Neural "
    "Network (CNN)** dengan arsitektur **MobileNetV2 Transfer Learning**."
)
st.divider()

menu = st.sidebar.radio(
    "Menu Dashboard",
    [
        "Beranda",
        "Informasi Dataset",
        "Evaluasi Model",
        "Galeri Visualisasi",
        "Prediksi Gambar",
    ],
)

# ==========================================
# 5. BERANDA
# ==========================================
if menu == "Beranda":
    st.header("Dashboard Klasifikasi OceanicLife")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Training", n_train)
    c2.metric("Data Validation", n_val)
    c3.metric("Data Testing", n_test)
    c4.metric("Jumlah Kelas", len(class_names))

    st.subheader("Informasi Model")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Arsitektur Model**")
        st.write(config.get("base_model", "MobileNetV2 (ImageNet)"))
    with c2:
        st.info("**Metode Pelatihan**")
        st.write("Transfer Learning dan Fine-Tuning")
    with c3:
        st.info("**Ukuran Input**")
        size = config.get("img_size", list(IMG_SIZE))
        st.write(f"{size[0]} x {size[1]} piksel")

    if config:
        with st.expander("Detail konfigurasi pelatihan"):
            st.json(config)

    st.subheader("Kelas Organisme Laut")
    cols = st.columns(2)
    for i, name in enumerate(class_names, start=1):
        cols[(i - 1) % 2].write(f"{i}. {name}")

    st.warning(
        "Catatan: label dataset dibuat otomatis memakai CLIP (zero-shot / "
        "pseudo-label), sehingga hasil evaluasi mengukur kesesuaian dengan "
        "label CLIP, bukan label manusia."
    )

# ==========================================
# 6. INFORMASI DATASET
# ==========================================
elif menu == "Informasi Dataset":
    st.header("Informasi Dataset")
    st.write(
        f"Dataset OceanicLife digunakan untuk mengklasifikasikan gambar "
        f"organisme laut ke dalam {len(class_names)} kelas."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Training", n_train)
    c2.metric("Validation", n_val)
    c3.metric("Testing", n_test)

    if class_counts is not None:
        st.subheader("Distribusi Seluruh Dataset per Kelas")
        st.bar_chart(class_counts["jumlah"])

    if split_counts is not None:
        st.subheader("Pembagian Data per Kelas")
        st.dataframe(split_counts, use_container_width=True)

    if eda:
        st.subheader("Ringkasan EDA")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Gambar", eda.get("n_images", 0))
        c2.metric("Gambar Rusak", eda.get("n_corrupt", 0))
        c3.metric("Duplikat", eda.get("n_duplicates", 0))
        c4.metric("Rata-rata Brightness", f"{eda.get('brightness_mean', 0):.1f}")

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Ukuran gambar asli**")
            st.write(
                f"Median: {eda.get('width_median', 0):.0f} x "
                f"{eda.get('height_median', 0):.0f} px"
            )
            st.write(
                f"Lebar: {eda.get('width_min', 0)} - {eda.get('width_max', 0)} px"
            )
            st.write(
                f"Tinggi: {eda.get('height_min', 0)} - {eda.get('height_max', 0)} px"
            )
        with c2:
            flags = eda.get("quality_flags", {})
            if flags:
                st.write("**Flag kualitas gambar**")
                st.dataframe(
                    pd.DataFrame({"Kondisi": list(flags.keys()),
                                  "Jumlah": list(flags.values())}),
                    use_container_width=True,
                    hide_index=True,
                )

    st.subheader("Visualisasi Dataset")
    tab1, tab2, tab3 = st.tabs(["Distribusi Kelas", "Contoh per Kelas", "Statistik Gambar"])
    with tab1:
        show_image_if_exists("1_class_distribution.png")
    with tab2:
        show_image_if_exists("2_sample_per_class.png")
    with tab3:
        show_image_if_exists("3_image_stats.png")

# ==========================================
# 7. EVALUASI MODEL
# ==========================================
elif menu == "Evaluasi Model":
    st.header("Evaluasi Kinerja Model CNN")

    macro = metrics.get("macro", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%")
    c2.metric("Precision", f"{macro.get('precision', 0) * 100:.2f}%")
    c3.metric("Recall", f"{macro.get('recall', 0) * 100:.2f}%")
    c4.metric("F1-Score", f"{macro.get('f1', 0) * 100:.2f}%")
    st.caption("Precision, Recall, dan F1-Score di atas memakai Macro Average.")

    # Classification report
    if metrics_table is not None:
        st.subheader("Classification Report")
        fmt = {c: "{:.3f}" for c in ["precision", "recall", "f1-score"]
               if c in metrics_table.columns}
        st.dataframe(metrics_table.style.format(fmt), use_container_width=True)

    # Grafik training
    st.subheader("Grafik Accuracy dan Loss")
    if "accuracy" in history and "loss" in history:
        split_epoch = history.get("split_epoch")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for ax, key, title in [
            (axes[0], "accuracy", "Grafik Accuracy"),
            (axes[1], "loss", "Grafik Loss"),
        ]:
            ax.plot(history[key], label=f"Training {key.title()}")
            if f"val_{key}" in history:
                ax.plot(history[f"val_{key}"], label=f"Validation {key.title()}")
            if split_epoch:
                ax.axvline(split_epoch - 0.5, color="gray", linestyle="--",
                           alpha=0.7, label="Mulai fine-tuning")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(key.title())
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("history.json tidak ditemukan atau formatnya tidak sesuai.")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    if os.path.exists(CM_PATH):
        cm = np.load(CM_PATH)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
        )
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Kelas Sebenarnya")
        ax.set_title("Confusion Matrix")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("confusion_matrix.npy tidak ditemukan.")

    # Analisis otomatis
    st.subheader("Analisis Hasil Evaluasi")
    for line in build_analysis(metrics, metrics_table):
        st.write("- " + line)

# ==========================================
# 8. GALERI VISUALISASI
# ==========================================
elif menu == "Galeri Visualisasi":
    st.header("Galeri Visualisasi Hasil Notebook")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Contoh Prediksi", "Salah Prediksi", "Grad-CAM", "Review Label"
    ])
    with tab1:
        show_image_if_exists("6_sample_predictions.png", "Contoh prediksi pada data test")
    with tab2:
        show_image_if_exists("7_misclassified.png", "Kesalahan dengan keyakinan tertinggi")
    with tab3:
        show_image_if_exists("8_gradcam.png", "Area merah = bagian gambar paling berpengaruh")
    with tab4:
        show_image_if_exists("0_label_review_lowconf.png", "Gambar dengan confidence CLIP rendah")

# ==========================================
# 9. PREDIKSI GAMBAR
# ==========================================
elif menu == "Prediksi Gambar":
    st.header("Prediksi Organisme Laut")
    st.write("Unggah gambar organisme laut untuk mengetahui hasil klasifikasi model CNN.")

    uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")

        st.subheader("Gambar yang Diunggah")
        st.image(image, caption="Gambar Input", width=350)

        if st.button("Mulai Prediksi", type="primary"):
            with st.spinner("Model sedang melakukan prediksi..."):
                # Resize sama seperti notebook. Normalisasi (-1..1) sudah ada
                # di dalam model (layer Rescaling), jadi input tetap 0-255.
                img = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
                img_array = np.expand_dims(np.array(img).astype("float32"), axis=0)

                probabilities = model.predict(img_array, verbose=0)[0]

            idx = int(np.argmax(probabilities))
            predicted_class = class_names[idx]
            confidence = float(probabilities[idx]) * 100

            st.success("Prediksi berhasil!")
            st.subheader("Hasil Klasifikasi")
            st.markdown(f"### Organisme: {predicted_class}")
            st.metric("Confidence", f"{confidence:.2f}%")
            st.progress(float(probabilities[idx]))

            if confidence < 50:
                st.warning("Confidence rendah. Gambar mungkin tidak termasuk kelas "
                           "yang dikenal model atau kualitasnya kurang jelas.")

            st.subheader("Probabilitas Setiap Kelas")
            prob_df = pd.DataFrame({
                "Kelas": class_names,
                "Probabilitas": probabilities * 100,
            }).sort_values("Probabilitas", ascending=False)

            st.bar_chart(prob_df.set_index("Kelas"))
            st.dataframe(
                prob_df.style.format({"Probabilitas": "{:.2f}%"}),
                use_container_width=True,
                hide_index=True,
            )

# ==========================================
# 10. FOOTER
# ==========================================
st.divider()
st.caption("OceanicLife CNN Dashboard | Klasifikasi Organisme Laut Menggunakan MobileNetV2")