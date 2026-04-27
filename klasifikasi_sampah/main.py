import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skimage.feature import hog

# =========================
# DATASET PATH
# =========================
dataset_path = "dataset"

data = []
labels = []

# Map kategori untuk tampilan akhir
kategori_map = {
    "botol_plastik": "Anorganik",
    "kresek": "Anorganik",
    "kertas": "Anorganik",
    "kaleng": "Anorganik",
    "daun": "Organik",
    "kulit_buah": "Organik"
}

# =========================
# EKSTRAKSI FITUR (Warna + Bentuk)
# =========================
def extract_features(image):
    # Resize ke ukuran standar
    image = cv2.resize(image, (64, 64))

    # 1. Fitur Warna (HSV Histogram)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist_features = hist.flatten()

    # 2. Fitur Bentuk (HOG - Histogram of Oriented Gradients)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # HOG membantu mengenali struktur objek meski warnanya mirip
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)

    # Gabungkan semua fitur menjadi satu array
    return np.hstack([hist_features, hog_features])

# =========================
# LOAD DATASET
# =========================
print("Sedang memproses dataset, mohon tunggu...")

if not os.path.exists(dataset_path):
    print(f"Error: Folder '{dataset_path}' tidak ditemukan!")
    exit()

for label in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, label)
    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        features = extract_features(image)
        data.append(features)
        labels.append(label)

if len(data) < 10:
    print("Data terlalu sedikit untuk pelatihan!")
    exit()

# Preprocess data
data = np.array(data)
labels = np.array(labels)

# Label Encoding
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Scaling (Sangat penting untuk KNN agar fitur seimbang)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(
    data_scaled, labels_encoded, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
print("Training model dengan Weighted KNN...")
# Menggunakan weights='distance' agar hasil lebih stabil
model = KNeighborsClassifier(n_neighbors=7, weights='distance')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Akurasi Model: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# =========================
# LIVE DETECTION SETUP
# =========================
history = []
HISTORY_SIZE = 15 # Ditingkatkan agar transisi lebih smooth

cap = cv2.VideoCapture(0)
print("Kamera Aktif. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Area deteksi (ROI)
    x1, y1, x2, y2 = int(w*0.2), int(h*0.2), int(w*0.8), int(h*0.8)
    crop = frame[y1:y2, x1:x2]
    crop_blur = cv2.GaussianBlur(crop, (5, 5), 0)

    # Ekstraksi & Scaling fitur real-time
    fitur = extract_features(crop_blur).reshape(1, -1)
    fitur_scaled = scaler.transform(fitur)

    # Prediksi
    # Mencari probabilitas untuk filter noise
    probs = model.predict_proba(fitur_scaled)[0]
    max_prob = np.max(probs)
    pred_idx = np.argmax(probs)
    pred_label = le.inverse_transform([pred_idx])[0]

    # Cek stabilitas gambar (std dev rendah = gambar kosong/polos)
    std_dev = np.std(crop_blur)

    # =========================
    # LOGIKA FILTER (Threshold)
    # =========================
    if std_dev < 15 or max_prob < 0.4:
        current_result = "Tidak Terdeteksi"
    else:
        current_result = pred_label

    # Smoothing dengan Moving Average/Majority Vote
    history.append(current_result)
    if len(history) > HISTORY_SIZE:
        history.pop(0)

    final_result = max(set(history), key=history.count)

    # Visualisasi
    if final_result == "Tidak Terdeteksi":
        txt_label = "Mencari objek..."
        txt_kategori = "-"
        warna = (0, 0, 255) # Merah
    else:
        txt_label = final_result.replace("_", " ").title()
        txt_kategori = kategori_map.get(final_result, "Lainnya")
        warna = (0, 255, 0) # Hijau

    # Gambar UI
    cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)
    cv2.putText(frame, f"Jenis   : {txt_label}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna, 2)
    cv2.putText(frame, f"Kategori: {txt_kategori}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna, 2)
    cv2.putText(frame, f"Score: {max_prob:.2f}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Smart Waste Classifier Pro", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()