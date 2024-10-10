import cv2
import numpy as np
import imutils
import time

# Load pre-trained model MobileNet SSD for human detection
prototxt = "path/to/MobileNetSSD_deploy.prototxt"
model = "path/to/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Classes untuk objek yang bisa dikenali oleh MobileNetSSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Video stream dari kamera atau CCTV
cap = cv2.VideoCapture(0)  # Ganti 0 dengan path ke video jika menggunakan video file

# Fungsi untuk mengirim notifikasi (sesuaikan dengan API yang digunakan)
def kirim_notifikasi():
    print("Manusia terdeteksi! Mengirim notifikasi...")

# Variabel untuk pelacakan status deteksi manusia
manusia_terdeteksi = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])

            # Cek apakah objek terdeteksi adalah "person" (manusia)
            if CLASSES[idx] == "person":
                manusia_terdeteksi = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Jika manusia terdeteksi dan notifikasi belum dikirim
                if manusia_terdeteksi:
                    kirim_notifikasi()
                    manusia_terdeteksi = False  # Reset status setelah mengirim notifikasi

    cv2.imshow("CCTV", frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
