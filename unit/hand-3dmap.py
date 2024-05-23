# %%
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Inisialisasi MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Fungsi untuk mendapatkan landmark tangan dalam mode 3D
def get_hand_landmarks(image_path):
    # Buka gambar
    image = cv2.imread(image_path)
    # Konversi gambar ke RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Inisialisasi model tangan
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        # Proses gambar
        results = hands.process(image_rgb)
        
        # Mendapatkan 3D landmarks
        if results.multi_hand_landmarks:
            hand_landmarks_3d = []
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_3d = []
                for landmark in hand_landmarks.landmark:
                    # Get 3D coordinates
                    landmarks_3d.append((landmark.x, landmark.y, -landmark.z)) # negative z to flip the coordinate system
                hand_landmarks_3d.append(landmarks_3d)
        
        return hand_landmarks_3d

# Path gambar masukan
image_path = "../a.jpg"

# Dapatkan landmark tangan dalam mode 3D
hand_landmarks_3d = get_hand_landmarks(image_path)

# Hitung jarak antara titik-titik landmark tangan
def calculate_distances(landmarks_3d):
    distances = []
    for i in range(len(landmarks_3d)):
        for j in range(i+1, len(landmarks_3d)):
            dist = np.sqrt((landmarks_3d[i][0] - landmarks_3d[j][0])**2 +
                           (landmarks_3d[i][1] - landmarks_3d[j][1])**2 +
                           (landmarks_3d[i][2] - landmarks_3d[j][2])**2)
            distances.append(dist)
    return distances

distances = calculate_distances(hand_landmarks_3d[0])  # Jarak untuk satu tangan

# Visualisasi jarak antar titik landmark tangan dalam proyeksi 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot titik landmark tangan
for landmark in hand_landmarks_3d[0]:
    ax.scatter(landmark[0], landmark[1], landmark[2], color='blue', marker='o')

# Tampilkan plot
plt.show()

# Tampilkan jarak antara titik landmark tangan
print("Jarak antar titik landmark tangan:")
for i, dist in enumerate(distances, start=1):
    print(f"Jarak {i}: {dist}")
