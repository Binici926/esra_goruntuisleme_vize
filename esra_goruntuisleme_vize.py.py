import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Sabitler
MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # Yeşil renk: sağ/sol el metni için
DRAW_COLOR = (255, 255, 0)  # Sarı
CENTER_COLOR = (0, 255, 255)  # Ortadaki daire için turkuaz
TEXT_COLOR = (255, 0, 0)  # Mesafe yazısı için mavi

# Belirli landmark'ın (parmak noktası) görüntü üzerindeki koordinatlarını döndürür
def get_pixel_coordinates(landmarks, index, image_height, image_width):
    landmark = landmarks[index]
    return int(landmark.x * image_width), int(landmark.y * image_height)

# Parmaklar arası mesafeye göre parlaklık ve keskinlik ayarlaması yapar
def adjust_brightness_and_sharpness(image, distance):
    brightness_factor = 1 + (distance / 100)
    sharpness_factor = min(5, distance / 20)

    # Parlaklık ayarı
    bright_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

    # Keskinlik ayarı (bulanıklığı azaltarak)
    if sharpness_factor > 0:
        kernel_size = max(1, int(sharpness_factor)) * 2 + 1  # Tek sayı olmalı
        sharp_image = cv2.GaussianBlur(bright_image, (kernel_size, kernel_size), 0)
    else:
        sharp_image = bright_image

    return sharp_image

# Görüntü üzerine el landmark'larını, mesafeyi ve yön bilgisini çizer
def draw_hand_annotations(image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(image)
    height, width, _ = annotated_image.shape

    for idx, landmarks in enumerate(hand_landmarks_list):
        # Baş ve işaret parmağının uç koordinatları
        x1, y1 = get_pixel_coordinates(landmarks, 8, height, width)  # İşaret parmağı
        x2, y2 = get_pixel_coordinates(landmarks, 4, height, width)  # Baş parmak

        # Mesafe hesaplama
        distance = int(np.hypot(x2 - x1, y2 - y1))

        # Görüntüyü parlaklık ve keskinlik açısından güncelle
        annotated_image = adjust_brightness_and_sharpness(annotated_image, distance)

        # Landmark'ları ve bağlantı çizgilerini çiz
        annotated_image = cv2.circle(annotated_image, (x1, y1), 9, DRAW_COLOR, 5)
        annotated_image = cv2.circle(annotated_image, (x2, y2), 9, DRAW_COLOR, 5)
        annotated_image = cv2.line(annotated_image, (x1, y1), (x2, y2), DRAW_COLOR, 5)

        # Ortadaki mesafe gösterimi
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        annotated_image = cv2.circle(annotated_image, (center_x, center_y), 9, CENTER_COLOR, 5)
        annotated_image = cv2.putText(
            annotated_image, str(distance), (center_x, center_y),
            cv2.FONT_HERSHEY_COMPLEX, 2, TEXT_COLOR, 4
        )

        # Landmark çizimi için proto listeye dönüştür
        landmark_proto = landmark_pb2.NormalizedLandmarkList()
        landmark_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks
        ])

        # MediaPipe ile çizim
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            landmark_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Sağ/sol el bilgisini üst köşeye yaz
        handedness = handedness_list[idx]
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        text_x = int(min(x_coords) * width)
        text_y = int(min(y_coords) * height) - MARGIN

        cv2.putText(
            annotated_image, f"{handedness[0].category_name}",
            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA
        )

    return annotated_image

# HandLandmarker modelini yükler
def initialize_hand_detector(model_path='hand_landmarker.task'):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    return vision.HandLandmarker.create_from_options(options)

# Ana döngü: kameradan görüntü al, işle ve göster
def main():
    detector = initialize_hand_detector()
    cam = cv2.VideoCapture(0)

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            break

        # BGR'den RGB'ye dönüştür
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # El algılama işlemi
        detection_result = detector.detect(mp_image)

        # Sonuçları çiz ve görüntüyü göster
        annotated = draw_hand_annotations(mp_image.numpy_view(), detection_result)
        cv2.imshow("Hand Detection", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        # Çıkmak için 'q' tuşuna bas
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
