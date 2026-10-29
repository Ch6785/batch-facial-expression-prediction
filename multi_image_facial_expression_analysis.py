
import os
from collections import Counter
from tkinter import Tk, filedialog

import cv2
from PIL import Image
from transformers import pipeline

MODEL_NAME = "trpakov/vit-face-expression"

print("Loading pretrained facial expression model...")
classifier = pipeline(
    "image-classification",
    model=MODEL_NAME,
    device=-1
)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

EMOTION_MAP = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "neutral": "neutral",
    "surprise": "happy",
    "fear": "sad",
    "disgust": "angry"
}


def predict_expression(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        print(f"No face detected: {os.path.basename(image_path)}")
        return None

    # Select the largest face
    x, y, w, h = max(
        faces, key=lambda face: face[2] * face[3]
    )

    # Prepare face for pretrained model
    face_bgr = image[y:y + h, x:x + w]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)

    predictions = classifier(face_pil, top_k=1)

    raw_emotion = predictions[0]["label"].lower()
    confidence = predictions[0]["score"] * 100
    emotion = EMOTION_MAP.get(raw_emotion, "neutral")

    # Draw face box and predicted emotion
    label = f"{emotion.upper()} ({confidence:.1f}%)"

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        label,
        (x, max(y - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    output_path = (
        os.path.splitext(image_path)[0] + "_prediction.jpg"
    )
    cv2.imwrite(output_path, image)

    return emotion


def main():
    root = Tk()
    root.withdraw()

    image_paths = filedialog.askopenfilenames(
        title="Select multiple facial images",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp")
        ]
    )

    root.destroy()

    if not image_paths:
        print("No images selected.")
        return

    results = Counter()
    total = 0

    for path in image_paths:
        emotion = predict_expression(path)

        if emotion is not None:
            print(f"{os.path.basename(path)}: {emotion.upper()}")
            results[emotion] += 1
            total += 1

    print("\n===== EXPRESSION SUMMARY =====")

    for emotion in ["happy", "sad", "angry", "neutral"]:
        print(f"{emotion.capitalize()} = {results[emotion]}")

    print(f"Images selected: {len(image_paths)}")
    print(f"Faces successfully classified: {total}")
    print("Prediction images are saved beside the original images.")


if __name__ == "__main__":
    main()
