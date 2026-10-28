
import cv2
import joblib
import os
from collections import Counter
from tkinter import Tk, filedialog

# Load the trained model
MODEL_PATH = "emotion_model.pkl"
model = joblib.load(MODEL_PATH)

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


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
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, (48, 48))
    features = face.flatten().reshape(1, -1)

    prediction = str(model.predict(features)[0]).lower()

    # Mark the predicted expression
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        image,
        prediction.upper(),
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    output_path = os.path.splitext(image_path)[0] + "_prediction.jpg"
    cv2.imwrite(output_path, image)

    return prediction


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






