
from transformers import pipeline
from PIL import Image
from pathlib import Path

MODEL_NAME = "trpakov/vit-face-expression"

print("Loading pretrained model. First run may take a few minutes...")

classifier = pipeline(
    "image-classification",
    model=MODEL_NAME
)

image_folder = Path("images")
extensions = {".jpg", ".jpeg", ".png"}

for image_path in sorted(image_folder.iterdir()):
    if image_path.suffix.lower() not in extensions:
        continue

    # Skip previously generated prediction images
    if image_path.stem.endswith("_prediction"):
        continue

    try:
        image = Image.open(image_path).convert("RGB")
        results = classifier(image, top_k=3)

        print(f"\nImage: {image_path.name}")
        for result in results:
            print(
                f"  {result['label']}: "
                f"{result['score'] * 100:.2f}%"
            )

    except Exception as error:
        print(f"Could not process {image_path.name}: {error}")
