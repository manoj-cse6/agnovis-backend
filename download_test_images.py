import os
from PIL import Image

os.makedirs("test_images", exist_ok=True)

files = [
    "apple_scab.jpg",
    "tomato_early_blight.jpg",
    "corn_common_rust.jpg",
]

for filename in files:
    filepath = os.path.join("test_images", filename)
    # Create a 224x224 RGB image
    img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    img.save(filepath, "JPEG")
    print(f"Created 224x224 image: {filepath}")

print("\nDone! Updated test_images/ with proper dimensions.")