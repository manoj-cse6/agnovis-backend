import os
import shutil
import urllib.request
import zipfile

target_dir = "test_dataset"
zip_path = "plantvillage_sample.zip"

# Clear old directory
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.makedirs(target_dir, exist_ok=True)

# Direct raw zip download link from GitHub
url = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"

headers = {"User-Agent": "Mozilla/5.0"}

print("Downloading authentic PlantVillage leaves directly from GitHub...")
req = urllib.request.Request(url, headers=headers)

with (
    urllib.request.urlopen(req) as response,
    open(zip_path, "wb") as out_file,
):
    out_file.write(response.read())

print("Extracting 3 leaf images per disease class...")

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    # Filter for files inside raw/color directory
    color_files = [
        f
        for f in zip_ref.namelist()
        if "raw/color/" in f and f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    # Group files by folder
    folder_map = {}
    for file_path in color_files:
        parts = file_path.split("/")
        # Disease class folder name
        disease_folder = parts[-2]
        if disease_folder not in folder_map:
            folder_map[disease_folder] = []
        folder_map[disease_folder].append(file_path)

    # Extract 3 images per disease
    copied_count = 0
    for disease_folder, files in folder_map.items():
        dest_folder = os.path.join(target_dir, disease_folder)
        os.makedirs(dest_folder, exist_ok=True)

        for img_path in files[:3]:
            file_name = os.path.basename(img_path)
            extracted_data = zip_ref.read(img_path)

            with open(os.path.join(dest_folder, file_name), "wb") as f:
                f.write(extracted_data)
            copied_count += 1

# Clean up temporary zip
if os.path.exists(zip_path):
    os.remove(zip_path)

print(
    f"\nDone! Extracted {copied_count} real PlantVillage leaf images into '{target_dir}/'."
)