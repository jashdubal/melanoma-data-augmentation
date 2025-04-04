import requests
from PIL import Image
import os
from io import BytesIO
import time

# URL to fetch the image from
url = "https://thispersondoesnotexist.com/"

# Create output directories
output_dir = "ISIC-images-split/clear_skin"
os.makedirs(output_dir, exist_ok=True)

# Define patch size
patch_size = 200

# Define static coordinates for patches (centered)
patch_centers = {
    "left_cheek": (360, 625),
    "right_cheek": (668, 625),
}

# Fetch 1000 images
FETCH_COUNT = 10
for i in range(FETCH_COUNT):
    try:
        # Dynamically fetch the image from the URL
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch image {i+1}, status code: {response.status_code}")
            continue

        # Open the image from the downloaded bytes
        img = Image.open(BytesIO(response.content))
        img = img.convert("RGB")

        # Save the full face image
        full_face_path = os.path.join(output_dir, f"full_face_{i+1:04d}.png")
        img.save(full_face_path)

        # Extract and save patches
        for label, (cx, cy) in patch_centers.items():
            x1 = cx - patch_size // 2
            y1 = cy - patch_size // 2
            patch = img.crop((x1, y1, x1 + patch_size, y1 + patch_size))
            patch_path = os.path.join(output_dir, f"{label}_{i+1:04d}.png")
            patch.save(patch_path)
        
        print(f"Processed image {i+1}/{FETCH_COUNT}")
        
        # Wait 200ms before fetching the next image
        time.sleep(0.2)
        
    except Exception as e:
        print(f"Error processing image {i+1}: {str(e)}")

print("All images saved successfully.")