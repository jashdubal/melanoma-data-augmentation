import os
import logging
import random
import requests
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
NUM_IMAGES = 300
TRAIN_RATIO = 0.9
TEST_RATIO = 0.05
VAL_RATIO = 0.05  # Should sum to 1 with TRAIN_RATIO and TEST_RATIO
BASE_DIR = "ISIC-images-split"
RANDOM_CLASS = "random"

# Define paths
TRAIN_DIR = os.path.join(BASE_DIR, "train", RANDOM_CLASS)
TEST_DIR = os.path.join(BASE_DIR, "test", RANDOM_CLASS)
VAL_DIR = os.path.join(BASE_DIR, "validation", RANDOM_CLASS)

def create_directories():
    """Create the necessary directories for train, test, and validation splits."""
    for directory in [TRAIN_DIR, TEST_DIR, VAL_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def crop_to_square(image):
    """Crop an image to 1:1 aspect ratio from the center."""
    width, height = image.size
    
    # Determine the side length for the square
    side_length = min(width, height)
    
    # Calculate crop coordinates
    left = (width - side_length) // 2
    top = (height - side_length) // 2
    right = left + side_length
    bottom = top + side_length
    
    # Crop and return the image
    return image.crop((left, top, right, bottom))

def download_and_process_images():
    """Download, crop, and save random images with the specified split."""
    
    # Load a subset of the LAION dataset
    subset_size = 'train[:0.001%]'  # Adjusted to ensure sufficient samples
    logger.info(f"Loading dataset with subset size {subset_size}...")
    dataset = load_dataset('laion/laion400m', split=subset_size)
    
    # Shuffle the dataset to randomize the order
    shuffled_dataset = dataset.shuffle(seed=42)
    
    # Make sure we have enough samples
    if len(shuffled_dataset) < NUM_IMAGES:
        logger.warning(f"Only {len(shuffled_dataset)} images available, which is less than the requested {NUM_IMAGES}.")
        num_to_download = len(shuffled_dataset)
    else:
        num_to_download = NUM_IMAGES
    
    # Select the required number of samples
    sampled_dataset = shuffled_dataset.select(range(num_to_download))
    
    # Calculate split counts
    train_count = int(num_to_download * TRAIN_RATIO)
    test_count = int(num_to_download * TEST_RATIO)
    val_count = num_to_download - train_count - test_count
    
    logger.info(f"Split: {train_count} training, {test_count} testing, {val_count} validation")
    
    # Process images
    for i, sample in enumerate(tqdm(sampled_dataset, desc="Downloading and processing images")):
        try:
            # Extract the image URL
            image_url = sample['URL']
            
            # Download the image
            response = requests.get(image_url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to download image from {image_url}. Status code: {response.status_code}")
                continue
            
            # Open and crop the image
            image = Image.open(BytesIO(response.content))
            image = crop_to_square(image)
            
            # Determine which split this image belongs to
            if i < train_count:
                save_dir = TRAIN_DIR
            elif i < train_count + test_count:
                save_dir = TEST_DIR
            else:
                save_dir = VAL_DIR
            
            # Generate a filename based on the ISIC naming convention
            filename = f"RANDOM_{i:06d}.jpg"
            save_path = os.path.join(save_dir, filename)
            
            # Save the image
            image.save(save_path)
            
        except Exception as e:
            logger.error(f"Error processing image {i}: {str(e)}")
    
    logger.info(f"Successfully downloaded and processed {num_to_download} images")

if __name__ == "__main__":
    create_directories()
    download_and_process_images() 