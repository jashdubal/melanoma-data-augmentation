import os
import logging
import random
import requests
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from tqdm import tqdm
from itertools import islice

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
NUM_IMAGES = 10  # Set to desired number of images
TRAIN_RATIO = 0.9
TEST_RATIO = 0.05
VAL_RATIO = 0.05  # Should sum to 1 with TRAIN_RATIO and TEST_RATIO
BASE_DIR = "ISIC-images-split"
RANDOM_CLASS = "random"
# Add buffer for failed downloads (fetch more samples than needed)
SAMPLE_BUFFER = 10.0  # Try to fetch 2x the number of samples needed

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
    """Download, crop, and save random images with the specified split using streaming dataset."""
    
    # Load dataset in streaming mode
    logger.info("Loading LAION dataset in streaming mode...")
    streaming_dataset = load_dataset('laion/laion400m', split='train', streaming=True)
    
    # Shuffle the dataset
    shuffled_dataset = streaming_dataset.shuffle(seed=42, buffer_size=10000)
    
    # Calculate split counts
    train_count = int(NUM_IMAGES * TRAIN_RATIO)
    test_count = int(NUM_IMAGES * TEST_RATIO)
    val_count = NUM_IMAGES - train_count - test_count
    
    logger.info(f"Target split: {train_count} training, {test_count} testing, {val_count} validation")
    
    # Initialize counters for successful downloads
    train_success = 0
    test_success = 0
    val_success = 0
    total_success = 0
    
    # Fetch more samples than needed to account for failures
    buffer_size = int(NUM_IMAGES * SAMPLE_BUFFER)
    samples_iterator = islice(shuffled_dataset, buffer_size)
    
    # Progress bar will update as we process images
    pbar = tqdm(total=NUM_IMAGES, desc="Successfully downloaded images")
    
    # Process images
    for i, sample in enumerate(samples_iterator):
        # If we've already got enough successful downloads, break
        if total_success >= NUM_IMAGES:
            break
            
        try:
            # Extract the image URL
            image_url = sample['url']
            
            # Download the image
            response = requests.get(image_url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to download image from {image_url}. Status code: {response.status_code}")
                continue
            
            # Open and crop the image
            image = Image.open(BytesIO(response.content))
            image = crop_to_square(image)
            
            # Determine which split this image belongs to based on current counts
            if train_success < train_count:
                save_dir = TRAIN_DIR
                train_success += 1
            elif test_success < test_count:
                save_dir = TEST_DIR
                test_success += 1
            else:
                save_dir = VAL_DIR
                val_success += 1
            
            # Generate a filename based on the ISIC naming convention
            filename = f"RANDOM_{total_success:06d}.jpg"
            save_path = os.path.join(save_dir, filename)
            
            # Save the image
            image.save(save_path)
            
            # Increment total successful downloads and update progress bar
            total_success += 1
            pbar.update(1)
            
        except Exception as e:
            logger.error(f"Error processing image {i}: {str(e)}")
    
    pbar.close()
    
    # Report final counts
    logger.info(f"Successfully downloaded and processed {total_success} images")
    logger.info(f"Final split: {train_success} training, {test_success} testing, {val_success} validation")
    
    # Check if we got enough images
    if total_success < NUM_IMAGES:
        logger.warning(f"Could only download {total_success}/{NUM_IMAGES} images. Consider increasing SAMPLE_BUFFER.")

if __name__ == "__main__":
    create_directories()
    download_and_process_images() 