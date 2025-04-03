import os
import logging
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Define dataset directory
DATASET_DIR = "ISIC-images-split/train"

# Define augmentations
AUGMENTATIONS = [
    'rotate90CW', 'rotate90CCW', 'rotate180',
    # 'flip_horizontal', 'flip_vertical',
    # 'zoom_in_10', 'zoom_in_20',
    # 'brightness_adjust', 'contrast_adjust',
    # 'saturation_adjust', 'hue_adjust', 'gamma_adjust',
    'random_center_crop'
]

# Augmentation functions
def apply_augmentation(image, aug_type):
    if aug_type == 'rotate90CW':
        return image.rotate(-90, expand=True)
    elif aug_type == 'rotate90CCW':
        return image.rotate(90, expand=True)
    elif aug_type == 'rotate180':
        return image.rotate(180, expand=True)
    elif aug_type == 'flip_horizontal':
        return TF.hflip(image)
    elif aug_type == 'flip_vertical':
        return TF.vflip(image)
    elif aug_type == 'zoom_in_10':
        return zoom_image(image, 0.9)
    elif aug_type == 'zoom_in_20':
        return zoom_image(image, 0.8)
    elif aug_type == 'brightness_adjust':
        return TF.adjust_brightness(image, 1.2)
    elif aug_type == 'contrast_adjust':
        return TF.adjust_contrast(image, 1.2)
    elif aug_type == 'saturation_adjust':
        return TF.adjust_saturation(image, 1.2)
    elif aug_type == 'hue_adjust':
        return TF.adjust_hue(image.convert('RGB'), 0.035)
    elif aug_type == 'gamma_adjust':
        return TF.adjust_gamma(image, 0.85)
    elif aug_type == 'random_center_crop':
        return random_center_crop(image)
    else:
        raise ValueError(f"Invalid augmentation type: {aug_type}")

def zoom_image(image, zoom_factor):
    width, height = image.size
    new_width, new_height = int(width * zoom_factor), int(height * zoom_factor)

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    image = image.crop((left, top, right, bottom))
    return image.resize((width, height), Image.LANCZOS)

def random_center_crop(image, crop_min=0.75, crop_max=0.9):
    width, height = image.size
    crop_ratio = random.uniform(crop_min, crop_max)
    new_width, new_height = int(width * crop_ratio), int(height * crop_ratio)

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    image = image.crop((left, top, right, bottom))
    return image.resize((width, height), Image.LANCZOS)

# Main augmentation function
def augment_dataset():
    for class_dir in os.listdir(DATASET_DIR):
        class_path = os.path.join(DATASET_DIR, class_dir)
        if not os.path.isdir(class_path):
            continue

        images = [img for img in os.listdir(class_path) if img.endswith('.jpg')]

        for image_name in tqdm(images, desc=f"Augmenting {class_dir}"):
            image_path = os.path.join(class_path, image_name)
            image = Image.open(image_path)

            for aug in AUGMENTATIONS:
                augmented_image = apply_augmentation(image, aug)
                new_image_name = f"{image_name[:-4]}_{aug}.jpg"
                augmented_image.save(os.path.join(class_path, new_image_name))
            break
            logging.info(f"Augmented {image_name} with all augmentations.")

if __name__ == '__main__':
    augment_dataset()
