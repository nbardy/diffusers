import os
import random
import shutil
from pathlib import Path
import argparse

import cv2
import numpy as np

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Organize images by aspect ratio.")
parser.add_argument("--src_dir", help="Directory containing the images.")
parser.add_argument("--dst_dir", help="Directory to store symbolic links.")
parser.add_argument("--spike_square", type=float,  help="Directory to store symbolic links.", required=False)

args = parser.parse_args()

import os
import re


def trim_and_sanitize_name(name, max_length, safe_char):
    # Trim the name if it exceeds the maximum length
    sanitized_name = name
    if len(sanitized_name) > max_length:
        name_root, extension = os.path.splitext(sanitized_name)
        name_root = name_root[:max_length - len(extension)]
        sanitized_name = name_root + extension

    # Replace special characters with the safe character
    for char in ";:\"":
        sanitized_name = sanitized_name.replace(char, safe_char)

    return sanitized_name



def create_symlink(file_path, link_path, overwrite=False, max_length=255, safe_char='-'):
    try:
        link_path = os.path.join(os.path.dirname(link_path), trim_and_sanitize_name(os.path.basename(link_path), max_length, safe_char))

        if overwrite and os.path.islink(link_path):
            os.unlink(link_path)
        if not os.path.exists(link_path):
            os.symlink(os.path.abspath(file_path), link_path)
        else:
            print(f"Link path {link_path} already exists. Skipping.")
    except OSError as e:
        print(f"OSError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Set the source and destination directories from the command-line arguments
src_dir = args.src_dir
dst_dir = args.dst_dir


# Available aspect ratios
aspect_ratios = [
    [512, 512], [768, 768], [256, 256], [256, 512], [1024, 768], [768, 1024],
    [1280, 704], [384, 576], [768, 1152], [576, 768], [768, 576], [256, 192], [192, 256]
]


# Create destination directories for each aspect ratio
for ar in aspect_ratios:
    Path(os.path.join(dst_dir, f"{ar[0]}x{ar[1]}")).mkdir(parents=True, exist_ok=True)

# Function to calculate aspect ratio error
def aspect_ratio_error(ar1, ar2):
    return abs(float(ar1[0]) / ar1[1] - float(ar2[0]) / ar2[1])

def create_random_zoom_crop(img, crop_size):
    height, width = img.shape[:2]
    max_zoom_factor = min(float(height) / crop_size, float(width) / crop_size)
    zoom_factor = random.uniform(1, max_zoom_factor)
    new_height, new_width = int(height / zoom_factor), int(width / zoom_factor)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    start_h, start_w = random.randint(0, new_height - crop_size), random.randint(0, new_width - crop_size)
    return resized_img[start_h:start_h + crop_size, start_w:start_w + crop_size]



# Function to create random crop
def create_random_crop(img, crop_size):
    height, width = img.shape[:2]
    start_h, start_w = random.randint(0, height - crop_size), random.randint(0, width - crop_size)
    return img[start_h:start_h + crop_size, start_w:start_w + crop_size]

# Iterate through each image
for filename in os.listdir(src_dir):
    file_path = os.path.join(src_dir, filename)
    img = cv2.imread(file_path)
    if img is None:
        print(f"Unable to read {file_path} as an image. Skipping.")
        continue

    # Ensure the file is a valid image
    if img is None:
        print(f"Unable to read {file_path} as an image.")
        continue

    height, width = img.shape[:2]

    # Find all aspect ratios close enough by error margin
    closest_ar = [ar for ar in aspect_ratios if aspect_ratio_error(ar, [width, height]) <= 0.3]

    if len(closest_ar) == 0:
        continue

    for ar in closest_ar:
        new_filename = filename
        if aspect_ratio_error(ar, [width, height]) < 0.15:
            new_filename = f"{filename.split('.')[0]} good crop.{filename.split('.')[1]}"
        elif aspect_ratio_error(ar, [width, height]) > 0.15:
            new_filename = f"{filename.split('.')[0]} bad crop.{filename.split('.')[1]}"

        # Create a symbolic link to the corresponding subdirectory
        link_path = os.path.join(dst_dir, f"{ar[0]}x{ar[1]}", new_filename)
        create_symlink(os.path.abspath(file_path), link_path)


    # if we want to add extra squares do it at this ratio
    if args.spike_square:
        if random.random() < args.spike_square:
            crop_size = random.choice([256, 512])
            random_zoom_crop = create_random_zoom_crop(img, crop_size)
            crop_filename = f"{filename.split('.')[0]}; random zoom crop; {crop_size}x{crop_size}.webp"
            cv2.imwrite(os.path.join(dst_dir, f"{ar[0]}x{ar[1]}", crop_filename), random_zoom_crop, compression_params)

        # With a 20% chance, create a random crop for 256 or 512 squares
        if random.random() < args.spike_square:
            crop_size = random.choice([256, 512])
            random_crop = create_random_crop(img, crop_size)
            crop_filename = f"{filename.split('.')[0]}; random crop; {crop_size}x{crop_size}.webp"
            cv2.imwrite(os.path.join(dst_dir, f"{ar[0]}x{ar[1]}", crop_filename), random_crop, compression_params)
