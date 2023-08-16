import os
import re
from PIL import Image as PILImage
import pandas as pd
import argparse
import shutil
import subprocess
import json

from huggingface_hub import create_repo

org = "facet"

PROGRESS_FILE = "progress.json"

# Directory containing the images
facsed = "/Users/nicholasbardy/git/cluster/"

# Directory containing the images
image_dirs = {
    "super image": "/Users/nicholasbardy/Downloads/Users/nicholasbardy/Downloads",
    "modern": "/Users/nicholasbardy/Desktop/datasets/unsplash_with_labels/images_labeled",
    "super product image": "/Users/nicholasbardy/Downloads/super_images_apr_18",
    "super basic image": "/Users/nicholasbardy/Desktop/datasets/SDXL 1.0 search images/super_aug",
    "super special image": "/Users/nicholasbardy/Desktop/datasets/SDXL 1.0 search images/all",
    "exciting super contest image": "/Users/nicholasbardy/Desktop/sdxl_comp_images/exciting",
    "pro super contest image": "/Users/nicholasbardy/Desktop/sdxl_comp_images/pro",
    "normal render": facsed + "Facet_SD_Dataset/Behance/Style/",
    "high_fashion": facsed + "Facet_SD_Dataset/Zara",
    "architecture": facsed + "Facet_SD_Dataset/Minimalissimo/Architecture/",
    "model": facsed + "Facet_SD_Dataset/Loewe/Person",
    "design": facsed + "Facet_SD_Dataset/Dribbble",
    "engineering": facsed + "Facet_SD_Dataset/TeenageEngineering/Product",
    "minimal": facsed + "Facet_SD_Dataset/Minimalissimo/",
    "photo": facsed + "Facet_SD_Dataset/Unsplash",
    "photo product": facsed + "Facet_SD_Dataset/Unsplash/Product",
    "cinema": {
        "csv": "/Users/nicholasbardy/git/dataset_raw/film_labeled_blip_t5xxl/captions.csv",
        "path": "/Users/nicholasbardy/git/dataset_raw",
        "keys": ["image_path", "caption"],
    },
}

""


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}


progress = load_progress()


def update_progress(newp):
    global progress
    progress = load_progress()
    if progress:
        progress.update(newp)
        save_progress(progress)


# Function to check if the git repository exists
def git_repo_exists(username, dataset_name):
    try:
        # Try to clone the repository
        subprocess.run(f"git clone https://huggingface.co/{username}/{dataset_name}", shell=True, check=True)
        # If cloning is successful, repository exists, so delete the cloned repository
        shutil.rmtree(dataset_name)
        return True
    except subprocess.CalledProcessError:
        # If cloning fails, repository does not exist
        return False


# Should create file if needed other wise append the line,
# add csv header image, text
def add_line_to_metadata(parent_folder, filename, label):
    # Create the metadata file if it doesn't exist
    if not os.path.exists(os.path.join(parent_folder, "metadata.csv")):
        with open(os.path.join(parent_folder, "metadata.csv"), "w") as f:
            f.write("image,text\n")

    # Append the line
    with open(os.path.join(parent_folder, "metadata.csv"), "a") as f:
        f.write(f"{filename},{label}\n")

    return True


## All images should get jpg, jpeg, png, webp and do it nested if needed also ignore caps for file extensions
# return sequable
def all_image_iterable(image_dir):
    for f in os.listdir(image_dir):
        full_path = os.path.join(image_dir, f)

        # Skip files that start with a period or are .DS_Store
        if f.startswith(".") or f.endswith(".DS_Store"):
            continue

        # If the current item is a directory, yield from it recursively
        if os.path.isdir(full_path):
            yield from all_image_iterable(full_path)
        elif re.search(r"\.(jpg|jpeg|png|webp)$", f, re.IGNORECASE):  # Check the file extension
            yield full_path


def copy_if_not_exists(src, dest):
    if os.path.exists(dest) and os.path.getsize(src) == os.path.getsize(dest):
        return False  # File already exists and has the same size
    os.makedirs(os.path.dirname(dest), exist_ok=True)  # Make sure the directories exist
    shutil.copy(src, dest)
    return True


def is_csv(image_dir):
    return isinstance(image_dir, dict) and "path" in image_dir and "csv" in image_dir and "keys" in image_dir


# Split size
SPLIT_SIZE = 1000

# Command-line arguments
parser = argparse.ArgumentParser(description="Process images and create a dataset.")
parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
parser.add_argument("--username", type=str, required=True, help="Username for hugging face")
# use dir name as tag
parser.add_argument("--use_dir_name", action="store_true", help="Use directory name as tag")
args = parser.parse_args()

default_tags = ["image"]


SPLIT_SIZE = 1000

parser = argparse.ArgumentParser(description="Process images and create a dataset.")
parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
parser.add_argument("--username", type=str, required=True, help="Username for hugging face")
parser.add_argument("--use_dir_name", action="store_true", help="Use directory name as tag")
args = parser.parse_args()

default_tags = ["image"]


def rename_csv(folder, old_name, new_name):
    old_path = os.path.join(folder, old_name)
    new_path = os.path.join(folder, new_name)
    if not os.path.exists(old_path):
        print(f"{old_name} not found in {folder}.")
        return False
    os.rename(old_path, new_path)
    print(f"Renamed {old_name} to {new_name}")
    return True


def mapCSV(folder, old_csv_name, new_csv_name, columnsIn, columnsOut):
    old_path = os.path.join(folder, old_csv_name)
    new_path = os.path.join(folder, new_csv_name)
    if not os.path.exists(old_path):
        print(f"{old_csv_name} not found in {folder}.")
        return False

    df = pd.read_csv(old_path)
    new_df = df[columnsIn]
    new_df.columns = columnsOut
    new_df.to_csv(new_path, index=False)
    print(f"Mapped CSV saved to {new_path}")
    return True


def copy_dataset(image_dir):
    path = image_dir["path"]
    keys = image_dir["keys"]
    csv = image_dir["csv"]

    # copy dir
    dest_folder = os.path.join(args.dataset_name, path)
    if os.path.exists(dest_folder):
        print(f"Dataset already exists at {dest_folder}")
        return False

    # copy csv
    shutil.copy(csv, dest_folder)

    # map
    mapCSV(dest_folder, os.path.basename(csv), "metadata.csv", keys, ["image", "text"])


def make_dataset():
    global progress

    for category, image_dir in image_dirs.items():
        if progress and category == progress["category"]:
            start_idx = progress["index"]
        else:
            start_idx = 0

        # detect to copy labels
        if is_csv(image_dir):
            copy_dataset(image_dir)

            continue

            # Skip images until the start index if needed
        all_images = all_image_iterable(image_dir)

        # Resume by skipping the generator
        for _ in range(start_idx):
            next(all_images, None)

        for idx, image_path in enumerate(all_images):
            idx = idx + start_idx
            if idx % 10 == 9:
                save_progress({"category": category, "index": idx + 1})

            full_image_path = os.path.join(image_dir, image_path)

            # Read image using Pillow
            try:
                image_pil = PILImage.open(full_image_path)
            except Exception as e:
                # skip if image is not readable
                print(f"Skipping {full_image_path} due to {e}")
                continue

            # Tags and labeling code
            image_tags = []  # Process tags as per requirements

            image_tags.append(default_tags)
            if args.use_dir_name:
                image_tags.append(category)

            image_tags.extend(default_tags)
            flat_image_tags = [item for sublist in image_tags for item in sublist]
            text = " ".join(flat_image_tags)

            # we need to make directory of a max image size for easier handling
            # so we add uniform slices
            slice_folder = f"slice_{idx//SPLIT_SIZE}"
            slice_path = os.path.join(args.dataset_name, category, slice_folder)

            relative_image_path = os.path.relpath(image_path, image_dir)  # Get the relative path
            dest_path = os.path.join(slice_path, relative_image_path)

            # Copy files and split into slices
            if copy_if_not_exists(full_image_path, dest_path):
                add_line_to_metadata(slice_path, dest_path, text)

            # Close the image
            image_pil.close()


# check that not done
if not progress.get("make_dataset", False):
    print("Already made dataset")
else:
    print("Making dataset")
    make_dataset()

    print("Done creating dataset")
    print("uploading to huggingface")
    # names
    print(f"dataset name: {args.dataset_name}")
    print(f"username: {args.username}")

    update_progress({"make_dataset": True, "start_upload": 0})


# Create the repository
if not progress.get("create_repo", False):
    print("creating repo")
    # in python now
    create_repo(org + "/" + args.dataset_name, private=True)

    update_progress({"make_dataset": True, "start_upload": 0, "create_repo": True})
else:
    print("Already created repo")

# with multline string for each command using line breaks
if not progress.get("upload", False):
    print("uploading")
    # if not git init
    if subprocess.run(f"cd {args.dataset_name} && git status", shell=True).returncode != 0:
        subprocess.run(f"cd {args.dataset_name} && git init && git lfs install", shell=True)

    subprocess.run(
        f"""
        cd {args.dataset_name}
        git remote add origin https://huggingface.co/datasets/{args.username}/{args.dataset_name}
        git pull origin main
        git lfs track *
        git add .
        git commit -m 'Add files'
        """,
        shell=True,
    )

    # git push
    subprocess.run(f"cd {args.dataset_name} && git push origin main", shell=True)

    update_progress({"make_dataset": True, "start_upload": 0, "create_repo": True, "upload": True})
    print("Done uploading")
else:
    print("Already uploaded")

# print HF url with border
print("| == Dataset URL == |")
print(f"https://huggingface.co/datasets/{args.username}/{args.dataset_name}")
print("| ================= |")
