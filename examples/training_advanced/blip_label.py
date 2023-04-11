import torch
import os

import argparse
import glob

from PIL import Image

parser = argparse.ArgumentParser()
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading models")

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)
model.to(device)

print("Done loading models")

def get_caption(image):

    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return generated_text

# Folder containing the images
parser.add_argument('--image_folder', type=str, required=True)
parser.add_argument('--output_folder', type=str, required=True)
parser.add_argument('--prefix', type=str, required=True)
parser.add_argument('--questions', type=str, required=False)
parser.add_argument('--nested', action="store_true", default=False)
parser.add_argument('--folder_name', action="store_true", default=False)


args = parser.parse_args()
image_folder = args.image_folder
output_folder = args.output_folder


def all_images_flat(image_folder):
    return (
        glob.glob(image_folder + "/*.jpg")
        + glob.glob(image_folder + "/*.jpeg")
        + glob.glob(image_folder + "/*.webp")
        + glob.glob(image_folder + "/*.png")
    )


# nested_all_images
# Will crawl folders recursively one depth
def nested_all_images(image_folder):
    # loop through all child dirs
    for child_dir in os.listdir(image_folder):
        # if the child dir is a dir
        if os.path.isdir(os.path.join(image_folder, child_dir)):
            # loop through all images in the child dir
            for image in all_images(os.path.join(image_folder, child_dir)):
                # yield the image
                yield image


def all_images(image_folder, nested=False):
    if nested:
        return nested_all_images(image_folder)
    else:
        return all_images_flat(image_folder)




instruct = "Your instructions are to complete the text with maximum information coverage completing the captions by appending lots of items with commas, complete the following: "
questions = {
    "syn-face": [
        "Command: Write a poem for this piece. Long Poem:",
        "Q: What are stands out about this person? a:",
        "Q: What is the emotion on this face? A:",
        "Q: What is the important characterists of the light in the image?",
        "Q: What is the skin color, and race? Long Anser:",
        "Q: What are they wearing? A:"
    ],
    "film": [
        "Describe the film scene. Film Scene:",
        "Name and describe All the Characters in the scene. Chracter Descriptions:",
        "Question: What is the scene about?, Answer:",
    ],
    "art": [
        "Question: What would be on the museum plaque for this image?",
        "Instructions: Write a long poem about the art piece; Long Poem:",
        "How many colors are in the image and what are the top 5?",
    ]
}


# Glob for jpg or jpeg image
all_i = all_images(image_folder, args.nested)
all_i = [*all_i]
print("N Images: " + str(len(all_i)))
for image_file in all_i:
    print("loading " + image_file)
    image_raw = Image.open(image_file).convert('RGB')

    caption_adv = None
    if args.questions:
        # generate caption
        captions = []
        # For arg in questions
        for q in questions[args.questions]:
           inputs = processor(images=image_raw, text=q, return_tensors="pt").to(device, torch.float16)
           generated_ids = model.generate(**inputs)
           generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
           print(generated_text)
           captions.append(generated_text)

        caption_adv = (";").join(captions)

    inputs = processor(images=image_raw, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    if caption_adv:
        caption = caption + "; " + caption_adv


    if args.folder_name:
        # get the parent folder name
        parent_folder = os.path.basename(os.path.dirname(image_file))

        # append it to the caption with ";"
        caption = parent_folder + "; " + caption

    if args.prefix:
        caption = caption + "; " + args.prefix

    # remove both forward and backward slashes
    caption = caption.replace('/', '').replace('\\', '')

    # Calculate the remaining space for the caption
    remaining_space = 250 - len(output_folder) - len(".jpg")

    # Trim the caption accordingly
    caption_final = caption[:remaining_space]

    new_path = output_folder + "/" + caption_final + ".jpg"

    print("Labeling " + image_file + " as " + caption_final)
    print("Saving: " + new_path)

    try:
        os.symlink(image_file, new_path)
        print("Labeling " + image_file + " as " + caption_final)
        print("Saving: " + new_path)
    except FileExistsError:
        print("Skipping " + image_file + " as " + new_path + " already exists")


