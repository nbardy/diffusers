import os
import faiss
import torch
import numpy as np
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch

# Initialize auxiliary label pool embeddings
aux_image_embeddings = []
aux_text_embeddings = []
instruction_map = {}

# Define a function to load images and perform pre-processing
def load_image(path):
    image = Image.open(path)
    return image


# Load the model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device)
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

def pre_compute(directory, image_files, instructions):
    image_files = image_files[:200]  # Only process the first 200 images

    for instruction in instructions:
        # Get the instruction embedding
        instruction_inputs = processor(text=[instruction], return_tensors="pt", padding=True)
        with torch.no_grad():
            instruction_embedding = model.get_text_features(**instruction_inputs.to(device))
        instruction_embedding /= instruction_embedding.norm(dim=-1, keepdim=True)
        instruction_map[instruction] = instruction_embedding.cpu().numpy()

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)

        # Get the image embedding
        image = load_image(image_path)
        image_inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_embedding = model.get_image_features(**image_inputs.to(device))
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        aux_image_embeddings.append(image_embedding.cpu().numpy())

        # Get the text embedding
        text = image_file
        text_inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_embedding = model.get_text_features(**text_inputs.to(device))
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        aux_text_embeddings.append(text_embedding.cpu().numpy())



# Define instructions
instructions = [
    "Make an image from the given text prompt: <prompt>",
    "Make an image from the image",
    "Make an image from 5 related concepts"
]

# Load and precompute embeddings
directory = "/home/paperspace/datasets/aux_images_1/data_1" 

image_files = os.listdir(directory)
pre_compute(directory, image_files,  instructions)

# Create separate FAISS indices for images and texts
image_index = faiss.IndexFlatL2(model.config.projection_dim)
text_index = faiss.IndexFlatL2(model.config.projection_dim)

# Add embeddings to the indices
image_index.add(np.vstack(aux_image_embeddings))
text_index.add(np.vstack(aux_text_embeddings))


