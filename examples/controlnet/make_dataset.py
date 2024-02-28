import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO
import numpy as np
from datasets import load_dataset
from huggingface_hub import Repository, HfApi

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process COCO 2017 dataset')
parser.add_argument('--rows', type=int, default=5, help='Number of rows to process')
args = parser.parse_args()
N = args.rows

# Load the COCO 2017 dataset
dataset = load_dataset("rafaelpadilla/coco2017", split=f"train[:{N}]")

from PIL import ImageDraw, ImageFont

import plotly.graph_objects as go

# Function to render bounding boxes using Plotly
def render_bounding_boxes(image, objects):
    fig = go.Figure()

    # Display the image
    fig.add_trace(go.Image(z=image))

    # Draw bounding boxes and labels on the image
    for idx, obj in enumerate(objects['bbox']):
        label = objects['label'][idx]
        color = f"rgba({np.random.randint(0,255)}, {np.random.randint(0,255)}, {np.random.randint(0,255)}, 1)"

        # Calculate bounding box coordinates
        x0, y0, x1, y1 = obj

        # Draw the bounding box
        fig.add_shape(type="rect",
                      x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color=color, width=4),
                      name=label)

        # Add the label
        fig.add_trace(go.Scatter(x=[x0], y=[y0],
                                 text=[label],
                                 mode="text",
                                 textposition="bottom center",
                                 textfont=dict(family="Arial, sans-serif",
                                               size=24,
                                               color="white"),
                                 showlegend=False))

    # Hide axis ticks and labels
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

    # Remove legend
    fig.update_layout(showlegend=False)

    # Return the figure object
    return fig

updated_dataset = []

# Process and update the dataset
for i, row in enumerate(dataset):
    print(f"Processing image {i+1}/{N}")
    image = row['image']
    print(image)
    objects = row['objects']
    image_with_boxes = render_bounding_boxes(image, objects)
    
    # Convert PIL Image to bytes
    img_byte_arr = BytesIO()
    image_with_boxes.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    update_row = row.copy()
    update_row['image_with_boxes'] = img_byte_arr
    updated_dataset.append(update_row)


    # Update the dataset
    dataset[i]['image_with_boxes'] = img_byte_arr

# Authentication and upload to Hugging Face
repo_name = "nbardy/coco-2017-boxes-as-imgs"
import huggingface_hub

repo = huggingface_hub.Repository(repo_name)
repo.push_to_hub()
