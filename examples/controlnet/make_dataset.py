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

# Function to draw a single bounding box
def draw_single_bounding_box(fig, obj, label, color):
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

# Function to render bounding boxes for a single object
def render_single_bounding_box(image, obj, label):
    fig = go.Figure()

    # Start with a white image background the same size as image
    white_background = np.ones_like(image) * 255 # Assuming image is a numpy array
    fig.add_trace(go.Image(z=white_background))

    color = f"rgba({np.random.randint(0,255)}, {np.random.randint(0,255)}, {np.random.randint(0,255)}, 1)"
    draw_single_bounding_box(fig, obj, label, color)

    # Hide axis ticks and labels
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

    # Remove legend
    fig.update_layout(showlegend=False)

    return fig

# Function to render bounding boxes for multiple objects
def render_multiple_bounding_boxes(image, objects):
    fig = go.Figure()

    # Start with a white image background the same size as image
    white_background = np.ones_like(image) * 255 # Assuming image is a numpy array
    fig.add_trace(go.Image(z=white_background))

    # Draw bounding boxes and labels on the image
    for idx, obj in enumerate(objects['bbox']):
        label = objects['label'][idx]
        color = f"rgba({np.random.randint(0,255)}, {np.random.randint(0,255)}, {np.random.randint(0,255)}, 1)"
        draw_single_bounding_box(fig, obj, label, color)

    # Hide axis ticks and labels
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

    # Remove legend
    fig.update_layout(showlegend=False)

    return fig

updated_dataset = []

# Process and update the dataset
for i, row in enumerate(dataset):
    print(f"Processing image {i+1}/{N}")
    image = row['image']
    objects = row['objects']
    
    # Process each object in the image
    for idx, obj in enumerate(objects['bbox']):
        label = objects['label'][idx]
        fig_single = render_single_bounding_box(image, obj, label)  # Render a single bounding box
        
        # Convert Plotly figure to PIL Image for single bounding box
        fig_bytes_single = fig_single.to_image(format="png")  # Convert the figure to PNG format in bytes
        img_pil_single = Image.open(BytesIO(fig_bytes_single))  # Open the bytes as a PIL Image
        
        # Convert PIL Image to bytes for storage
        img_byte_arr_single = BytesIO()
        img_pil_single.save(img_byte_arr_single, format='PNG')
        img_byte_arr_single = img_byte_arr_single.getvalue()

        # Update the row with the new image with a single bounding box
        row[f'condition_image_{idx}'] = img_byte_arr_single
        row[f'condition_prompt_{idx}'] = f"Draw a bounding box around the {label}"

    # Render multiple bounding boxes
    fig_multi = render_multiple_bounding_boxes(image, objects)
    
    # Convert Plotly figure to PIL Image for multiple bounding boxes
    fig_bytes_multi = fig_multi.to_image(format="png")
    img_pil_multi = Image.open(BytesIO(fig_bytes_multi))
    
    # Convert PIL Image to bytes for storage
    img_byte_arr_multi = BytesIO()
    img_pil_multi.save(img_byte_arr_multi, format='PNG')
    img_byte_arr_multi = img_byte_arr_multi.getvalue()

    # Update the row with the new image with multiple bounding boxes
    row['row_multi'] = img_byte_arr_multi

    updated_dataset.append(row)

# Authentication and upload to Hugging Face
repo_name = "nbardy/coco-2017-boxes-as-imgs"
import huggingface_hub

repo = huggingface_hub.Repository(repo_name)
repo.push_to_hub()
