from datasets import load_dataset
import openai
import requests
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

# Load the dataset
wikiart_dataset = load_dataset("huggan/wikiart")


def download_image(image_id):
    # Get the image data using the image_id
    image_data = wikiart_dataset["train"][image_id]["image"]

    # Convert the image data to a PIL Image
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image


# Set up the BLIP-2 model
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_basic_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


def generate_expert_labeling_question(caption):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. You create perfect labeling questions for the expert labeler, we want to label art verbosely with long text strings by combining many questions",
            },
            {
                "role": "user",
                "content": f"Get expert labeling questions, given the original image labels for original caption {caption}",
            },
        ],
    )
    return response.choices[0].message.content.strip()


def generate_expert_labels_critique(image, expert_labeling_question):
    inputs = processor(image, text=expert_labeling_question, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=200)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


def convert_expert_conversation_data_to_labels(expert_labels_critique):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a critical assistant. You critique the expert labeler you need to turn all of this data into a single label",
            },
            {
                "role": "user",
                "content": f"Given this expert conversation about an image, create an image caption for a model training run for a text to image model. The expert label should start with a short poem describing the art piece, from 2-16 words, then end with a series of comma separated tags. The tags should maximally and elegantly cover a wide range of 3000 tags, 2000 tags should come from the most common language in the art world, and the other 1000 should be more eclectic. Include the subject of the image",
            },
            {"role": "assistant", "content": expert_labels_critique},
        ],
    )
    return response.choices[0].message.content.strip()


def get_image_ids(num_images):
    # Get num_images random unique image IDs from the huggan/wikiart dataset
    image_ids = random.sample(range(len(wikiart_dataset["train"])), num_images)
    return image_ids
