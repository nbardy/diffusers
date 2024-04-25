## This file loads the dataset
#


import os
import random
import torch
from torchvision import transforms
from torchvision.transforms.functional import crop
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
    AutoProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)

from prior_projection import PriorTransformer, encode_image, encode_prompt


import numpy as np


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


caption_column = "text"


def tokenize_captions(examples, tokenizers, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
    tokenizer_one = tokenizers[0]
    tokenizer_two = tokenizers[1]

    tokens_one = tokenize_prompt(tokenizer_one, captions)
    tokens_two = tokenize_prompt(tokenizer_two, captions)
    return tokens_one, tokens_two


def preprocess_train(examples, tokenizers):
    args = {
        "resolution": 1024,
        "center_crop": False,
        "random_flip": True,
    }
    image_column = "image"
    images = [image.convert("RGB") for image in examples[image_column]]
    original_sizes = []
    all_images = []
    crop_top_lefts = []
    for image in images:
        original_sizes.append((image.height, image.width))
        image = train_resize(image)
        if random.random() < 0.5:
            image = train_flip(image)
        if args["center_crop"]:
            y1 = max(0, int(round((image.height - args["resolution"]) / 2.0)))
            x1 = max(0, int(round((image.width - args["resolution"]) / 2.0)))
            image = train_crop(image)
        else:
            y1, x1, h, w = train_crop.get_params(
                image, (args["resolution"], args["resolution"])
            )
            image = crop(image, y1, x1, h, w)
        crop_top_left = (y1, x1)
        crop_top_lefts.append(crop_top_left)
        # print type of image
        image = train_transforms(image)
        all_images.append(image)

    examples["original_sizes"] = original_sizes
    examples["crop_top_lefts"] = crop_top_lefts
    examples["pixel_values"] = all_images
    tokens_one, tokens_two = tokenize_captions(examples, tokenizers)
    examples["input_ids_one"] = tokens_one
    examples["input_ids_two"] = tokens_two
    return examples


def collate_fn(examples):
    # Stack pixel values into a single tensor and convert to float

    pixel_values = torch.stack(
        [example["pixel_values"] for example in examples]
    ).float()

    original_sizes = torch.tensor([example["original_sizes"] for example in examples])
    crop_top_lefts = torch.tensor([example["crop_top_lefts"] for example in examples])

    # Stack input_ids into single tensors
    input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
    input_ids_two = torch.stack(
        [example["input_ids_two"] for example in examples]
    )  # BxSeqLen
    return {
        "pixel_values": pixel_values,
        "input_ids_one": input_ids_one,
        "input_ids_two": input_ids_two,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }


def main():
    # Assuming the existence of a DATASET_NAME_MAPPING for dataset column names
    DATASET_NAME_MAPPING = {
        "Nbardy/photo_geometric": ("image", "text"),
    }

    model_name = "stabilityai/stable-diffusion-xl-base-1.0"

    tokenizer_one = AutoTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer",
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer_2",
        use_fast=False,
    )

    args = {
        "dataset_name": "Nbardy/photo_geometric",
        "resolution": 1024,
        "center_crop": False,
        "random_flip": True,
        "train_batch_size": 16,
        "dataloader_num_workers": 0,
    }

    # Load the dataset
    dataset = load_dataset(
        args["dataset_name"],
        cache_dir="./cache",
    )

    # Tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained("gpt2")
    tokenizer_two = AutoTokenizer.from_pretrained("gpt2")

    # Image transformations
    global train_resize, train_crop, train_flip, train_transforms
    train_resize = transforms.Resize(
        args["resolution"], interpolation=transforms.InterpolationMode.BILINEAR
    )
    train_crop = transforms.RandomCrop(args["resolution"])
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # Preprocess the dataset
    processed_dataset = (
        dataset["train"]
        .select([1, 2])
        .with_transform(
            lambda x: preprocess_train(x, [tokenizer_one, tokenizer_two]),
            output_all_columns=True,
        )
    )

    # Create the DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        processed_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args["train_batch_size"],
        num_workers=args["dataloader_num_workers"],
    )

    m1 = "openai/clip-vit-large-patch14"
    m2 = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

    image_encoder_one = CLIPVisionModel.from_pretrained(m1)
    image_encoder_two = CLIPVisionModelWithProjection.from_pretrained(m2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_encoder_one.eval().to(device)
    image_encoder_two.eval().to(device)

    processor_one = AutoProcessor.from_pretrained(m1)
    processor_two = AutoProcessor.from_pretrained(m2)

    tokenizer_one = AutoTokenizer.from_pretrained(m1, use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(m2, use_fast=False)

    text_encoder_one = CLIPTextModel.from_pretrained(m1)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(m2)
    text_encoder_one.eval().to(device)
    text_encoder_two.eval().to(device)

    # Loop over and print the first 5 items
    for i, batch in enumerate(train_dataloader):
        image_embeds, pooled_image_embeds = encode_image(
            image_encoder=image_encoder_one,
            image_encoder_with_projection=image_encoder_two,
            image_processors=[processor_one, processor_two],
            image=batch["pixel_values"],
        )

        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoder=text_encoder_one,
            text_encoder_with_projection=text_encoder_two,
            tokenizers=[tokenizer_one, tokenizer_two],
            text_input_ids_list=[
                batch["input_ids_one"],
                batch["input_ids_two"],
            ],
        )

        # image_embeds torch.Size([1, 257, 2688])
        #  =>
        # prompt_embeds torch.Size([1, 77, 2048])

        # pooled_image_embeds torch.Size([1, 2304])
        #  =>
        # pooled_prompt_embeds torch.Size([1, 2048])

        image_to_text_transformer_2 = PriorTransformer(
            input_shape=[257, 2688],
            output_shape=[77, 2048],
        )

        pooled_transformer = PriorTransformer(
            input_shape=[2304],
            output_shape=[2048],
        )

        pooled_transformer.eval().to(device)
        image_to_text_transformer_2.eval().to(device)

        pooled_image_embed_projected_to_text = pooled_transformer(pooled_image_embeds)
        image_embed_projected_to_text = image_to_text_transformer_2(image_embeds)

        print(
            "Shape of pooled_image_embed_projected_to_text:",
            pooled_image_embed_projected_to_text.shape,
        )
        print(
            "Shape of image_embed_projected_to_text:",
            image_embed_projected_to_text.shape,
        )

        print("Shape of prompt_embeds:", prompt_embeds.shape)
        print("Shape of pooled_prompt_embeds:", pooled_prompt_embeds.shape)

        # assert projection makes image equal to text size

        if image_embed_projected_to_text.shape != prompt_embeds.shape:
            print("Mismatch in shapes:")
            print(
                "image_embed_projected_to_text.shape:",
                image_embed_projected_to_text.shape,
            )
            print("prompt_embeds.shape:", prompt_embeds.shape)

        if pooled_image_embed_projected_to_text.shape != pooled_prompt_embeds.shape:
            print("Mismatch in pooled shapes:")
            print(
                "pooled_image_embed_projected_to_text.shape:",
                pooled_image_embed_projected_to_text.shape,
            )
            print("pooled_prompt_embeds.shape:", pooled_prompt_embeds.shape)


if __name__ == "__main__":
    main()
