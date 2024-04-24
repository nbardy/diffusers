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


import numpy as np

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


def print_clip_dict(clip_output):
    """
    Prints the keys and shapes of the elements in the CLIP output object or tuple.

    Args:
    clip_output: A CLIPTextModelOutput object or a tuple of torch.FloatTensor
                 containing various elements depending on the configuration and inputs.
    """
    if isinstance(clip_output, tuple):
        # Handle the case when clip_output is a tuple
        print("CLIP output is a tuple. Elements and their shapes:")
        for i, tensor in enumerate(clip_output):
            print(f"Element {i}: shape {tensor.shape}")
    else:
        # Handle the case when clip_output is a CLIPTextModelOutput object
        print("CLIP output is a CLIPTextModelOutput object. Keys and their shapes:")
        if hasattr(clip_output, "text_embeds") and clip_output.text_embeds is not None:
            print(f"text_embeds: {clip_output.text_embeds.shape}")
        if (
            hasattr(clip_output, "last_hidden_state")
            and clip_output.last_hidden_state is not None
        ):
            print(f"last_hidden_state: {clip_output.last_hidden_state.shape}")
        if (
            hasattr(clip_output, "hidden_states")
            and clip_output.hidden_states is not None
        ):
            print("hidden_states:")
            for i, hidden_state in enumerate(clip_output.hidden_states):
                print(f"  Layer {i}: {hidden_state.shape}")
        if hasattr(clip_output, "attentions") and clip_output.attentions is not None:
            print("attentions:")
            for i, attention in enumerate(clip_output.attentions):
                print(f"  Layer {i}: {attention.shape}")


def encode_prompt(
    text_encoder, text_encoder_with_projection, tokenizers, text_input_ids_list
):
    prompt_embeds_list = []

    text_encoders = [text_encoder, text_encoder_with_projection]

    for i, text_encoder in enumerate(text_encoders):
        tokenizer = tokenizers[i]

        text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
            return_dict=False,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

    print("encode prompt base")
    print("")
    print("prompt_embeds", prompt_embeds.shape)
    print("pooled_prompt_embeds", pooled_prompt_embeds.shape)
    return prompt_embeds, pooled_prompt_embeds


def encode_prompt_return_dict(
    text_encoder, text_encoder_with_projection, tokenizers, text_input_ids_list
):
    # Initialize lists to hold embeddings and pooled outputs
    prompt_embeds_list = []
    pooled_prompt_embeds_list = []

    # Process with the standard text encoder
    tokenizer = tokenizers[0]
    text_input_ids = text_input_ids_list[0]
    outputs = text_encoder(
        text_input_ids.to(text_encoder.device),
        output_hidden_states=True,
        return_dict=True,
    )
    # Extract the last hidden state and pooled output
    last_hidden_state = outputs.last_hidden_state  # BxSeqxH
    pooled_prompt_embeds = outputs.pooler_output  # BxH
    bs_embed, seq_len, _ = last_hidden_state.shape
    last_hidden_state = last_hidden_state.view(
        bs_embed, seq_len, -1
    )  # Reshape for consistency
    prompt_embeds_list.append(last_hidden_state)
    pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    # Process with the projected text encoder
    tokenizer_proj = tokenizers[1]
    text_input_ids_proj = text_input_ids_list[1]
    outputs_proj = text_encoder_with_projection(
        text_input_ids_proj.to(text_encoder_with_projection.device),
        output_hidden_states=True,
        return_dict=True,
    )
    # Extract the last hidden state and pooled output
    last_hidden_state_proj = outputs_proj.last_hidden_state  # BxSeqxH
    pooled_prompt_embeds_proj = outputs_proj.text_embeds  # BxH
    bs_embed_proj, seq_len_proj, _ = last_hidden_state_proj.shape
    last_hidden_state_proj = last_hidden_state_proj.view(
        bs_embed_proj, seq_len_proj, -1
    )  # Reshape for consistency
    prompt_embeds_list.append(last_hidden_state_proj)
    pooled_prompt_embeds_list.append(pooled_prompt_embeds_proj)

    # Concatenate embeddings and pooled outputs from all text encoders
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)  # BxSeqx(sum of Hs)
    pooled_prompt_embeds = torch.concat(
        pooled_prompt_embeds_list, dim=-1
    )  # Bx(sum of Hs)

    print("encode prompt return dict")
    print("")
    print("prompt_embeds", prompt_embeds.shape)
    print("pooled_prompt_embeds", pooled_prompt_embeds.shape)

    return prompt_embeds, pooled_prompt_embeds


def encode_image(image_encoder, image_encoder_with_projection, image_processors, image):
    # Initialize lists to hold embeddings and pooled outputs
    image_embeds_list = []
    pooled_embeds_list = []

    # Ensure the image tensor is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    # Normalize from [-1, 1] to [0, 1]
    normalized_image = (image + 1) / 2

    # Process and encode with the first image encoder
    inputs_one = image_processors[0](images=normalized_image, return_tensors="pt")
    inputs_one = inputs_one.to(device)
    outputs = image_encoder(**inputs_one, output_hidden_states=True, return_dict=True)
    image_embeds_list.append(outputs.last_hidden_state)  # BxSeqxH
    pooled_embeds_list.append(outputs.pooler_output)  # BxH

    # Process and encode with the projected image encoder
    inputs_two = image_processors[1](images=normalized_image, return_tensors="pt")
    inputs_two = inputs_two.to(device)
    outputs_projection = image_encoder_with_projection(
        **inputs_two, output_hidden_states=True, return_dict=True
    )
    image_embeds_list.append(outputs_projection.last_hidden_state)  # BxSeqxH
    pooled_embeds_list.append(outputs_projection.image_embeds)  # BxH

    # Debugging shapes

    image_embeds = torch.concat(image_embeds_list, dim=-1)
    pooled_embeds = torch.concat(pooled_embeds_list, dim=-1)

    print("image_embeds", image_embeds.shape)
    print("pooled_embeds", pooled_embeds.shape)

    return image_embeds, pooled_embeds


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


def tokenize_captions(examples, is_train=True):
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
    tokens_one = tokenize_prompt(tokenizer_one, captions)
    tokens_two = tokenize_prompt(tokenizer_two, captions)
    return tokens_one, tokens_two


def preprocess_train(examples):
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
    tokens_one, tokens_two = tokenize_captions(examples)
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
        .select([0])
        .with_transform(preprocess_train, output_all_columns=True)
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

        prompt_embeds, pooled_prompt_embeds = encode_prompt_return_dict(
            text_encoder=text_encoder_one,
            text_encoder_with_projection=text_encoder_two,
            tokenizers=[tokenizer_one, tokenizer_two],
            text_input_ids_list=[
                batch["input_ids_one"],
                batch["input_ids_two"],
            ],
        )

        from prior_projection import PriorTransformer
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

        print( "Shape of pooled_image_embed_projected_to_text:", pooled_image_embed_projected_to_text.shape)
        print( "Shape of image_embed_projected_to_text:", image_embed_projected_to_text.shape)

        print("Shape of prompt_embeds:", prompt_embeds.shape)
        print("Shape of pooled_prompt_embeds:", pooled_prompt_embeds.shape)

        # assert projection makes image equal to text size

        if image_embed_projected_to_text.shape != prompt_embeds.shape:
            print("Mismatch in shapes:")
            print( "image_embed_projected_to_text.shape:", image_embed_projected_to_text.shape)
            print("prompt_embeds.shape:", prompt_embeds.shape)

        if pooled_image_embed_projected_to_text.shape != pooled_prompt_embeds.shape:
            print("Mismatch in pooled shapes:")
            print( "pooled_image_embed_projected_to_text.shape:", pooled_image_embed_projected_to_text.shape)
            print("pooled_prompt_embeds.shape:", pooled_prompt_embeds.shape)


if __name__ == "__main__":
    main()
