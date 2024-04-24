import torch
from torch import nn
from collections import OrderedDict


def encode_prompt(
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



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# clip style Residual Attention Block
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# A single transformer layer that projecsts the input to a new space
class PriorTransformer(nn.Module):
    def __init__(self, final_size=768):
        super().__init__()
        self.layer = ResidualAttentionBlock(d_model=1024, n_head=8)
        # Should project to the final size


# Three steps
# Projection [BxN] => Bx(N/4)x8
# Self-Attention [Bx(N/4)x8] => Bx(N/4)x8
# Projection [Bx(N/4)x8] => BxK
# where N = input_dim
#       K = output_dim
class PriorTransformer1D(nn.Module):
    def __init__(self, input_shape, output_shape, heads=8):
        super().__init__()
        self.input_projection = nn.Linear(input_shape[0], output_shape[0], bias=False)

    def forward(self, x: torch.Tensor):
        x = self.input_projection(x)
        return x


from einops import rearrange

# Follows the same as 2D 
class PriorTransformer2D(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        # Let's do two proections with a rearrange in the middle first we take the final dim and project to the inner dim
        self.input_projection_1 = nn.Linear(input_shape[1], output_shape[1], bias=False)
        self.input_projection_2 = nn.Linear(input_shape[0], output_shape[0], bias=False)
        # self.attn = ResidualAttentionBlock(d_model=input_shape[1], n_head=8)

    def forward(self, x: torch.Tensor):
        x = self.input_projection_1(x)
        x = rearrange(x, "b n d -> b d n")
        x = self.input_projection_2(x)
        x = rearrange(x, "b d n -> b n d")
        # x = self.attn(x)

        return x


class PriorTransformer(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        if len(input_shape) == 1:
            self.transformer = PriorTransformer1D(input_shape, output_shape)
        elif len(input_shape) == 2:
            self.transformer = PriorTransformer2D(input_shape, output_shape)

    def forward(self, x: torch.Tensor):
        return self.transformer(x)

if __name__ == "__main__":
    # image_embeds torch.Size([1, 257, 2688])
    #  =>
    # prompt_embeds torch.Size([1, 77, 2048])
    model = PriorTransformer(input_shape=(257, 2688), output_shape=(77, 2048))
    x = torch.randn(1, 257, 2688)
    out = model(x)

    assert out.shape == (1, 77, 2048), f"Expected output shape (1, 77, 2048), got {out.shape}"
    print("Passed 1D test")

    # pooled_image_embeds torch.Size([1, 2304])
    #  =>
    # pooled_prompt_embeds torch.Size([1, 2048])
    model = PriorTransformer(input_shape=(2304,), output_shape=(2048,))
    x = torch.randn(1, 2304)
    out = model(x)

    assert out.shape == (1, 2048), f"Expected output shape (1, 2048), got {out.shape}"
    print("Passed 2D test")

