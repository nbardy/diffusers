import torch
from torch import nn
from collections import OrderedDict


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

