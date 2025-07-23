from functools import partial
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from swin.swin_transformer import PatchMerging, SwinTransformerBlock

def _get_window_and_shift_size(
    shift_size: list[int], size_dhw: list[int], window_size: list[int]
) -> tuple[list[int], list[int]]:
    for i in range(3):
        if size_dhw[i] <= window_size[i]:
            window_size[i] = size_dhw[i]
            shift_size[i] = 0

    return window_size, shift_size


torch.fx.wrap("_get_window_and_shift_size")


def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: list[int]
) -> Tensor:
    window_vol = window_size[0] * window_size[1] * window_size[2]
    # In 3d case we flatten the relative_position_bias
    relative_position_bias = relative_position_bias_table[
        relative_position_index[:window_vol, :window_vol].flatten()  # type: ignore[index]
    ]
    relative_position_bias = relative_position_bias.view(window_vol, window_vol, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


torch.fx.wrap("_get_relative_position_bias")


def _compute_pad_size_3d(size_dhw: tuple[int, int, int], patch_size: tuple[int, int, int]) -> tuple[int, int, int]:
    pad_size = [(patch_size[i] - size_dhw[i] % patch_size[i]) % patch_size[i] for i in range(3)]
    return pad_size[0], pad_size[1], pad_size[2]


torch.fx.wrap("_compute_pad_size_3d")


def _compute_attention_mask_3d(
    x: Tensor,
    size_dhw: tuple[int, int, int],
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
) -> Tensor:
    # generate attention mask
    attn_mask = x.new_zeros(*size_dhw)
    num_windows = (size_dhw[0] // window_size[0]) * (size_dhw[1] // window_size[1]) * (size_dhw[2] // window_size[2])
    slices = [
        (
            (0, -window_size[i]),
            (-window_size[i], -shift_size[i]),
            (-shift_size[i], None),
        )
        for i in range(3)
    ]
    count = 0
    for d in slices[0]:
        for h in slices[1]:
            for w in slices[2]:
                attn_mask[d[0] : d[1], h[0] : h[1], w[0] : w[1]] = count
                count += 1

    # Partition window on attn_mask
    attn_mask = attn_mask.view(
        size_dhw[0] // window_size[0],
        window_size[0],
        size_dhw[1] // window_size[1],
        window_size[1],
        size_dhw[2] // window_size[2],
        window_size[2],
    )
    attn_mask = attn_mask.permute(0, 2, 4, 1, 3, 5).reshape(
        num_windows, window_size[0] * window_size[1] * window_size[2]
    )
    attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


torch.fx.wrap("_compute_attention_mask_3d")


def shifted_window_attention_3d(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: list[int],
    num_heads: int,
    shift_size: list[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    training: bool = True,
    qkv_lora_weight: Optional[Tensor] = None,
) -> Tensor:

    b, t, h, w, c = input.shape
    # pad feature maps to multiples of window size
    pad_size = _compute_pad_size_3d((t, h, w), (window_size[0], window_size[1], window_size[2]))
    x = F.pad(input, (0, 0, 0, pad_size[2], 0, pad_size[1], 0, pad_size[0]))
    _, tp, hp, wp, _ = x.shape
    padded_size = (tp, hp, wp)

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))

    # partition windows
    num_windows = (
        (padded_size[0] // window_size[0]) * (padded_size[1] // window_size[1]) * (padded_size[2] // window_size[2])
    )
    x = x.view(
        b,
        padded_size[0] // window_size[0],
        window_size[0],
        padded_size[1] // window_size[1],
        window_size[1],
        padded_size[2] // window_size[2],
        window_size[2],
        c,
    )
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
        b * num_windows, window_size[0] * window_size[1] * window_size[2], c
    )  # B*nW, Wd*Wh*Ww, C

    # multi-head attention
    qkv = F.linear(x, qkv_weight, qkv_bias)
    if qkv_lora_weight is not None:
        qkv = qkv + qkv_lora_weight(x)

    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, c // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * (c // num_heads) ** -0.5
    attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask to handle shifted windows with varying size
        attn_mask = _compute_attention_mask_3d(
            x,
            (padded_size[0], padded_size[1], padded_size[2]),
            (window_size[0], window_size[1], window_size[2]),
            (shift_size[0], shift_size[1], shift_size[2]),
        )
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), c)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    # reverse windows
    x = x.view(
        b,
        padded_size[0] // window_size[0],
        padded_size[1] // window_size[1],
        padded_size[2] // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        c,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(b, tp, hp, wp, c)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))

    # unpad features
    x = x[:, :t, :h, :w, :].contiguous()
    return x


torch.fx.wrap("shifted_window_attention_3d")

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, lora_alpha=1.0):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(out_features, r) * 0.01)
            self.lora_B = nn.Parameter(torch.randn(r, in_features) * 0.01)
            self.scaling = lora_alpha / r
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        if self.r > 0:
            return F.linear(x, self.lora_A @ self.lora_B) * self.scaling
        else:
            return torch.zeros_like(x)


class ShiftedWindowAttention3d(nn.Module):
    """
    See :func:`shifted_window_attention_3d`.
    """

    def __init__(
        self,
        dim: int,
        window_size: list[int],
        shift_size: list[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(window_size) != 3 or len(shift_size) != 3:
            raise ValueError("window_size and shift_size must be of length 2")

        self.window_size = window_size  # Wd, Wh, Ww
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self) -> None:
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                self.num_heads,
            )
        )  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self) -> None:
        # get pair-wise relative position index for each token inside the window
        coords_dhw = [torch.arange(self.window_size[i]) for i in range(3)]
        coords = torch.stack(
            torch.meshgrid(coords_dhw[0], coords_dhw[1], coords_dhw[2], indexing="ij")
        )  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        # We don't flatten the relative_position_index here in 3d case.
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self, window_size: list[int]) -> torch.Tensor:
        return _get_relative_position_bias(self.relative_position_bias_table, self.relative_position_index, window_size)  # type: ignore

    def lorify(self, r: int = 8, alpha: float = 1.0):
        self.qkv_lora = LoRALinear(self.dim, self.dim * 3, r=r, lora_alpha=alpha)
        print("Using LoRA in ShiftedWindowAttention with r =", r, "and alpha =", alpha)

    def forward(self, x: Tensor) -> Tensor:
        _, t, h, w, _ = x.shape
        size_dhw = [t, h, w]
        window_size, shift_size = self.window_size.copy(), self.shift_size.copy()
        # Handle case where window_size is larger than the input tensor
        window_size, shift_size = _get_window_and_shift_size(shift_size, size_dhw, window_size)

        relative_position_bias = self.get_relative_position_bias(window_size)

        return shifted_window_attention_3d(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            window_size,
            self.num_heads,
            shift_size=shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
            qkv_lora_weight=self.qkv_lora if hasattr(self, 'qkv_lora') else None,
        )


class PatchEmbed3d(nn.Module):
    def __init__(
        self,
        patch_size: list[int],
        in_channels: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.tuple_patch_size = (patch_size[0], patch_size[1], patch_size[2])

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.tuple_patch_size,
            stride=self.tuple_patch_size,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        # padding
        _, _, t, h, w = x.size()
        pad_size = _compute_pad_size_3d((t, h, w), self.tuple_patch_size)
        x = F.pad(x, (0, pad_size[2], 0, pad_size[1], 0, pad_size[0]))
        x = self.proj(x)  # B C T Wh Ww
        x = x.permute(0, 2, 3, 4, 1)  # B T Wh Ww C
        if self.norm is not None:
            x = self.norm(x)
        return x

class STAdapter(nn.Module):
    def __init__(self, in_channels, adapter_channels=64, kernel_size=(3, 3, 3)):
        super().__init__()
        self.in_channels = in_channels
        self.adapter_channels = adapter_channels

        self.norm1 = nn.LayerNorm(in_channels)
        self.down_proj = nn.Linear(in_channels, adapter_channels)

        self.dw_conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(k // 2 for k in kernel_size),
            groups=adapter_channels
        )

        self.norm2 = nn.LayerNorm(adapter_channels)
        self.up_proj = nn.Linear(adapter_channels, in_channels)

        nn.init.constant_(self.dw_conv.weight, 0.)
        nn.init.constant_(self.dw_conv.bias, 0.)
        nn.init.constant_(self.down_proj.bias, 0.)
        nn.init.constant_(self.up_proj.bias, 0.)

        print("USING STAdapter --------------------")

    def forward(self, x):
        BT, H, W, C = x.shape
        assert C == self.in_channels
        T = self.T
        B = BT // T

        x_id = x

        x = x.view(B, T, H, W, C)
        x = self.norm1(x)
        x = self.down_proj(x)

        x = x.permute(0, 4, 1, 2, 3).contiguous()

        x = self.dw_conv(x)

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm2(x)
        x = self.up_proj(x)

        x = x.view(BT, H, W, C)
        return x_id + x


class STAdapterPE(nn.Module):
    def __init__(self, in_channels, adapter_channels=64, kernel_size=(3, 3, 3)):
        super().__init__()
        self.in_channels = in_channels
        self.adapter_channels = adapter_channels

        self.norm1 = nn.LayerNorm(in_channels)
        self.down_proj = nn.Linear(in_channels, adapter_channels)

        self.dw_conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(k // 2 for k in kernel_size),
            groups=adapter_channels
        )

        self.temporal_pos_emb = nn.Parameter(torch.zeros(1000, adapter_channels))  # (T, C)

        self.norm2 = nn.LayerNorm(adapter_channels)
        self.up_proj = nn.Linear(adapter_channels, in_channels)

        nn.init.constant_(self.dw_conv.weight, 0.)
        nn.init.constant_(self.dw_conv.bias, 0.)
        nn.init.constant_(self.down_proj.bias, 0.)
        nn.init.constant_(self.up_proj.bias, 0.)

        print("USING STAdapterPE --------------------")

    def forward(self, x):
        BT, H, W, C = x.shape
        assert C == self.in_channels
        T = self.T
        B = BT // T

        x_id = x

        x = x.view(B, T, H, W, C)
        x = self.norm1(x)
        x = self.down_proj(x)

        # 🔸 Add Temporal Positional Encoding
        pos_emb = self.temporal_pos_emb[:T]  # (T, C)
        pos_emb = pos_emb[None, :, None, None, :]  # (1, T, 1, 1, C)
        x = x + pos_emb  # (B, T, H, W, C)

        x = x.permute(0, 4, 1, 2, 3).contiguous()

        x = self.dw_conv(x)

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm2(x)
        x = self.up_proj(x)

        x = x.view(BT, H, W, C)
        return x_id + x



class TemporalAdapterPE(nn.Module):
    def __init__(self, in_channels, adapter_channels=64, max_frames=64):
        super().__init__()
        self.in_channels = in_channels
        self.adapter_channels = adapter_channels
        self.T = None

        # Norm and projection
        self.norm1 = nn.LayerNorm(in_channels)
        self.down_proj = nn.Linear(in_channels, adapter_channels)

        # 🔸 Temporal Positional Encoding
        self.temporal_pos_emb = nn.Parameter(torch.zeros(max_frames, adapter_channels))  # (T, C)

        # Block 1
        self.block1_conv1x1 = nn.Conv3d(adapter_channels, adapter_channels, kernel_size=1, padding=0)
        self.block1_bn1 = nn.BatchNorm3d(adapter_channels)

        self.block1_conv3x3_1 = nn.Conv3d(adapter_channels, adapter_channels, kernel_size=3, padding=1)
        self.block1_bn2 = nn.BatchNorm3d(adapter_channels)
        self.block1_conv3x3_2 = nn.Conv3d(adapter_channels, adapter_channels, kernel_size=3, padding=1)
        self.block1_bn3 = nn.BatchNorm3d(adapter_channels)

        # Block 2
        self.block2_conv3x3_1 = nn.Conv3d(adapter_channels, adapter_channels, kernel_size=3, padding=1)
        self.block2_bn1 = nn.BatchNorm3d(adapter_channels)
        self.block2_conv3x3_2 = nn.Conv3d(adapter_channels, adapter_channels, kernel_size=3, padding=1)
        self.block2_bn2 = nn.BatchNorm3d(adapter_channels)

        self.norm2 = nn.LayerNorm(adapter_channels)
        self.up_proj = nn.Linear(adapter_channels, in_channels)

        # Bias and weight initialization
        nn.init.constant_(self.down_proj.bias, 0.)
        nn.init.constant_(self.up_proj.bias, 0.)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        print("USING TemporalAdapterPE --------------------")

    def forward(self, x):
        # Input shape: (BT, H, W, C)
        # print(x.shape)

        B, T, H, W, C = x.shape

        x_id = x

        x = x.view(B, T, H, W, C)
        x = self.norm1(x)
        x = self.down_proj(x)

        # 🔸 Add Temporal Positional Encoding
        pos_emb = self.temporal_pos_emb[:T]  # (T, C)
        pos_emb = pos_emb[None, :, None, None, :]  # (1, T, 1, 1, C)
        x = x + pos_emb  # (B, T, H, W, C)

        # (B, T, H, W, C) -> (B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # Block 1
        stream1 = self.block1_bn1(self.block1_conv1x1(x))

        stream2 = self.block1_conv3x3_1(x)
        stream2 = self.block1_bn2(stream2)
        stream2 = F.gelu(stream2)
        stream2 = self.block1_conv3x3_2(stream2)
        stream2 = self.block1_bn3(stream2)

        x = stream1 + stream2

        # x = F.gelu(x)

        # Block 2
        residual = x
        x = self.block2_conv3x3_1(x)
        x = self.block2_bn1(x)
        x = F.gelu(x)
        x = self.block2_conv3x3_2(x)
        x = self.block2_bn2(x)

        x = x + residual
        x = F.gelu(x)

        # (B, C, T, H, W) -> (B, T, H, W, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm2(x)
        x = self.up_proj(x)

        # print(x.shape)


        # x = x.view(BT, H, W, C)
        return x_id + x



class TemporalAdapter(nn.Module):
    def __init__(self, in_channels, adapter_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.adapter_channels = adapter_channels

        # For reshaping in forward
        self.T = None

        # Norm and projection
        self.norm1 = nn.LayerNorm(in_channels)
        self.down_proj = nn.Linear(in_channels, adapter_channels)

        # Block 1
        # Stream 1
        self.block1_conv1x1 = nn.Conv3d(adapter_channels, adapter_channels, kernel_size=1, padding=0)
        self.block1_bn1 = nn.BatchNorm3d(adapter_channels)

        # Stream 2
        self.block1_conv3x3_1 = nn.Conv3d(adapter_channels, adapter_channels, kernel_size=3, padding=1)
        self.block1_bn2 = nn.BatchNorm3d(adapter_channels)
        self.block1_conv3x3_2 = nn.Conv3d(adapter_channels, adapter_channels, kernel_size=3, padding=1)
        self.block1_bn3 = nn.BatchNorm3d(adapter_channels)

        # Block 2
        self.block2_conv3x3_1 = nn.Conv3d(adapter_channels, adapter_channels, kernel_size=3, padding=1)
        self.block2_bn1 = nn.BatchNorm3d(adapter_channels)
        self.block2_conv3x3_2 = nn.Conv3d(adapter_channels, adapter_channels, kernel_size=3, padding=1)
        self.block2_bn2 = nn.BatchNorm3d(adapter_channels)

        # Norm and up projection
        self.norm2 = nn.LayerNorm(adapter_channels)
        self.up_proj = nn.Linear(adapter_channels, in_channels)

        # Bias init (optional)
        nn.init.constant_(self.down_proj.bias, 0.)
        nn.init.constant_(self.up_proj.bias, 0.)

        # Initialize conv weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        print("Using TemporalAdapter --------------------")

    def forward(self, x):
        # Input shape: (BT, H, W, C)
        BT, H, W, C = x.shape
        T = self.T
        B = BT // T

        x_id = x

        x = x.view(B, T, H, W, C)
        x = self.norm1(x)
        x = self.down_proj(x)

        # (B, T, H, W, C) -> (B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # Block 1
        stream1 = self.block1_bn1(self.block1_conv1x1(x))
        
        stream2 = self.block1_conv3x3_1(x)
        stream2 = self.block1_bn2(stream2)
        stream2 = F.gelu(stream2)
        stream2 = self.block1_conv3x3_2(stream2)
        stream2 = self.block1_bn3(stream2)

        x = stream1 + stream2

        # Block 2
        residual = x
        x = self.block2_conv3x3_1(x)
        x = self.block2_bn1(x)
        x = F.gelu(x)
        x = self.block2_conv3x3_2(x)
        x = self.block2_bn2(x)

        x = x + residual
        x = F.gelu(x)

        # (B, C, T, H, W) -> (B, T, H, W, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm2(x)
        x = self.up_proj(x)

        x = x.view(BT, H, W, C)
        return x_id + x

    

class ModifiedSwinLayer(nn.Module):
    def __init__(self, swin_layer, inC, adapter=3):
        super(ModifiedSwinLayer, self).__init__()
        self.swin_layer = swin_layer

        if adapter == 1:
            self.temporal_adapter = STAdapter(inC, adapter_channels=64, kernel_size=(3, 3, 3))
        if adapter == 2:
            self.temporal_adapter = TemporalAdapter(inC, adapter_channels=64)
        if adapter == 3:
            self.temporal_adapter = TemporalAdapterPE(inC, adapter_channels=64, max_frames=1000)
        if adapter == 4:
            self.temporal_adapter = STAdapterPE(inC, adapter_channels=64, kernel_size=(3, 3, 3))

        self.adapter = adapter
            

    def forward(self, x):
        x = self.swin_layer(x)
        x = self.temporal_adapter(x) + x
        
        return x


class SwinTransformer3d(nn.Module):
    def __init__(
        self,
        patch_size: list[int],
        embed_dim: int,
        depths: list[int],
        num_heads: list[int],
        window_size: list[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 400,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
        patch_embed: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        if block is None:
            block = partial(SwinTransformerBlock, attn_layer=ShiftedWindowAttention3d)

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        if patch_embed is None:
            patch_embed = PatchEmbed3d

        # split image into non-overlapping patches
        self.patch_embed = patch_embed(patch_size=patch_size, embed_dim=embed_dim, norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=dropout)

        layers: list[nn.Module] = []
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: list[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                        attn_layer=ShiftedWindowAttention3d,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        self.num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_weights(self, w_path):
        weights = torch.load(w_path, map_location="cpu")
        msg = self.load_state_dict(weights, strict=False)
        
        print(msg)

    def modify(self, adapter=3, ins = [96, 192, 384, 768], lora=False):
        self.adapter = adapter
        if adapter == 0: return
        if adapter:
            self.features = nn.Sequential(
                ModifiedSwinLayer(self.features[0], inC=ins[0], adapter=adapter),
                self.features[1],
                ModifiedSwinLayer(self.features[2], inC=ins[1], adapter=adapter),
                self.features[3],
                ModifiedSwinLayer(self.features[4], inC=ins[2], adapter=adapter),
                self.features[5],
                ModifiedSwinLayer(self.features[6], inC=ins[3], adapter=adapter),
            )

            if lora:
                for i in range(len(self.features)):
                    if isinstance(self.features[i], ModifiedSwinLayer):
                        self.features[i].swin_layer[0].attn.lorify(r=8, alpha=1.0)
                        self.features[i].swin_layer[1].attn.lorify(r=8, alpha=1.0)

            print("ADAPTERS INITIALIZED --------------------")

    def forward(self, x: Tensor) -> Tensor:
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.repeat(1, 1, 2, 1, 1, 1)  # (B, T, 2, C, H, W)
        x = x.view(B, 2 * T, C, H, W) # Merge the T and repeat dimensions
        x = x.permute(0, 2, 1, 3, 4)

        x = self.patch_embed(x)  # B _T _H _W C
        B, T, H, W, C = x.shape

        x = self.pos_drop(x)
        x = self.features(x)  # B _T _H _W C
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)  # B, C, _T, _H, _W
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = x.reshape(B, T, self.num_features).permute(0, 2, 1)  # (B, D, T)

        return x
