import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, NamedTuple, Optional

import torch
import torch.nn as nn
from vit.misc import MLP, Conv2dNormActivation
import torchvision.models as vision_models
import torch.nn.functional as F




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
        x_id = x

        BT, HW, C = x.shape
        HW-=1
        
        cls_token = x[:, 0, :].unsqueeze(1)
        x = x[:, 1:, :]
        x = x.reshape(BT, int(HW**0.5), int(HW**0.5), C)

        BT, H, W, C = x.shape
        assert C == self.in_channels
        T = self.T
        B = BT // T

        

        x = x.view(B, T, H, W, C)
        x = self.norm1(x)
        x = self.down_proj(x)

        x = x.permute(0, 4, 1, 2, 3).contiguous()

        x = self.dw_conv(x)

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm2(x)
        x = self.up_proj(x)

        x = x.view(BT, H, W, C)
        x = x.reshape(BT, HW, C)
        x = torch.cat([cls_token, x], dim=1)

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
        x_id = x

        BT, HW, C = x.shape
        HW-=1
        
        cls_token = x[:, 0, :].unsqueeze(1)
        x = x[:, 1:, :]
        x = x.reshape(BT, int(HW**0.5), int(HW**0.5), C)

        # Input shape: (BT, H, W, C)
        BT, H, W, C = x.shape
        T = self.T
        B = BT // T

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
        x = x.reshape(BT, HW, C)
        x = torch.cat([cls_token, x], dim=1)

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
        x_id = x

        # Input shape: (BT, H, W, C)
        BT, HW, C = x.shape
        HW-=1
        
        cls_token = x[:, 0, :].unsqueeze(1)
        x = x[:, 1:, :]
        x = x.reshape(BT, int(HW**0.5), int(HW**0.5), C)


        BT, H, W, C = x.shape
        T = self.T
        B = BT // T

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
        x = x.reshape(BT, HW, C)
        x = torch.cat([cls_token, x], dim=1)

        return x_id + x
    

class ModifiedVITLayer(nn.Module):
    def __init__(self, VIT_layer, inC, adapter=3):
        super(ModifiedVITLayer, self).__init__()
        self.VIT_layer = VIT_layer

        if adapter == 1:
            self.temporal_adapter = STAdapter(inC, adapter_channels=64, kernel_size=(3, 3, 3))
        if adapter == 2:
            self.temporal_adapter = TemporalAdapter(inC, adapter_channels=64)
        if adapter == 3:
            self.temporal_adapter = TemporalAdapterPE(inC, adapter_channels=64, max_frames=1000)

    def forward(self, x):
        self.temporal_adapter.T = self.T
        x = self.VIT_layer(x)
        x = self.temporal_adapter(x) + x
        
        return x



class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def modify(self, adapter=3, inC=768):
        self.adapter = adapter
        if adapter == 0: return
        if adapter:
            self.temporals = nn.ModuleList()
            for i in range(len(self.layers)):
                self.layers[i] = ModifiedVITLayer(self.layers[i], inC=inC, adapter=adapter)

            print("ADAPTERS INITIALIZED --------------------")

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding

        input = self.dropout(input)
        for i in range(len(self.layers)):
            self.layers[i].T = self.T
            input = self.layers[i](input)

        input = self.ln(input)

        return input


        # return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[list[ConvStemConfig]] = None,
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x
    
    def modify(self, adapter=3, inC=768):
        self.encoder.modify(adapter=adapter, inC=inC)

    def forward(self, x: torch.Tensor):
        x = x.permute(0,2,1,3,4)
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)

        x = self._process_input(x)
        n = x.shape[0]

        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        self.encoder.T = T
        x = self.encoder(x)
        x = x[:, 0].reshape(B, T, -1).permute(0, 2, 1)

        return x


def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    progress: bool,
    **kwargs: Any,
) -> VisionTransformer:

    image_size = kwargs.pop("image_size", 224)

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
    )

    return model
