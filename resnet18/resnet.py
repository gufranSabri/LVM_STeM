from functools import partial
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

import torchvision
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



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
        BT, C, H, W = x.shape
        T = self.T
        B = BT // T

        x_id = x

        x = x.reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
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

        x = x.view(BT, H, W, C).permute(0, 3, 1, 2)  # (BT, C, H, W)
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
        BT, C, H, W = x.shape
        T = self.T
        B = BT // T

        x_id = x

        x = x.reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
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

        x = x.view(BT, H, W, C).permute(0, 3, 1, 2)  # (BT, C, H, W)
        return x_id + x


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
    
    def load_weights(self, weights):
        msg = self.load_state_dict(weights.get_state_dict(progress=True))
        print(msg)

    def modify(self, adapter=3, ins = [96, 192, 384, 768]):
        self.adapter = adapter
        if adapter == 0: return
        if adapter:
            self.layer1 = ModifiedResLayer(self.layer1, inC=ins[0], adapter=adapter)
            self.layer2 = ModifiedResLayer(self.layer2, inC=ins[1], adapter=adapter)
            self.layer3 = ModifiedResLayer(self.layer3, inC=ins[2], adapter=adapter)
            self.layer4 = ModifiedResLayer(self.layer4, inC=ins[3], adapter=adapter)

            print("ADAPTERS INITIALIZED --------------------")

    def _forward_impl(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # Reshape

        if self.adapter != 0:
            self.layer1.T = T
            self.layer2.T = T
            self.layer3.T = T
            self.layer4.T = T

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) # [BT, 64, 56, 56]
        x = self.layer2(x) # [BT, 128, 28, 28]
        x = self.layer3(x) # [BT, 256, 14, 14]
        x = self.layer4(x) # [BT, 512, 7, 7]

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = x.reshape(B, T, -1).permute(0,2,1)  # Reshape back to (B, D, T)
        
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    

class ModifiedResLayer(nn.Module):
    def __init__(self, res_layer, inC, adapter=3):
        super(ModifiedResLayer, self).__init__()
        self.res_layer = res_layer

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
        self.temporal_adapter.T = self.T
        x = self.res_layer(x)
        x = self.temporal_adapter(x) + x
        
        return x



if __name__ == "__main__":
    resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
    model_w = torchvision.models.ResNet18_Weights.DEFAULT
    resnet18.load_weights(model_w)
    resnet18.modify(adapter=3, ins=[64, 128, 256, 512])

    x = torch.randn(1, 8, 3, 224, 224)

    y = resnet18(x)
    print(y.shape)
