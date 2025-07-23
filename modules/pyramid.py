import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Pyramid(nn.Module):
    def __init__(self):
        super(Pyramid, self).__init__()
        self.up2_to_1 = UpsampleBlock(768, 384)
        self.up1_to_0 = UpsampleBlock(384, 192)

        # Pooling to collapse H, W → 1, 1
        self.spatial_pool = lambda x: F.adaptive_avg_pool3d(x, (x.shape[2], 1, 1))  # (T, 1, 1)
        self.B = 2

    def forward(self, pl):
        # pl: list of tensors [pl0, pl1, pl2] with shape [BT, H, W, C]
        out = []

        # Reshape from [BT, H, W, C] to [B, T, C, H, W]
        def to_5d(x):
            BT, H, W, C = x.shape
            T = BT // self.B
            return x.view(self.B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        def to_4d(x):
            # x is [B, C, T, 1, 1] → [BT, 1, 1, C]
            B, C, T, _, _ = x.shape
            return x.permute(0, 2, 3, 4, 1).reshape(B, T, C).permute(0, 2, 1).contiguous()

        x2 = to_5d(pl[2])  # [B, T, C, H, W]
        x1 = to_5d(pl[1])
        x0 = to_5d(pl[0])

        # Step 1: Upsample x2 and add to x1
        up_x2 = self.up2_to_1(x2)
        x1 = x1 + up_x2
        pooled1 = self.spatial_pool(x1)  # [B, T, C, 1, 1]
        out.append(to_4d(pooled1))       # append [BT, 1, 1, C]

        # Step 2: Upsample x1 and add to x0
        up_x1 = self.up1_to_0(x1)
        x0 = x0 + up_x1
        pooled0 = self.spatial_pool(x0)
        out.append(to_4d(pooled0))

        return out


if __name__ == "__main__":
    BT = 32

    pl0 = torch.randn(BT, 28, 28, 192)  # block 1 output
    pl1 = torch.randn(BT, 14, 14, 384)  # block 2 output
    pl2 = torch.randn(BT, 7, 7, 768)    # block 3 output

    # Stack into a list
    pyramid_inputs = [pl0, pl1, pl2]

    # Create the model
    model = Pyramid()

    # Forward pass
    outputs = model(pyramid_inputs)

    # Check outputs
    for i, out in enumerate(outputs):
        print(f"Output {i} shape:", out.shape)
