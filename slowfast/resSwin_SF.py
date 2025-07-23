import torch
import torch.nn as nn
import torch.nn.functional as F
from swin.swin_transformer import SwinTransformer
from resnet18.resnet import *
from slowfast_modules.fuse_helper import *


class SlowFastLoader(nn.Module):
    def __init__(self, alpha):
        super(SlowFastLoader, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        assert(len(x.shape) == 5)

        x_f = x[:]
        x_s = x[:,:,::self.alpha]

        return (x_s, x_f)
    

class ResSwinSF(nn.Module):
    def __init__(self, alpha, ins1, ins2, **configs):
        super().__init__()

        self.slow_stream =  SwinTransformer(**configs["1"])
        self.fast_stream = ResNet(BasicBlock, [2, 2, 2, 2])

        self.alpha = alpha
        self.loader = SlowFastLoader(alpha)

        self.fuse_helper_ft1 = FuseBiAdd_d(
            dim_in1=ins2[0],
            dim_in2=ins1[0],
            fusion_kernel=5,
            alpha=alpha,
            beta_inv=1,
        )
        self.fuse_helper_ft2 = FuseBiAdd_d(
            dim_in1=ins2[1],
            dim_in2=ins1[1],
            fusion_kernel=5,
            alpha=alpha,
            beta_inv=1,
        )
        self.fuse_helper_ft3 = FuseBiAdd_d(
            dim_in1=ins2[2],
            dim_in2=ins1[2],
            fusion_kernel=5,
            alpha=alpha,
            beta_inv=1,
        )
        self.fuse_helper_ft4 = FuseBiAdd_d(
            dim_in1=ins2[3],
            dim_in2=ins1[3],
            fusion_kernel=5,
            alpha=alpha,
            beta_inv=1,
        )

    def load_weights(self, model_w1, model_w2):
        self.slow_stream.load_weights(model_w1)
        self.fast_stream.load_weights(model_w2)

    def modify(self, adapter=3, ins1=[96, 192, 384, 768], ins2=[64, 128, 256, 512], lora=False):
        self.slow_stream.modify(adapter=adapter, ins=ins1, lora=lora)
        self.fast_stream.modify(adapter=adapter, ins=ins2)
        self.adapter = adapter

    def preprocesssing(self, x_s, x_f):
        x_f = self.fast_stream.conv1(x_f)
        x_f = self.fast_stream.bn1(x_f)
        x_f = self.fast_stream.relu(x_f)
        x_f = self.fast_stream.maxpool(x_f)

        return x_s, x_f
    
    def block_forward(self, f_idx, x_s, x_f, T_s, T_f):
        # [B_Ts, H, W, C] [B_Tf, C, H, W]

        x_s = self.slow_stream.features[f_idx](x_s)
        x_s = self.slow_stream.features[f_idx + 1](x_s)

        if f_idx == 0: x_f = self.fast_stream.layer1(x_f)
        elif f_idx == 2: x_f = self.fast_stream.layer2(x_f)
        elif f_idx == 4: x_f = self.fast_stream.layer3(x_f)
        elif f_idx == 6: x_f = self.fast_stream.layer4(x_f)

        # [2, H, W, C] [8, C, H, W]

        BT_s, H, W, C = x_s.shape
        BT_f, C, H, W = x_f.shape
        x_s = x_s.view(BT_s // T_s, T_s, H, W, -1).permute(0, 4, 1, 2, 3)
        x_f = x_f.view(BT_f // T_f, T_f, C, H, W).permute(0, 2, 1, 3, 4)

        if f_idx == 0:
            x_s, x_f = self.fuse_helper_ft1([x_s, x_f])
        elif f_idx == 2:
            x_s, x_f = self.fuse_helper_ft2([x_s, x_f])
        elif f_idx == 4:
            x_s, x_f = self.fuse_helper_ft3([x_s, x_f])
        elif f_idx == 6:
            x_s, x_f = self.fuse_helper_ft4([x_s, x_f])

        # [B, C, T_s, H, W] [B, C, T_f, H, W]

        x_s = x_s.permute(0, 2, 3, 4, 1).contiguous().reshape(BT_s, H, W, -1)
        x_f = x_f.permute(0, 2, 3, 4, 1).contiguous().reshape(BT_f, C, H, W)

        # [B_Ts, H, W, C] [B_Tf, C, H, W]

        return x_s, x_f
    

    def post_block_forward(self, x_s, x_f, T_s, T_f):
        BT_s, H, W, C = x_s.shape
        BT_f, C, H, W = x_f.shape

        x_s = self.slow_stream.norm(x_s)
        x_s = self.slow_stream.permute(x_s)
        x_s = self.slow_stream.avgpool(x_s)
        x_s = self.slow_stream.flatten(x_s)

        x_f = self.fast_stream.avgpool(x_f)
        x_f = torch.flatten(x_f, 1)

        x_s = x_s.reshape(BT_s // T_s, T_s, -1).permute(0, 2, 1)
        x_f = x_f.reshape(BT_f // T_f, T_f, -1).permute(0, 2, 1)

        return x_s, x_f


    def forward(self, x, x_hm):
        x_s, x_f = x_hm, x
        del x_hm

        # x = self.loader(x) # [B, C, T//4, H, W] || [B, C, T, H, W]

        # x_s, x_f = x
        x_s = x_s.permute(0, 2, 1, 3, 4)
        x_f = x_f.permute(0, 2, 1, 3, 4)

        B, T_f, C, H, W = x_f.shape
        B, T_s, C, H, W = x_s.shape

        if self.adapter != 0:
            self.slow_stream.features[1].T = T_s
            self.slow_stream.features[3].T = T_s
            self.slow_stream.features[5].T = T_s
            self.slow_stream.features[7].T = T_s

            self.fast_stream.layer1.T = T_f
            self.fast_stream.layer2.T = T_f
            self.fast_stream.layer3.T = T_f
            self.fast_stream.layer4.T = T_f

        x_s = x_s.reshape(B * T_s, C, H, W)
        x_f = x_f.reshape(B * T_f, C, H, W)

        x_s, x_f = self.preprocesssing(x_s, x_f)

        x_s, x_f = self.block_forward(0, x_s, x_f, T_s, T_f)
        x_s, x_f = self.block_forward(2, x_s, x_f, T_s, T_f)
        x_s, x_f = self.block_forward(4, x_s, x_f, T_s, T_f)
        x_s, x_f = self.block_forward(6, x_s, x_f, T_s, T_f)

        x_s, x_f = self.post_block_forward(x_s, x_f, T_s, T_f)
        x_s = F.interpolate(x_s, size=(T_f,), mode='linear', align_corners=False)

        x = torch.cat([x_s, x_f], dim=1)

        return x