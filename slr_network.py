import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
from modules.mstcn import MSTCN
from swin.swin_transformer import SwinTransformer
from vit.vit import _vision_transformer
import torchvision.models as vision_models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True,
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights

        c2d_type, t_model = c2d_type.split("_")[0], c2d_type.split("_")[1]
        self.t_model = t_model
        self.c2d_type = c2d_type

        print(f"c2d_type: {c2d_type}, t_model: {t_model}")

        if "swin" in c2d_type:
            swin_t_config = {
                "patch_size": [4, 4],
                "embed_dim": 96,
                "depths": [2, 2, 6, 2],  # Swin-T specific depth configuration
                "num_heads": [3, 6, 12, 24],  # Number of attention heads
                "window_size": [7, 7],  # Window size for local self-attention
                "mlp_ratio": 4.0,
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "stochastic_depth_prob": 0.2,  # Higher stochastic depth than Swin-S
                "num_classes": 1000,  # Default for ImageNet, change if needed
            }

            swin_s_config = {
                "patch_size": [4, 4],
                "embed_dim": 96,
                "depths": [2, 2, 18, 2],  # Swin-S specific depth configuration
                "num_heads": [3, 6, 12, 24],  # Number of attention heads
                "window_size": [7, 7],  # Window size for local self-attention
                "mlp_ratio": 4.0,
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "stochastic_depth_prob": 0.2,  # Higher stochastic depth than Swin-S
                "num_classes": 1000,  # Default for ImageNet, change if needed
            }

            swin_b_config = {
                "patch_size": [4, 4],
                "embed_dim": 128,
                "depths": [2, 2, 18, 2],  # Swin-B specific depth configuration
                "num_heads": [4, 8, 16, 32],  # Number of attention heads
                "window_size": [7, 7],  # Window size for local self-attention
                "mlp_ratio": 4.0,
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "stochastic_depth_prob": 0.2,  # Higher stochastic depth than Swin-S
                "num_classes": 1000,  # Default for ImageNet, change if needed
            }

            configs = {
                "swins": swin_s_config,
                "swint": swin_t_config,
                "swinb": swin_b_config,
            }
            models = {
                "swins": vision_models.swin_s,
                "swint": vision_models.swin_t,
                "swinb": vision_models.swin_b,
            }
            weights = {
                "swins": vision_models.Swin_S_Weights.IMAGENET1K_V1,
                "swint": vision_models.Swin_T_Weights.IMAGENET1K_V1,
                "swinb": vision_models.Swin_B_Weights.IMAGENET1K_V1,
            }
            ins = {
                "swint": [96, 192, 384, 768],
                "swins": [96, 192, 384, 768],
                "swinb": [128, 256, 512, 1024],
            }


            model_w = models[c2d_type](weights=weights[c2d_type])
            self.conv2d = SwinTransformer(**configs[c2d_type])
            self.conv2d.load_weights(model_w)
            self.conv2d.modify(adapter=int(self.t_model.split("-")[1]) if "-" in self.t_model else 0, ins = ins[c2d_type])
            del model_w

            hidden_size = 768 if c2d_type in ["swint", "swins"] else 1024

            print("Swin model loaded")

        elif "vit" in c2d_type:
            ins = {
                "vitb": 768,
                "vitl": 1024,
                "vith": 1280,
            }

            if c2d_type == "vitb":
                model_w = vision_models.vit_b_16(weights=vision_models.ViT_B_16_Weights.IMAGENET1K_V1)
                self.conv2d = _vision_transformer(
                    patch_size=16,
                    num_layers=12,
                    num_heads=12,
                    hidden_dim=768,
                    mlp_dim=3072,
                    progress=True,
                )

            if c2d_type == "vitl":
                model_w = vision_models.vit_l_16(weights=vision_models.ViT_L_16_Weights.IMAGENET1K_V1)
                self.conv2d = _vision_transformer(
                    patch_size=16,
                    num_layers=24,
                    num_heads=16,
                    hidden_dim=1024,
                    mlp_dim=4096,
                    progress=True,
                )

            if c2d_type == "vith":
                model_w = vision_models.vit_h_14(weights=vision_models.ViT_H_14_Weights.IMAGENET1K_V1)
                self.conv2d = _vision_transformer(
                    patch_size=14,
                    num_layers=32,
                    num_heads=16,
                    hidden_dim=1280,
                    mlp_dim=5120,
                    progress=True,
                )

            msg = self.conv2d.load_state_dict(model_w.state_dict())
            print(msg)
            self.conv2d.modify(adapter=int(self.t_model.split("-")[1]) if "-" in self.t_model else 0, inC = ins[c2d_type])
            del model_w

            hidden_size = ins[c2d_type]

            print("VIT model loaded")


        if "mstcn" in self.t_model:
            print(f"Using MSTCN")
            self.conv1d = MSTCN(input_size=hidden_size, hidden_size=hidden_size, num_classes=num_classes)
        else:
            print(f"Using TemporalConv")
            self.conv1d = TemporalConv(input_size=hidden_size, hidden_size=hidden_size, num_classes=num_classes, conv_type=conv_type, use_bn=use_bn)

        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier

        # print model summary
        print(f"Model summary for {c2d_type} with {t_model}:")
        num_params = sum(p.numel() for p in self.conv2d.parameters() if p.requires_grad)
        print(f"Total number of parameters: {num_params / 1e6:.2f}M")

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            framewise = self.conv2d(x.permute(0,2,1,3,4)) # framewise -> [2, 2304, 188] -> [B, D, T]
        else:
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        x = conv1d_outputs['visual_feat'] # x: T, B, C
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(conv1d_outputs['visual_feat'], lgt)
        outputs = self.classifier(tm_outputs['predictions'])

        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            "framewise_features": framewise,
            "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
            "loss_LiftPool_u": conv1d_outputs['loss_LiftPool_u'] if 'loss_LiftPool_u' in conv1d_outputs.keys() else None,
            "loss_LiftPool_p": conv1d_outputs['loss_LiftPool_p'] if 'loss_LiftPool_p' in conv1d_outputs.keys() else None,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
                
            elif k == 'Cu':
                loss += weight*ret_dict['loss_LiftPool_u'] if ret_dict['loss_LiftPool_u'] is not None else 0
            elif k == 'Cp':
                loss += weight*ret_dict['loss_LiftPool_p'] if ret_dict['loss_LiftPool_p'] is not None else 0

        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss