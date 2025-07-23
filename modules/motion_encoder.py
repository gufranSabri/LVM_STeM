import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEImageProcessor, VideoMAEModel

from transformers import VideoMAEModel, VideoMAEImageProcessor
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn
import torch
import numpy as np

class SpatiotemporalProjector(nn.Module):
    def __init__(self, target_h=None, target_w=None, target_c=None):
        """
        target_t: desired temporal length (optional)
        target_h: desired height (optional)
        target_w: desired width (optional)
        target_c: desired channels (optional)
        """
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w
        self.target_c = target_c

        # Placeholder for 1x1 conv to project channels
        self.channel_proj = None
        if target_c is not None:
            self.channel_proj = nn.Linear(in_features=768, out_features=target_c)

    def forward(self, x):  # x: [B, T, H, W, C]
        B, T, H, W, C = x.shape

        # Reshape to [B*T, C, H, W] for spatial ops
        x = x.view(B * T, H, W, C).permute(0, 3, 1, 2)  # [B*T, C, H, W]

        # If projecting channels
        if self.channel_proj is not None:
            # Project per-pixel using linear layer (flatten spatial)
            x = x.permute(0, 2, 3, 1)  # [B*T, H, W, C]
            x = x.to(torch.float32)
            x = self.channel_proj(x)   # [B*T, H, W, target_c]
            x = x.permute(0, 3, 1, 2)  # [B*T, target_c, H, W]
            C = self.target_c

        # If resizing spatial dims
        if self.target_h is not None or self.target_w is not None:
            h = self.target_h if self.target_h is not None else H
            w = self.target_w if self.target_w is not None else W
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        # Reshape back to [B, T, H, W, C]
        H_new, W_new = x.shape[-2:]
        x = x.view(B, T, C, H_new, W_new).permute(0, 1, 3, 4, 2)  # [B, T, H, W, C]

        # If resizing temporal dim
        if self.target_t is not None and self.target_t != T:
            B, T, H, W, C = x.shape
            # Rearrange to [B, C, T, H, W]
            x = x.permute(0, 4, 1, 2, 3)
            # Interpolate the temporal dimension
            x = F.interpolate(x, size=(self.target_t, H, W), mode='trilinear', align_corners=False)
            # Rearrange back to [B, new_T, H, W, C]
            x = x.permute(0, 2, 3, 4, 1)

        return x


def denormalize(video):
    # video: [B, T, C, H, W] assumed to be normalized as ((x/255. - 0.45) / 0.225)
    video = video * 0.225 + 0.45
    video = torch.clamp(video, 0, 1)  # ensure within [0,1]
    return video


# class VideoMAEEncoder(nn.Module):
#     def __init__(self, model_name="MCG-NJU/videomae-base", target_h=7, target_w=7, target_c=256):
#         super().__init__()
#         self.processor = VideoMAEImageProcessor.from_pretrained(model_name, do_rescale=False)
#         self.model = VideoMAEModel.from_pretrained(model_name, torch_dtype=torch.float16)

#         # print(self.model)

#         # Setup LoRA config
#         lora_config = LoraConfig(
#             r=8,
#             lora_alpha=16,
#             target_modules=["query", "valye", "key"],  # Q and V projections in attention
#             lora_dropout=0.1,
#             bias="none",
#             task_type=TaskType.FEATURE_EXTRACTION
#         )

#         self.model = get_peft_model(self.model, lora_config)
#         self.model.train()  # Only LoRA layers will train

#         # print number of trainable parameters
#         trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
#         print(f"Number of trainable parameters in VideoMAEEncoder: {trainable_params}")

#         self.proj1 = SpatiotemporalProjector(target_h=56, target_w=56, target_c=96)
#         self.proj2 = SpatiotemporalProjector(target_h=28, target_w=28, target_c=192)
#         self.proj3 = SpatiotemporalProjector(target_h=14, target_w=14, target_c=384)
#         self.proj4 = SpatiotemporalProjector(target_h=7, target_w=7, target_c=768)

#     def forward(self, video):  # video: (B, T, H, W, C)
#         B, T, H, W, C = video.shape
#         video = denormalize(video)  # Denormalize
#         video_np = video.permute(0, 1, 4, 2, 3).cpu().numpy()
#         video_list = [[np.transpose(frame, (1, 2, 0)) for frame in vid] for vid in video_np]

#         new_video_list1, new_video_list2, new_video_list3, new_video_list4 = [], [], [], []

#         for i in range(len(video_list)):
#             vid = video_list[i]
#             if len(vid) % 16 != 0:
#                 pad_len = 16 - (len(vid) % 16)
#                 vid.extend([vid[-1]] * pad_len)

#             clippified_vid = [vid[j:j+16] for j in range(0, len(vid), 16)]
#             inputs = self.processor(clippified_vid, return_tensors="pt")
#             inputs = {k: v.to("cuda").half() for k, v in inputs.items()}
#             outputs = self.model(**inputs)

#             outputs = outputs.last_hidden_state.reshape(len(clippified_vid), -1, 14, 14, 768)
#             outputs = torch.cat([o for o in outputs], dim=0).unsqueeze(0)  # (B, T, H, W, C)

#             self.proj1.target_t = outputs.shape[1]
#             self.proj2.target_t = outputs.shape[1]
#             self.proj3.target_t = outputs.shape[1]
#             self.proj4.target_t = outputs.shape[1]

#             new_video_list1.append(self.proj1(outputs).squeeze(0))
#             new_video_list2.append(self.proj2(outputs).squeeze(0))
#             new_video_list3.append(self.proj3(outputs).squeeze(0))
#             new_video_list4.append(self.proj4(outputs).squeeze(0))

#         new_video_list1 = torch.stack(new_video_list1).reshape(B*T, 56, 56, 96)
#         new_video_list2 = torch.stack(new_video_list2).reshape(B*T, 28, 28, 192)
#         new_video_list3 = torch.stack(new_video_list3).reshape(B*T, 14, 14, 384)
#         new_video_list4 = torch.stack(new_video_list4).reshape(B*T, 7, 7, 768)
#         return new_video_list1, new_video_list2, new_video_list3, new_video_list4

class VideoMAEEncoder(nn.Module):
    def __init__(self, model_name="MCG-NJU/videomae-base", target_h=7, target_w=7, target_c=256):
        super().__init__()
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name, do_rescale=False)
        self.model = VideoMAEModel.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.proj1 = SpatiotemporalProjector(
            target_h=56,  # Desired height
            target_w=56,  # Desired width
            target_c=96  # Desired channels (output dimension)
        )
        self.proj2 = SpatiotemporalProjector(
            target_h=28,  # Desired height
            target_w=28,  # Desired width
            target_c=192  # Desired channels (output dimension)
        )
        self.proj3 = SpatiotemporalProjector(
            target_h=14,  # Desired height
            target_w=14,  # Desired width
            target_c=384  # Desired channels (output dimension)
        )
        self.proj4 = SpatiotemporalProjector(
            target_h=7,  # Desired height
            target_w=7,  # Desired width
            target_c=768  # Desired channels (output dimension)
        )

    def forward(self, video):  # video: (B, T, H, W, C)
        B, T, H, W, C = video.shape
        video = denormalize(video)  # Denormalize the video first
        video_np = video.permute(0, 1, 4, 2, 3).cpu().numpy()  # (B, T, C, H, W) → needed shape
        video_list = [
            [np.transpose(frame, (1, 2, 0)) for frame in vid]  # each frame: (C, H, W) → (H, W, C)
            for vid in video_np
        ]

        new_video_list1 = []
        new_video_list2 = []
        new_video_list3 = []
        new_video_list4 = []
        
        for i in range(len(video_list)):
            vid = video_list[i]
            if len(vid) % 16 != 0:
                # Pad the video to make its length a multiple of 16
                pad_length = 16 - (len(vid) % 16)
                vid.extend([vid[-1]] * pad_length)

            clippified_vid = []
            for j in range(0, len(vid), 16):
                clippified_vid.append(vid[j:j+16])

            # with torch.no_grad():
            inputs = self.processor(clippified_vid, return_tensors="pt")
            inputs = {k: v.to("cuda").half() for k, v in inputs.items()}
            outputs = self.model(**inputs)

            outputs = outputs.last_hidden_state.reshape(len(clippified_vid), -1, 14, 14, 768)
            outputs = [o for o in outputs]
            outputs = torch.cat(outputs, dim=0).unsqueeze(0)  # (B, T, H, W, C)

            self.proj1.target_t = self.T
            self.proj2.target_t = self.T
            self.proj3.target_t = self.T
            self.proj4.target_t = self.T
            
            outputs1 = self.proj1(outputs).squeeze(0)  # Remove batch dimension
            outputs2 = self.proj2(outputs).squeeze(0)  # Remove batch dimension
            outputs3 = self.proj3(outputs).squeeze(0)  # Remove batch dimension
            outputs4 = self.proj4(outputs).squeeze(0)  # Remove batch dimension

            new_video_list1.append(outputs1)
            new_video_list2.append(outputs2)
            new_video_list3.append(outputs3)
            new_video_list4.append(outputs4)

        
        new_video_list1 = torch.stack(new_video_list1).reshape(B*T, 56, 56, 96)
        new_video_list2 = torch.stack(new_video_list2).reshape(B*T, 28, 28, 192)
        new_video_list3 = torch.stack(new_video_list3).reshape(B*T, 14, 14, 384)
        new_video_list4 = torch.stack(new_video_list4).reshape(B*T, 7, 7, 768)
        return new_video_list1, new_video_list2, new_video_list3, new_video_list4