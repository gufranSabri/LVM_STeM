import argparse
import cv2
import numpy as np
import torch
import timm
from slr_network_gc import SLRModel
from dataset.dataloader_video import BaseFeeder
from tqdm import tqdm

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True, help='Use NVIDIA GPU acceleration')
    parser.add_argument('--method', type=str, default='scorecam', help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
    parser.add_argument('--aug_smooth', action='store_true', help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true', help='Reduce noise by taking the first principle componenet of cam_weights*activations')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=7, width=7):
    T = tensor.shape[0]
    tensor = tensor[T//2:(T//2)+1]
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(3))

    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    model = SLRModel(1296, "swins_mstcn-3", 2)
    cnn_sd = torch.load("/data/ahmed026/phoenix/swinS_TAPE_MSTCN/_best_model.pt", map_location=torch.device('cpu'))["model_state_dict"]
    for k in list(cnn_sd.keys()):
        if 'conv2d' in k:
            new_k = k.replace('conv2d.module.', 'conv2d.')
            cnn_sd[new_k] = cnn_sd.pop(k)

    msg = model.load_state_dict(cnn_sd)
    print(msg)
    model.eval()

    target_layers = [model.conv2d.norm]

    cam = methods[args.method](model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    arg = {'mode': 'test', 'datatype': 'video', 'num_gloss': -1, 'drop_ratio': 1.0, 'frame_interval': 1, 'image_scale': 1.0, 'input_size': 224, 'prefix': '/data/ahmed026/datasets/phoenix2014', 'transform_mode': False}
    dataset = BaseFeeder(
        gloss_dict=np.load("./preprocess/phoenix2014/gloss_dict.npy", allow_pickle=True).item(),
        kernel_size=['K5', 'P2', 'K5', 'P2'],
        dataset='phoenix2014',
        **arg
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=1,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )

    data = next(iter(loader))

    vid = data[0]
    vid_lgt = data[1]
    label = data[2]
    label_lgt = data[3]
    act_label = data[4]

    T = vid.shape[1]

    for i in tqdm(range(T)):
        input_tensor = vid[:,i,:,:,:]  # [B, T, C, H, W] -> [B, C, H, W]
        rgb_image = input_tensor.permute(0, 2, 3, 1)
        rgb_image = rgb_image.squeeze().numpy()

        cam.batch_size = 1
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=None,
            eigen_smooth=args.eigen_smooth,
            aug_smooth=args.aug_smooth
        )

        grayscale_cam = grayscale_cam[0, :]

        grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())
        grayscale_cam = (grayscale_cam * 255).astype(np.uint8)

        colored_cam = cv2.applyColorMap(grayscale_cam, cv2.COLORMAP_JET)

        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        rgb_image = (rgb_image * 255).astype(np.uint8)

        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(rgb_image, 0.3, colored_cam, 0.7, 0)

        cv2.imwrite(f'./gradcams/swin/{i}.jpg', overlay)