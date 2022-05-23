import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid, save_image
from utils import *
from copy import deepcopy

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

category_dict = {
    'voc': ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
    'coco': ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
}

def pil_loader(path):
    """
    path: the path to image
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def pil_saver(x, path):
    """
    x: numpy array
    """
    img = Image.fromarray(x)
    img.save(path)


def cv2_saver(x, path):
    """
    x: numpy array
    """
    cv2.imwrite(path, x)


def tensor2image(x):
    grid = make_grid(x.unsqueeze(0), nrow=1, padding=0, pad_value=0,
                     normalize=True, range=None)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    return grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()


def save_tensor_as_image(x, save_dir, fn=None):
    """
    x:        (N, 3, H, W)
    save_dir: --
    fn:       (N,)
    """
    N = x.size(0)
    for i in range(N):
        image = tensor2image(x[i])
        save_path = os.path.join(save_dir, str(i) if fn is None else fn[i])
        pil_saver(image, save_path)


def applyColorMap(x, am):
    """
    image: (3, H, W)
    am:    (H/f, W/f) range(0,1)
    """
    _, H, W = x.shape
    am = np.uint8(am * 255)

    colormap = cv2.applyColorMap(cv2.resize(am, (W, H)), cv2.COLORMAP_JET)
    image = tensor2image(x)

    cam = colormap + 0.4 * image
    cam = cam / np.max(cam)
    cam = np.uint8(cam * 255).copy()

    return cam


def save_colormap(x, am, save_dir, fn):
    """
    x(image):           shape likes (H, W)
    am(activation map): activation map (H/f, W/f)
    save_dir:           --
    fn(file name):      (1,)
    """
    colormap = applyColorMap(x, am)
    # pil_saver(colormap[...,::-1], os.path.join(save_dir, fn))
    cv2_saver(colormap, os.path.join(save_dir, fn))

def save_tensor_as_grid_image(x, save_dir, fn=None):
    pass

def visual_debug(x, target, am, save_dir, step, flag='step', num_classes=80, dataset='coco', phase='train'):

    n, c, h, w = x.size()
    categories = deepcopy(category_dict[dataset])
    if num_classes == 80 or num_classes == 20:
        categories.pop(0)

    for i in range(x.size(0)):
        image = get_numpy_from_tensor(x[i])
        image = denormalize(image, imagenet_mean, imagenet_std)  # [..., ::-1]

        temp = am[i, target[i].view(num_classes) == 1, :, :]
        label = torch.nonzero(target[i].view(num_classes) == 1, as_tuple=False).squeeze(1)

        fig, axes = plt.subplots(1, temp.size(0) + 1)
        fig.tight_layout()  # 调整整体空白
        plt.subplots_adjust(wspace=0, hspace=0)  # 调整子图间距
        axes[0].imshow(image)
        axes[0].axis('off')

        for j in range(label.size(0)):
            temp_ = get_numpy_from_tensor(temp[j])
            temp_ = cv2.resize(temp_, (h, w))
            axes[j + 1].imshow(temp_)
            axes[j + 1].set_title(categories[label[j]])
            axes[j + 1].axis('off')

        plt.savefig("{}/{}/{}_{}_{}.png".format(save_dir, phase, step, i, flag), bbox_inches='tight')
        plt.close()

        # if i == 10:
        #     break

def visual_debug_single(x, target, am, save_dir, step, flag='step', num_classes=80, dataset='coco', phase='train'):

    n, c, h, w = x.size()
    categories = deepcopy(category_dict[dataset])
    if num_classes == 80 or num_classes == 20:
        categories.pop(0)

    for i in range(x.size(0)):
        image = get_numpy_from_tensor(x[i])
        image = denormalize(image, imagenet_mean, imagenet_std)  # [..., ::-1]

        temp = am[i, 0]
        fig, axes = plt.subplots(1, 2)
        fig.tight_layout()  # 调整整体空白
        plt.subplots_adjust(wspace=0, hspace=0)  # 调整子图间距
        axes[0].imshow(image)
        axes[0].axis('off')

        temp_ = get_numpy_from_tensor(temp)
        temp_ = cv2.resize(temp_, (h, w))
        axes[1].imshow(temp_)
        axes[1].axis('off')

        plt.savefig("{}/{}/{}_{}_{}.png".format(save_dir, phase, step, i, flag), bbox_inches='tight')
        plt.close()

        # if i == 10:
        #     break







































