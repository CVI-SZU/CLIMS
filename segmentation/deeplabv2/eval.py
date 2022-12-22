
from __future__ import absolute_import, division, print_function

import json
import multiprocessing
import os

import click
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchnet.meter import MovingAverageValueMeter
from tqdm import tqdm

from libs.datasets import get_dataset
from libs.models import DeepLabV2_ResNet101_MSC
from libs.utils import DenseCRF, PolynomialLR, scores
import argparse

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")
    return device

def evaluation(config_path, model_path, cuda=True):
    """
    Evaluation on test set
    """

    # Configuration
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.set_grad_enabled(False)

    # Dataset
    dataset = get_dataset('voc_test')(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.TEST,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=False,
    )
    print(dataset)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=False,
    )

    # Model
    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model = nn.DataParallel(model)
    model.eval()
    model.to(device)

    # Path to save logits
    logit_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "features",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.TEST,
        "logit",
    )
    makedirs(logit_dir)
    print("Logit dst:", logit_dir)

    # Path to save scores
    save_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "scores",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.TEST,
    )
    # makedirs(save_dir)
    # save_path = os.path.join(save_dir, "scores.json")
    # print("Score dst:", save_path)

    preds, gts = [], []
    for image_ids, images in tqdm(
        loader, total=len(loader), dynamic_ncols=True
    ):
        # Image
        images = images.to(device)

        # Forward propagation
        logits = model(images)

        # Save on disk for CRF post-processing
        for image_id, logit in zip(image_ids, logits):
            filename = os.path.join(logit_dir, image_id + ".npy")
            np.save(filename, logit.cpu().numpy())

        # Pixel-wise labeling
        # _, H, W = images.shape
        # logits = F.interpolate(
        #     logits, size=(H, W), mode="bilinear", align_corners=False
        # )
        # probs = F.softmax(logits, dim=1)
        # labels = torch.argmax(probs, dim=1)
        #
        # preds += list(labels.cpu().numpy())
        # gts += list(gt_labels.numpy())

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    # score = scores(gts, preds, n_class=CONFIG.DATASET.N_CLASSES)

    # with open(save_path, "w") as f:
    #     json.dump(score, f, indent=4, sort_keys=True)

def crf_eval(config_path, n_jobs=multiprocessing.cpu_count()):
    """
    CRF post-processing on pre-computed logits
    """

    # Configuration
    CONFIG = OmegaConf.load(config_path)
    torch.set_grad_enabled(False)
    print("# jobs:", n_jobs)

    # Dataset
    dataset = get_dataset('voc_test')(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.TEST,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=False,
    )
    print(dataset)

    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )

    # Path to logit files
    logit_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "features",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.TEST,
        "logit",
    )
    print("Logit src:", logit_dir)
    if not os.path.isdir(logit_dir):
        print("Logit not found, run first: python main.py test [OPTIONS]")
        quit()

    # Path to save scores
    save_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "features",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.TEST,
        "results/VOC2012/Segmentation/comp5_test_cls",
    )
    makedirs(save_dir)
    # save_path = os.path.join(save_dir, "scores_crf.json")
    # print("Score dst:", save_path)
    import imageio
    # Process per sample
    def process(i):
        image_id, image = dataset.__getitem__(i)

        filename = os.path.join(logit_dir, image_id + ".npy")
        logit = np.load(filename)

        _, H, W = image.shape
        logit = torch.FloatTensor(logit)[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8).transpose(1, 2, 0)
        prob = postprocessor(image, prob)
        label = np.argmax(prob, axis=0)
        # for image_id, logit in zip(image_ids, logits):
        filename = os.path.join(save_dir, image_id + ".png")
        # imageio.imwrite(filename, label.astype(np.uint8))
        colorful(label.astype(np.uint8), filename)

    # CRF in multi-process
    joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(process)(i) for i in range(len(dataset))]
    )

def crf_eval_val(config_path, n_jobs=multiprocessing.cpu_count()):
    """
    CRF post-processing on pre-computed logits
    """

    # Configuration
    CONFIG = OmegaConf.load(config_path)
    torch.set_grad_enabled(False)
    print("# jobs:", n_jobs)

    # Dataset
    dataset = get_dataset('voc_test')(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.VAL,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=False,
    )
    print(dataset)

    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )

    # Path to logit files
    logit_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "features",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.VAL,
        "logit",
    )
    print("Logit src:", logit_dir)
    if not os.path.isdir(logit_dir):
        print("Logit not found, run first: python main.py test [OPTIONS]")
        quit()

    # Path to save scores
    save_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "features",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.VAL,
        "results/VOC2012/Segmentation/comp5_test_cls",
    )
    makedirs(save_dir)
    # save_path = os.path.join(save_dir, "scores_crf.json")
    # print("Score dst:", save_path)
    import imageio
    # Process per sample
    def process(i):
        image_id, image = dataset.__getitem__(i)

        filename = os.path.join(logit_dir, image_id + ".npy")
        logit = np.load(filename)

        _, H, W = image.shape
        logit = torch.FloatTensor(logit)[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8).transpose(1, 2, 0)
        prob = postprocessor(image, prob)
        label = np.argmax(prob, axis=0)
        # for image_id, logit in zip(image_ids, logits):
        filename = os.path.join(save_dir, image_id + ".png")
        # imageio.imwrite(filename, label.astype(np.uint8))
        colorful(label.astype(np.uint8), filename)

    # CRF in multi-process
    joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(process)(i) for i in range(len(dataset))]
    )

def colorful(out, name):
    arr = out.astype(np.uint8)
    im = Image.fromarray(arr)

    palette = []
    for i in range(256):
        palette.extend((i, i, i))
    palette[:3 * 21] = np.array([[0, 0, 0],
                                 [128, 0, 0],
                                 [0, 128, 0],
                                 [128, 128, 0],
                                 [0, 0, 128],
                                 [128, 0, 128],
                                 [0, 128, 128],
                                 [128, 128, 128],
                                 [64, 0, 0],
                                 [192, 0, 0],
                                 [64, 128, 0],
                                 [192, 128, 0],
                                 [64, 0, 128],
                                 [192, 0, 128],
                                 [64, 128, 128],
                                 [192, 128, 128],
                                 [0, 64, 0],
                                 [128, 64, 0],
                                 [0, 192, 0],
                                 [128, 192, 0],
                                 [0, 64, 128]
                                 ], dtype='uint8').flatten()

    im.putpalette(palette)
    im.save(name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)

    args = parser.parse_args()

    # Evaluation on test set
    evaluation(config_path=args.config_path, model_path=args.model_path)
    crf_eval(config_path=args.config_path)

    # visualization on val set with CRF
    # crf_eval_val(config_path=args.config_path,)