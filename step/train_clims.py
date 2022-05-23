import cv2
import os
import torch
import os.path as osp
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib
from imutils import visual_debug
from clip_utils import clip_forward
from clip_loss import SimMaxLoss, SimMinLoss, BackgroundSuppressionLoss
import voc12.dataloader
from misc import pyutils, torchutils
import os, math


def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss': loss.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss')))

    return

# GLOBAL_SEED = 2
# import numpy as np
# import random
# def set_seed(seed):
#     print('11')
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
# GLOBAL_WORKER_ID = None
# def worker_init_fn(worker_id):
#     global GLOBAL_WORKER_ID
#     GLOBAL_WORKER_ID = worker_id
#     set_seed(GLOBAL_SEED + worker_id)

def run(args):
    model = getattr(importlib.import_module(args.clims_network), 'CLIMS')(n_classes=20)

    # initialize backbone network with baseline CAM
    model.load_state_dict(torch.load('cam-baseline-voc12/res50_cam.pth'), strict=True)
    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.clims_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.clims_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * args.clims_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.clims_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    # Loss
    hyper = [float(h) for h in args.hyper.split(',')]
    OTMLoss = SimMaxLoss()
    BTMLoss = SimMinLoss()
    CBSLoss = BackgroundSuppressionLoss(dname='voc')
    print(hyper)

    # CLIP
    import clip
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(args.clip, device=device)
    # for p in clip_model.parameters():
    #     p.requires_grad = False
    clip_model.eval()

    if args.clip == 'RN50x4':
        clip_input_size = 288
    else:
        clip_input_size = 224

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    # transform multi-hot label to class index label
    def preprocess(labels):
        new_labels = []
        for n in range(labels.size(0)):
            for idx in range(0, labels.size(1)):
                temp = torch.zeros(1, labels.size(1)).long()
                if labels[n, idx] == 1:
                    temp[0, idx] = 1
                new_labels.append(temp)
        return torch.cat(new_labels, dim=0).cuda()

    hyper = [float(h) for h in args.hyper.split(',')]
    for ep in range(args.clims_num_epoches):

        print('Epoch %d/%d' % (ep + 1, args.clims_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            img = img.cuda()
            label = pack['label'].cuda(non_blocking=True)

            fg_label = preprocess(label.cpu())

            x = model(img)
            N, _, _, _ = x.size()
            optimizer.zero_grad()

            # foreground indices
            fg_indices = torch.nonzero(label.reshape(-1) == 1, as_tuple=False).squeeze()

            cam_224 = F.interpolate(x, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True).reshape(N * 20, 1, clip_input_size,
                                                                                                clip_input_size)
            img_224 = F.interpolate(img, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)

            fg_224_eval = []
            bg_224_eval = []
            temp_idx = torch.nonzero(label == 1, as_tuple=False)
            for j in range(temp_idx.shape[0]):
                fg_224_eval.append(cam_224[fg_indices[j]] * img_224[temp_idx[j, 0]])
                bg_224_eval.append((1 - cam_224[fg_indices[j]]) * img_224[temp_idx[j, 0]])

            fg_224_eval = torch.stack(fg_224_eval, dim=0)
            bg_224_eval = torch.stack(bg_224_eval, dim=0)

            L_OTM = OTMLoss(clip_forward(clip_model, fg_224_eval, fg_label[fg_indices], dname='voc'), 1)

            L_BTM = BTMLoss(clip_forward(clip_model, bg_224_eval, fg_label[fg_indices], dname='voc'), 1)

            L_CBS = CBSLoss(clip_model, fg_224_eval)

            L_REG = torch.mean(x)

            loss = hyper[0] * L_OTM + hyper[1] * L_BTM + hyper[2] * L_CBS + hyper[3] * L_REG

            loss.backward()
            optimizer.step()

            avg_meter.add({'loss1': loss.item(), 'L_OTM': L_OTM.item(), 'L_BTM': L_BTM.item(), 'L_CBS': L_CBS.item(),
                           'L_REG': L_REG.item()})

            if (optimizer.global_step - 1) % 200 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'L_OTM:%.4f' % (avg_meter.pop('L_OTM')),
                      'L_BTM:%.4f' % (avg_meter.pop('L_BTM')),
                      'L_CBS:%.4f' % (avg_meter.pop('L_CBS')),
                      'L_REG:%.4f' % (avg_meter.pop('L_REG')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

                # visualize class activation maps during training if needed.
                # visual_debug(img, label, x, 'vis/clims_v2_voc12_cam_vis', optimizer.global_step, num_classes=21,
                #             dataset='coco', phase='train')

        # validate(model, val_data_loader)
        timer.reset_stage()

    torch.save(model.module.state_dict(), args.clims_weights_name + '.pth')
    torch.cuda.empty_cache()
