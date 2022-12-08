import cv2
import os
import torch
import os.path as osp
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import distributed

import importlib
from imutils import visual_debug
from clip_utils import clip_forward
from clip_loss import SimMaxLoss, SimMinLoss, BackgroundSuppressionLoss
import mscoco.dataloader
from misc import pyutils, torchutils
import os, math

def reduce_mean(tensor, nprocs):
    # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt


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


def run(args):
    model = getattr(importlib.import_module(args.clims_network), 'CLIMS')(n_classes=80)
    
    if (not args.use_distributed_train) or \
            (args.use_distributed_train and args.local_rank == 0):
        model.load_state_dict(torch.load('cam-baseline-coco/res50_cam.pth'), strict=True)
        print('model loaded')
    train_dataset = mscoco.dataloader.COCOClassificationDataset(
        image_dir=osp.join(args.mscoco_root, 'train2014/'),
        anno_path=osp.join(args.mscoco_root, 'annotations/instances_train2014.json'),
        labels_path='./mscoco/train_labels.npy',
        resize_long=(320, 640), hor_flip=True, crop_size=512, crop_method="random")
    
    # for multi-GPU
    if args.use_distributed_train:
        # Distributed Train prepare
        print('Using GPU num:', torch.cuda.device_count())
        distributed.init_process_group(backend="nccl")
        print('Torch Distributed world_size', torch.distributed.get_world_size())
        torch.cuda.set_device(args.local_rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                       shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        max_step = (len(train_dataset) // args.cam_batch_size // torch.cuda.device_count()) * args.clims_num_epoches
    else:
        train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        max_step = (len(train_dataset) // args.cam_batch_size) * args.clims_num_epoches
    
    
    # val_dataset = mscoco.dataloader.COCOClassificationDataset(
    #     image_dir=osp.join(args.mscoco_root, 'val2014/'),
    #     anno_path=osp.join(args.mscoco_root, 'annotations/instances_val2014.json'),
    #     labels_path='./mscoco/val_labels.npy', crop_size=512)
    # val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size, shuffle=False,
    #                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.clims_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * args.clims_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.clims_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    if args.use_distributed_train:
        model = model.cuda(args.local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model).cuda()

    model.train()

    # CLIP
    import clip
    if args.use_distributed_train:
        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load(args.clip, device=device)
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    if args.clip == 'RN50x4':
        clip_input_size = 288
    else:
        clip_input_size = 224

    # Loss
    hyper = [float(h) for h in args.hyper.split(',')]
    OTMLoss = SimMaxLoss()
    BTMLoss = SimMinLoss()
    CBSLoss = BackgroundSuppressionLoss(threshold=args.cbs_loss_thresh, dname='coco')
    print('clims on coco')
    print(hyper)


    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    def preprocess(labels):
        new_labels = []
        for n in range(labels.size(0)):
            for idx in range(0, labels.size(1)):
                temp = torch.zeros(1, labels.size(1)).long()
                if labels[n, idx] == 1:
                    temp[0, idx] = 1
                new_labels.append(temp)
        return torch.cat(new_labels, dim=0).cuda()

    for ep in range(args.clims_num_epoches):

        print('Epoch %d/%d' % (ep + 1, args.clims_num_epoches))
        if args.use_distributed_train:
            train_data_loader.sampler.set_epoch(ep)

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

            cam_224 = F.interpolate(x, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True).reshape(N * 80, 1, clip_input_size,
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

            L_OTM = OTMLoss(clip_forward(clip_model, fg_224_eval, fg_label[fg_indices], dname='coco'), 1)

            L_BTM = BTMLoss(clip_forward(clip_model, bg_224_eval, fg_label[fg_indices], dname='coco'), 1)

            L_CBS = CBSLoss(clip_model, fg_224_eval)

            L_REG = torch.mean(x)

            loss = hyper[0] * L_OTM + hyper[1] * L_BTM + hyper[2] * L_CBS + hyper[3] * L_REG

            loss.backward()
            optimizer.step()

            if args.use_distributed_train:
                loss = reduce_mean(loss, distributed.get_world_size())
                if args.local_rank != 0:
                    continue

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

                # visual_debug(img, label, x, 'vis/clims_coco_cam_vis', optimizer.global_step, num_classes=81,
                #             dataset='coco', phase='train')

        # validate(model, val_data_loader)
        timer.reset_stage()

    # torch.save(model.module.state_dict(),
    #            args.clims_weights_name + f'{hyper[0]}_{hyper[1]}_{hyper[2]}_{hyper[3]}_K({hyper[4]})_ep({args.clims_num_epoches})_lr({args.clims_learning_rate}).pth')
    # torch.save(model.module.state_dict(), args.clims_weights_name + '.pth')
    # torch.cuda.empty_cache()

    if args.use_distributed_train:
        distributed.barrier()
        if args.local_rank == 0:
            torch.save(model.module.state_dict(), args.clims_weights_name + '.pth')
        torch.cuda.empty_cache()
        distributed.destroy_process_group()
    else:
        torch.save(model.module.state_dict(), args.clims_weights_name + '.pth')
        torch.cuda.empty_cache()
