# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import sys
import matplotlib

matplotlib.use('Agg')
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import voc12.data
from models.baseline import Baseline
from utils import *
from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from shutil import copyfile
import torch.nn as nn
import matplotlib.pyplot as plt
from tool import pyutils, imutils, torchutils

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
# parser.add_argument('--data_dir', default='/data2/xjheng/dataset/VOC2012/', type=str)
parser.add_argument('--data_dir', default='/data1/xjheng/dataset/VOC2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_epoch', default=15, type=int)

parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=448, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)
#
# parser.add_argument('--min_image_size', default=448, type=int)
# parser.add_argument('--max_image_size', default=768, type=int)

parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--experiment', default='', type=str)
parser.add_argument('--augment', default='', type=str)
parser.add_argument('--depth', default=50, type=int)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    log_dir = create_directory('./experiments/logs/')
    data_dir = create_directory('./experiments/data/')
    model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory('./experiments/tensorboards/{}/'.format(args.experiment))

    log_path = log_dir + '{}.txt'.format(args.experiment)
    data_path = data_dir + '{}.json'.format(args.experiment)
    model_path = model_dir + '{}.pth'.format(args.experiment)
    cam_path = 'images/{}'.format(args.experiment)
    create_directory(cam_path)
    create_directory(cam_path + '/train')
    create_directory(cam_path + '/test')

    # zipDir('.', cam_path + '/{}.zip'.format(args.experiment))

    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)

    log_func('[i] {}'.format(args.experiment))
    log_func()

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    normalize_fn = Normalize(clip_mean, clip_std)

    train_list = 'voc12/train_aug.txt'
    val_list = 'voc12/val.txt'
    train_dataset = voc12.data.VOC12ClsDataset(train_list, voc12_root=args.data_dir,
                                               transform=transforms.Compose([
                                                   # imutils.RandomResizeLong(448, 768),
                                                   imutils.RandomResizeLong(args.min_image_size, args.max_image_size),
                                                   transforms.RandomHorizontalFlip(),
                                                   # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                                   np.asarray,
                                                   normalize_fn,
                                                   imutils.RandomCrop(args.image_size),
                                                   imutils.HWC_to_CHW,
                                                   torch.from_numpy
                                               ]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_dataset = voc12.data.VOC12ClsDataset(val_list, voc12_root=args.data_dir,
                                               transform=transforms.Compose([
                                                   # imutils.RandomResizeLong(448, 768),
                                                   # transforms.RandomHorizontalFlip(),
                                                   # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                                   np.asarray,
                                                   normalize_fn,
                                                   imutils.CenterCrop(args.image_size),
                                                   imutils.HWC_to_CHW,
                                                   torch.from_numpy
                                               ]))

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers, pin_memory=False)

    log_func('[i] #train data {}'.format(len(train_dataset)))
    log_func('[i] #valid data {}'.format(len(val_dataset)))
    log_func()
    meta_dic = read_json('./data/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])


    ###################################################################################
    # Network
    ###################################################################################
    model = Baseline(arch=args.architecture)
    param_groups = model.get_parameter_groups()
    model_info(model)

    model = model.cuda()
    model.train()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM' % (calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)

    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    class_loss_fn = nn.MultiLabelSoftMarginLoss().cuda()

    log_func('[i] The number of scratched weights : {}'.format(len(param_groups[0])))
    log_func('[i] The number of fc weights : {}'.format(len(param_groups[1])))

    max_step = len(train_dataset) // args.batch_size * args.max_epoch
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wd, max_step=max_step)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.max_epoch)

    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train': [],
        'validation': []
    }

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss', 'class_loss'])

    def evaluate(loader):

        model.eval()
        eval_timer.tik()
        preds = []
        targets = []
        with torch.no_grad():
            length = len(loader)
            for step, (_, images, labels) in enumerate(loader):
                images = images.cuda()
                labels = labels.cuda()

                logits, actmaps = model(images, with_cams=True)

                preds.append(torch.sigmoid(logits).cpu().detach())
                targets.append(labels.cpu().detach())
                mask = labels.unsqueeze(2).unsqueeze(3)
                cams = (make_cam(actmaps) * mask)
                # cams = (actmaps * mask)
                # cams = torch.sigmoid(cams)
                obj_cams = cams.max(dim=1)[0]
                if step % 20 == 0:
                    for b in range(images.size(0)):
                        fig, axes = plt.subplots(1, 1)
                        image = get_numpy_from_tensor(images[b])
                        cam = get_numpy_from_tensor(obj_cams[b])

                        image = denormalize(image, imagenet_mean, imagenet_std)[..., ::-1]
                        h, w, c = image.shape

                        cam = (cam * 255).astype(np.uint8)
                        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
                        cam = colormap(cam)

                        image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)[..., ::-1]
                        image = image.astype(np.float32) / 255.
                        axes.imshow(image)
                        axes.axis('off')
                        # axes[1].imshow(gt_masks[b])
                        # axes[1].axis('off')
                        plt.savefig('{}/test/{}-{}.png'.format(cam_path, step, b), bbox_inches='tight')
                        plt.close()
                        writer.add_image('CAM/{}'.format(b + 1), image, iteration, dataformats='HWC')

                sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()

        print(' ')
        model.train()
        best_th = 0.0
        best_mIoU = 0.0
        mAP_score = compute_mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
        # mAP_score = 0
        return mAP_score, best_th, best_mIoU

    writer = SummaryWriter(tensorboard_dir)
    for epoch in range(args.max_epoch):
        preds = []
        targets = []
        for iteration, (_, images, labels) in enumerate(train_loader):

            images, labels = images.cuda(), labels.cuda()
            N = images.size(0)

            #################################################################################################
            logits = model(images)

            #################################################################################################

            # compute loss
            class_loss = class_loss_fn(logits, labels)
            loss = class_loss

            preds.append(torch.sigmoid(logits).cpu().detach())
            targets.append(labels.cpu().detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_meter.add({
                'loss': loss.item(),
                'class_loss': class_loss.item(),
            })

            #################################################################################################
            # For Log
            #################################################################################################
            if (iteration + 1) % 50 == 0:
                loss, class_loss = train_meter.get(clear=True)
                learning_rate = float(get_learning_rate_from_optimizer(optimizer))

                data = {
                    'epoch': epoch,
                    'max_epoch': args.max_epoch,
                    'iteration': iteration + 1,
                    'learning_rate': learning_rate,
                    'loss': loss,
                    'class_loss': class_loss,
                    'time': train_timer.tok(clear=True),
                }
                data_dic['train'].append(data)
                write_json(data_path, data_dic)

                log_func('[i]\t'
                         'Epoch[{epoch:,}/{max_epoch:,}],\t'
                         'iteration={iteration:,}, \t'
                         'learning_rate={learning_rate:.4f}, \t'
                         'loss={loss:.4f}, \t'
                         'class_loss={class_loss:.4f}, \t'
                         'time={time:.0f}sec'.format(**data)
                         )

                writer.add_scalar('Train/loss', loss, iteration)
                writer.add_scalar('Train/class_loss', class_loss, iteration)
                writer.add_scalar('Train/area_loss', class_loss, iteration)
                writer.add_scalar('Train/learning_rate', learning_rate, iteration)

        mAP_score = compute_mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
        print("[train] mAP score: {}".format(mAP_score))
        #################################################################################################
        # Evaluation
        #################################################################################################
        if (epoch) % 1 == 0:
            mAP, threshold, mIoU = evaluate(val_loader)

            save_model_fn()
            log_func('[i] save model')

            data = {
                'epoch': epoch,
                'max_epoch': args.max_epoch,
                'train_mAP': mAP,
                'time': eval_timer.tok(clear=True),
            }
            data_dic['validation'].append(data)
            write_json(data_path, data_dic)

            log_func('[i]\t'
                     'Epoch[{epoch:,}/{max_epoch:,}],\t'
                     'train_mAP={train_mAP:.2f}%,\t'
                     'time={time:.0f}sec'.format(**data)
                     )
    write_json(data_path, data_dic)
    writer.close()

    print(args.experiment)
