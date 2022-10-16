# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>
# modified by Sierkinhane <sierkinhane@gmail.com>

import matplotlib

matplotlib.use('Agg')
from torchvision import transforms
from clip_utils import to_text
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import voc12.data
import clip
from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
import torch.nn as nn
from tool import imutils

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
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
parser.add_argument('--max_epoch', default=20, type=int)

parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--wd', default=1e-3, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=448, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)
#
parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--experiment', default='', type=str)
parser.add_argument('--augment', default='', type=str)



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

    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)

    log_func('[i] {}'.format(args.experiment))
    log_func()

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    # imagenet_mean = [0.485, 0.456, 0.406]
    # imagenet_std = [0.229, 0.224, 0.225]

    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    normalize_fn = Normalize(clip_mean, clip_std)

    train_list = 'voc12/train_aug.txt'
    val_list = 'voc12/val.txt'
    train_dataset = voc12.data.VOC12ClsDataset(train_list, voc12_root=args.data_dir,
                                               transform=transforms.Compose([
                                                   imutils.RandomResizeLong(args.min_image_size, args.max_image_size),
                                                   transforms.RandomHorizontalFlip(),
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

    def convert_models_to_fp32(model):
        for p in model.parameters():
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    clip.model.convert_weights(model)

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

    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train': [],
        'validation': []
    }

    train_timer = Timer()
    eval_timer = Timer()

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.max_epoch)

    train_meter = Average_Meter(['loss', 'loss_i', 'loss_t'])
    val_meter = Average_Meter(['loss', 'loss_i', 'loss_t'])

    def evaluate(loader):

        model.eval()
        eval_timer.tik()
        with torch.no_grad():
            length = len(loader)
            for step, (_, images, labels) in enumerate(loader):
                images = images.cuda()
                labels = labels.cuda()
                images = F.interpolate(images, (224, 224), mode='bilinear', align_corners=True)
                texts = to_text(labels, 'voc')
                texts = clip.tokenize(texts).cuda()
                logits_per_image, logits_per_text = model(images, texts)
                ground_truth = torch.arange(images.size(0)).cuda()
                loss_i = loss_img(logits_per_image, ground_truth)
                loss_t = loss_txt(logits_per_text, ground_truth)
                loss = loss_i + loss_t
                val_meter.add({
                    'loss': loss.item(),
                    'loss_i': loss_i.item(),
                    'loss_t': loss_t.item(),
                })

                if step % 20 == 0:
                    loss, loss_i, loss_t = val_meter.get(clear=True)
                    learning_rate = float(get_learning_rate_from_optimizer(optimizer))
                    data = {
                        'epoch': epoch,
                        'max_epoch': args.max_epoch,
                        'iteration': step + 1,
                        'learning_rate': learning_rate,
                        'loss': loss,
                        'loss_i': loss_i,
                        'loss_t': loss_t,
                        'time': train_timer.tok(clear=True),
                    }
                    # data_dic['validation'].append(data)
                    # write_json(data_path, data_dic)

                    log_func('[i]\t'
                             '[Test] Epoch[{epoch:,}/{max_epoch:,}],\t'
                             'iteration={iteration:,}, \t'
                             'learning_rate={learning_rate:.4f}, \t'
                             'loss={loss:.4f}, \t'
                             'loss_i={loss_i:.4f}, \t'
                             'loss_t={loss_t:.4f}, \t'
                             'time={time:.0f}sec'.format(**data)
                             )

        print(' ')
        model.train()
        best_th = 0.0
        best_mIoU = 0.0
        # mAP_score = compute_mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
        mAP_score = 0
        return mAP_score, best_th, best_mIoU


    writer = SummaryWriter(tensorboard_dir)
    for epoch in range(args.max_epoch):
        preds = []
        targets = []
        model.train()
        for iteration, (_, images, labels) in enumerate(train_loader):

            images, labels = images.cuda(), labels.cuda()
            N = images.size(0)
            images = F.interpolate(images, (224, 224), mode='bilinear', align_corners=True)

            #################################################################################################
            texts = to_text(labels, 'voc')
            texts = clip.tokenize(texts).cuda()
            logits_per_image, logits_per_text = model(images, texts)

            #################################################################################################

            # compute loss
            ground_truth = torch.arange(N).cuda()
            loss_i = loss_img(logits_per_image, ground_truth)
            loss_t = loss_txt(logits_per_text, ground_truth)
            loss = loss_i + loss_t
            optimizer.zero_grad()
            loss.backward()
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
            scheduler.step()

            train_meter.add({
                'loss': loss.item(),
                'loss_i': loss_i.item(),
                'loss_t': loss_t.item(),
            })

            #################################################################################################
            # For Log
            #################################################################################################
            if (iteration + 1) % 50 == 0:

                loss, loss_i, loss_t = train_meter.get(clear=True)
                learning_rate = float(get_learning_rate_from_optimizer(optimizer))

                data = {
                    'epoch': epoch,
                    'max_epoch': args.max_epoch,
                    'iteration': iteration + 1,
                    'learning_rate': learning_rate,
                    'loss': loss,
                    'loss_i': loss_i,
                    'loss_t': loss_t,
                    'time': train_timer.tok(clear=True),
                }
                data_dic['train'].append(data)
                write_json(data_path, data_dic)

                log_func('[i]\t'
                         'Epoch[{epoch:,}/{max_epoch:,}],\t'
                         'iteration={iteration:,}, \t'
                         'learning_rate={learning_rate:.6f}, \t'
                         'loss={loss:.4f}, \t'
                         'loss_i={loss_i:.4f}, \t'
                         'loss_t={loss_t:.4f}, \t'
                         'time={time:.0f}sec'.format(**data)
                         )

                writer.add_scalar('Train/loss', loss, iteration)
                writer.add_scalar('Train/learning_rate', learning_rate, iteration)

        #################################################################################################
        # Evaluation
        #################################################################################################
        if epoch % 1 == 0:
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
