# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>
# modified by Sierkinhane <sierkinhane@163.com>

import sys
import matplotlib

matplotlib.use('Agg')
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import voc12.data
from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.evaluate_utils import *
from models.clip_resnet import CLIMS
from tools.ai.augment_utils import *
import torch.nn as nn
from tool import pyutils, imutils, torchutils
from clip_utils import *
from imutils import *
from clip_loss import *
import clip
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

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
parser.add_argument('--max_epoch', default=10, type=int)
parser.add_argument('--lr', default=0.00025, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)
#
parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--experiment', default='', type=str)
parser.add_argument('--augment', default='', type=str)
parser.add_argument('--enhance', default=False, type=bool)
parser.add_argument('--depth', default=50, type=int)
parser.add_argument('--hyper', default=[10, 10 * 2.5, 29.5, 1.15], type=list)



if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    args.architecture = f'resnet{args.depth}'

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

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    set_seed(args.seed)

    GLOBAL_SEED = 1
    GLOBAL_WORKER_ID = None
    def worker_init_fn(worker_id):
        global GLOBAL_WORKER_ID
        GLOBAL_WORKER_ID = worker_id
        set_seed(GLOBAL_SEED + worker_id)

    log_func = lambda string='': log_print(string, log_path)

    log_func('[i] {}'.format(args.experiment))
    log_func(
        'hyper-parameters alpha:{}, beta:{}, gamma:{}, delta:{}'.format(args.hyper[0], args.hyper[1], args.hyper[2],
                                                                        args.hyper[3]))
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
                              shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

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
    import clip
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    checkpoint = torch.load("experiments/models/finetune-clip-v2-test.pth")

    # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"]
    checkpoint["input_resolution"] = clip_model.input_resolution  # default is 224
    checkpoint["context_length"] = clip_model.context_length  # default is 77
    checkpoint["vocab_size"] = clip_model.vocab_size
    clip_model.load_state_dict(checkpoint)
    clip_model.eval()

    model = CLIMS(arch=args.architecture)
    ckpt = torch.load('experiments/models/resnet50-baseline-test.pth', map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
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

    OTMLoss = SimMaxLoss()
    BTMLoss = SimMinLoss()
    CBSLoss = BackgroundSuppressionLoss()

    log_func('[i] The number of scratched weights : {}'.format(len(param_groups[0])))
    log_func('[i] The number of fc weights : {}'.format(len(param_groups[1])))

    max_step = len(train_dataset) // args.batch_size * args.max_epoch
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wd, max_step=max_step)
    print(args.lr, optimizer.param_groups[0]['lr'])

    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train': [],
        'validation': []
    }

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(
        ['loss', 'L_OTM', 'L_BTM', 'L_CBS', 'L_REG'])

    # transform multi-hot label to class index label, a simple one can be found at our improved version of CLIMS
    def preprocess(labels):

        for n in range(labels.size(0)):
            if n == 0:
                fg_label = torch.zeros([1, labels.size(1)]).long()
                inverse_label = labels[n].unsqueeze(0)
            else:
                fg_label = torch.cat([fg_label, torch.zeros([1, labels.size(1)]).long()])
                inverse_label = torch.cat([inverse_label, labels[n].unsqueeze(0)])

            for idx in range(0, labels.size(1)):
                temp = torch.zeros([1, labels.size(1)]).long()
                retemp = torch.zeros(1, labels.size(1)).long()
                if labels[n, idx] == 1:
                    temp[0, idx] = 1
                    retemp.copy_(labels[n].view(1, labels.size(1)))
                    retemp[0, idx] = 0
                fg_label = torch.cat([fg_label, temp])
                inverse_label = torch.cat([inverse_label, retemp])

        return fg_label.cuda(), inverse_label.cuda()

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
                N = images.size(0)

                actmaps = model(images)

                labels = torch.cat([torch.zeros((labels.size(0), 1)).cuda(), labels], dim=1)

                # foreground indices (with channel 0)
                labels = torch.cat([torch.ones((labels.size(0), 1)).cuda(), labels[:, 1:]], dim=1)

                cams = F.interpolate(actmaps, (448, 448), mode='bilinear', align_corners=True)

                mask = labels.unsqueeze(2).unsqueeze(3)

                cams = (cams * mask)

                obj_cams = cams[:, 1:].max(dim=1)[0]
                # obj_cams = cams.max(dim=1)[0]
                for b in range(images.size(0)):
                    # fig, axes = plt.subplots(1, 1)
                    image = get_numpy_from_tensor(images[b])
                    cam = get_numpy_from_tensor(obj_cams[b])

                    image = denormalize(image, imagenet_mean, imagenet_std)#[..., ::-1]
                    h, w, c = image.shape

                    cam = (cam * 255).astype(np.uint8)
                    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
                    cam = colormap(cam)

                    image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)
                    cv2.imwrite('{}/test/{}-{}.png'.format(cam_path, step, b), image.astype(np.uint8))

                sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()

        print(' ')
        model.train()
        best_th = 0.0
        best_mIoU = 0.0
        mAP_score = 0
        return mAP_score, best_th, best_mIoU


    writer = SummaryWriter(tensorboard_dir)
    for epoch in range(args.max_epoch):
        for iteration, (_, images, labels) in enumerate(train_loader):

            images, labels = images.cuda(), labels.cuda()
            N = images.size(0)
            fg_label, inverse_label = preprocess(labels.cpu())
            #################################################################################################
            actmaps = model(images)
            _, _, h, w = actmaps.size()
            #################################################################################################
            # break
            # foreground indices
            labels = torch.cat([torch.zeros((labels.size(0), 1)).cuda(), labels], dim=1)
            fg_indices = torch.nonzero(labels.reshape(-1) == 1, as_tuple=False).squeeze()

            bg_indices = torch.tensor([i * labels.size(1) for i in range(labels.size(0))])

            # foreground indices (with channel 0)
            labels = torch.cat([torch.ones((labels.size(0), 1)).cuda(), labels[:, 1:]], dim=1)
            indices = torch.nonzero(labels.reshape(-1) == 1, as_tuple=False).squeeze()

            cam_224 = F.interpolate(actmaps, (224, 224), mode='bilinear', align_corners=True).reshape(N * 21, 1, 224,
                                                                                                      224)
            img_224 = F.interpolate(images, (224, 224), mode='bilinear', align_corners=True)

            fg_224_eval = []
            bg_224_eval = []
            temp_idx = torch.nonzero(labels[:, 1:] == 1, as_tuple=False)
            for j in range(temp_idx.shape[0]):
                fg_224_eval.append(cam_224[fg_indices[j]] * img_224[temp_idx[j, 0]])
                bg_224_eval.append((1 - cam_224[fg_indices[j]]) * img_224[temp_idx[j, 0]])
            fg_224_eval = torch.stack(fg_224_eval, dim=0)
            bg_224_eval = torch.stack(bg_224_eval, dim=0)

            weights = cal_weights(fg_label[fg_indices])

            L_OTM = OTMLoss(clip_forward(clip_model, fg_224_eval, fg_label[fg_indices]), weights)

            L_BTM = BTMLoss(clip_forward(clip_model, bg_224_eval, fg_label[fg_indices]), weights)

            L_CBS = CBSLoss(clip_model, fg_224_eval, fg_label[fg_indices])

            L_REG = torch.mean(actmaps[:, 1:, :, :])

            loss = args.hyper[0] * L_OTM + args.hyper[1] * L_BTM + args.hyper[2] * L_CBS + args.hyper[3] * L_REG

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_meter.add({
                'loss': loss.item(),
                'L_OTM': L_OTM.item(),
                'L_BTM': L_BTM.item(),
                'L_CBS': L_CBS.item(),
                'L_REG': L_REG.item(),
            })

            #################################################################################################
            # For Log
            #################################################################################################
            if (iteration + 1) % 50 == 0:
                # break
                mask = labels.unsqueeze(2).unsqueeze(3)
                cams = actmaps

                visual_debug(images, labels, cams, cam_path, iteration, num_classes=21, dataset='voc', phase='train')
                loss, L_OTM, L_BTM, L_CBS, L_REG = train_meter.get(clear=True)
                learning_rate = float(get_learning_rate_from_optimizer(optimizer))

                data = {
                    'epoch': epoch,
                    'max_epoch': args.max_epoch,
                    'iteration': iteration + 1,
                    'learning_rate': learning_rate,
                    'loss': loss,
                    'L_OTM': L_OTM,
                    'L_BTM': L_BTM,
                    'L_CBS': L_CBS,
                    'L_REG': L_REG,
                    'time': train_timer.tok(clear=True),
                }
                data_dic['train'].append(data)
                write_json(data_path, data_dic)

                log_func('[i] Epoch[{epoch:,}/{max_epoch:,}] '
                         'iteration={iteration:,}, '
                         'learning_rate={learning_rate:.4f}, '
                         'loss={loss:.4f}, '
                         'L_OTM={L_OTM:.4f}, '
                         'L_BTM={L_BTM:.4f}, '
                         'L_CBS={L_CBS:.4f}, '
                         'L_REG={L_REG:.4f}, '
                         'time={time:.0f}sec'.format(**data)
                         )

                writer.add_scalar('Train/loss', loss, iteration)
                writer.add_scalar('Train/learning_rate', learning_rate, iteration)

        #################################################################################################
        # Evaluation
        #################################################################################################
        if (epoch + 1) % 1 == 0:

            save_model_fn()
            log_func('[i] save model')

        # break
    write_json(data_path, data_dic)
    writer.close()

    print(args.experiment)
