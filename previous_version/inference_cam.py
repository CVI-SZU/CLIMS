# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import sys
import copy

from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *

from tools.ai.demo_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *

from models.clip_resnet import CLIMS

palette = np.array([[0, 0, 0],
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
                    [0, 64, 128]])


parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='/home/lzr/data/VOC/VOC2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--domain', default='train', type=str)
parser.add_argument('--vis_dir', default='vis_cam', type=str)

parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    from iputils import get_host_ip

    ip = get_host_ip()
    if ip == '172.31.234.159':
        args.data_dir = '/data1/xjheng/dataset/VOC2012/'
    elif ip == '172.31.111.180':
        args.data_dir = '/home/lzr/data/VOC/VOC2012/'
    else:
        raise NotImplementedError

    experiment_name = args.tag

    if 'train' in args.domain:
        experiment_name += '@train'
    else:
        experiment_name += '@val'

    experiment_name += '@scale=%s'%args.scales
    
    pred_dir = create_directory(f'./experiments/predictions/{experiment_name}/')

    cam_path = create_directory(f'{args.vis_dir}/{experiment_name}')

    model_path = './experiments/models/' + f'{args.tag}.pth'
    print(model_path)

    set_seed(args.seed)
    log_func = lambda string='': print(string)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    # imagenet_mean = [0.485, 0.456, 0.406]
    # imagenet_std = [0.229, 0.224, 0.225]
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    normalize_fn = Normalize(clip_mean, clip_std)
    
    # for mIoU
    meta_dic = read_json('./data/VOC_2012.json')
    dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)
    
    ###################################################################################
    # Network
    ###################################################################################
    model = CLIMS(arch=args.architecture)

    model = model.cuda()
    model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    load_model(model, model_path, parallel=the_number_of_gpu > 1)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]
    
    model.eval()
    eval_timer.tik()

    def get_cam(ori_image, scale):
        # preprocessing
        image = copy.deepcopy(ori_image)
        image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)
        
        image = normalize_fn(image)
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image)
        flipped_image = image.flip(-1)
        
        images = torch.stack([image, flipped_image])
        images = images.cuda()
        
        # inferenece
        features = model(images, inference=True)[:, 1:]

        # postprocessing
        cams = F.relu(features)
        # cams = torch.sigmoid(features)
        cams = cams[0] + cams[1].flip(-1)

        return cams

    vis_cam = True
    with torch.no_grad():
        length = len(dataset)
        for step, (ori_image, image_id, label, gt_mask) in enumerate(dataset):
            ori_w, ori_h = ori_image.size

            npy_path = pred_dir + image_id + '.npy'

            
            strided_size = get_strided_size((ori_h, ori_w), 4)
            strided_up_size = get_strided_up_size((ori_h, ori_w), 16)

            cams_list = [get_cam(ori_image, scale) for scale in scales]

            strided_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_size)[0] for cams in cams_list]
            strided_cams = torch.sum(torch.stack(strided_cams_list), dim=0)
            
            hr_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_up_size)[0] for cams in cams_list]
            hr_cams = torch.sum(torch.stack(hr_cams_list), dim=0)[:, :ori_h, :ori_w]
            
            keys = torch.nonzero(torch.from_numpy(label))[:, 0]
            
            strided_cams = strided_cams[keys]
            strided_cams /= F.adaptive_max_pool2d(strided_cams, (1, 1)) + 1e-5
            
            hr_cams = hr_cams[keys]
            hr_cams /= F.adaptive_max_pool2d(hr_cams, (1, 1)) + 1e-5

            # print(ori_image.size, hr_cams.shape, strided_cams.shape)
            # print(hr_cams.max(), hr_cams.min())
            if vis_cam:

                cam = torch.sum(hr_cams, dim=0)
                cam = cam.unsqueeze(0).unsqueeze(0)

                cam = make_cam(cam).squeeze()
                cam = get_numpy_from_tensor(cam)

                image = np.array(ori_image)

                h, w, c = image.shape

                cam = (cam * 255).astype(np.uint8)
                cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
                cam = colormap(cam)

                image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)
                cv2.imwrite(f'{cam_path}/{image_id}.png', image.astype(np.uint8))
            if os.path.isfile(npy_path):
                continue
            # save cams
            keys = np.pad(keys + 1, (1, 0), mode='constant')
            np.save(npy_path, {"keys": keys, "cam": strided_cams.cpu(), "hr_cam": hr_cams.cpu().numpy()})
            
            sys.stdout.write('\r# Make CAM [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), hr_cams.size()))
            sys.stdout.flush()
        print()
    
    if args.domain == 'train_aug':
        args.domain = 'train'
    
    print("python3 evaluate.py --experiment_name {} --domain {}".format(experiment_name, args.domain))