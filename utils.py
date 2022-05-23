import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torch_utils import *
from demo_utils import *
from torchvision.utils import save_image
from torchvision import utils

CAT20_VOC2012 = ['aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse',
                 'motorbike', 'person', 'pottedplant',
                 'sheep', 'sofa', 'train',
                 'tvmonitor']

CAT21_VOC2012 = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse',
                 'motorbike', 'person', 'pottedplant',
                 'sheep', 'sofa', 'train',
                 'tvmonitor']

def convert_mxnet_to_torch(filename):
    import mxnet

    save_dict = mxnet.nd.load(filename)

    renamed_dict = dict()

    bn_param_mx_pt = {'beta': 'bias', 'gamma': 'weight', 'mean': 'running_mean', 'var': 'running_var'}

    for k, v in save_dict.items():

        v = torch.from_numpy(v.asnumpy())
        toks = k.split('_')

        if 'conv1a' in toks[0]:
            renamed_dict['conv1a.weight'] = v

        elif 'linear1000' in toks[0]:
            pass

        elif 'branch' in toks[1]:

            pt_name = []

            if toks[0][-1] != 'a':
                pt_name.append('b' + toks[0][-3] + '_' + toks[0][-1])
            else:
                pt_name.append('b' + toks[0][-2])

            if 'res' in toks[0]:
                layer_type = 'conv'
                last_name = 'weight'

            else:  # 'bn' in toks[0]:
                layer_type = 'bn'
                last_name = bn_param_mx_pt[toks[-1]]

            pt_name.append(layer_type + '_' + toks[1])

            pt_name.append(last_name)

            torch_name = '.'.join(pt_name)
            renamed_dict[torch_name] = v

        else:
            last_name = bn_param_mx_pt[toks[-1]]
            renamed_dict['bn7.' + last_name] = v

    return renamed_dict

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def compute_mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


def activation_map_visualization(step, images, cams, labels, save_path, phase='train', gt_masks=None, cam_actmaps=None):
    for n in range(labels.size(0)):
        image = get_numpy_from_tensor(images[n])
        image = denormalize(image, imagenet_mean, imagenet_std)  # [..., ::-1]
        h, w, c = image.shape
        cam = cams[n, labels[n].view(21) == 1, :, :]
        label = torch.nonzero(labels[n].view(21) == 1, as_tuple=False).squeeze()
        # print(label.size())
        if gt_masks is not None:
            fig, axes = plt.subplots(2, cam.size(0) + 1 + 2)
        else:
            fig, axes = plt.subplots(2, cam.size(0) + 1)
        fig.tight_layout()  # 调整整体空白
        plt.subplots_adjust(wspace=0, hspace=0)  # 调整子图间距
        axes[0, 0].imshow(image)
        axes[0, 0].axis('off')
        # axes[0, 0].set_title(pack[0][batch_index])
        axes[1, 0].axis('off')

        for j in range(label.size(0)):
            cam_ = get_numpy_from_tensor(cam[j])
            #cam_ = (cam_ * 255).astype(np.uint8)
            #cam_ = cv2.resize(cam_, (w, h), interpolation=cv2.INTER_LINEAR)
            #cam_ = colormap(cam_)

            #cam_ = cam_ + 0.4 * image
            #cam_ = cam_ / np.max(cam_)
            #cam_ = np.uint8(cam_ * 255)

            axes[0, j + 1].imshow(cam_)
            axes[0, j + 1].set_title(VOC2012_CAT_21[label[j]])
            axes[0, j + 1].axis('off')
            axes[1, j + 1].hist(
                cam[j].detach().cpu().numpy().ravel(), bins=50, range=(0, 1),
                color='pink', edgecolor='k', density=True)
            # axes[1, j + 1].set_title(VOC2012_CAT[label[j]])
            # axes[1, j + 1].axis('off')

        if cam_actmaps is not None:
            axes[0, label.size(0) + 1].imshow(gt_masks[n])
            cam = get_numpy_from_tensor(cam_actmaps[n])
            cam = (cam * 255).astype(np.uint8)
            cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
            cam = colormap(cam)

            image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)[..., ::-1]
            image = image.astype(np.float32) / 255.

            axes[0, label.size(0) + 2].imshow(image)
            axes[0, label.size(0) + 2].axis('off')

        plt.savefig("{}/{}/step_{}_{}.png".format(save_path, phase, step, n), bbox_inches='tight')
        plt.close()

def visualization(step, images, cams, labels, save_path, phase='train', gt_masks=None, cam_actmaps=None):
    # print(labels.size())
    for n in range(labels.size(0)):
        image = get_numpy_from_tensor(images[n])
        image = denormalize(image, imagenet_mean, imagenet_std)  # [..., ::-1]
        h, w, c = image.shape
        cam = cams[n, labels[n].view(20) == 1, :, :]
        # cam_act = cam_actmaps[n, labels[n].view(20) == 1, :, :]
        label = torch.nonzero(labels[n].view(20) == 1, as_tuple=False).squeeze(1)
        # print(label.size())
        if cam_actmaps is not None:
            fig, axes = plt.subplots(3, cam.size(0) + 1)
        else:
            fig, axes = plt.subplots(2, cam.size(0) + 1)
        fig.tight_layout()  # 调整整体空白
        plt.subplots_adjust(wspace=0, hspace=0)  # 调整子图间距
        axes[0, 0].imshow(image)
        axes[0, 0].axis('off')
        # axes[0, 0].set_title(pack[0][batch_index])
        axes[1, 0].axis('off')
        # axes[2, 0].axis('off')
        for j in range(label.size(0)):
            cam_ = get_numpy_from_tensor(cam[j])
            #cam_ = (cam_ * 255).astype(np.uint8)
            #cam_ = cv2.resize(cam_, (w, h), interpolation=cv2.INTER_LINEAR)
            #cam_ = colormap(cam_)

            #cam_ = cam_ + 0.4 * image
            #cam_ = cam_ / np.max(cam_)
            #cam_ = np.uint8(cam_ * 255)

            axes[0, j + 1].imshow(cam_)
            axes[0, j + 1].set_title(VOC2012_CAT[label[j]])
            axes[0, j + 1].axis('off')
            axes[1, j + 1].hist(
                cam[j].detach().cpu().numpy().ravel(), bins=50, range=(0, 1),
                color='pink', edgecolor='k', density=True)
            # axes[1, j + 1].set_title(VOC2012_CAT[label[j]])
            # axes[1, j + 1].axis('off')
            # axes[2, j + 1].axis('off')
            if cam_actmaps is not None:
                cam_ = get_numpy_from_tensor(cam_act[j])
                # print(cam_.shape)
                cam_ = (cam_ * 255).astype(np.uint8)
                cam_ = cv2.resize(cam_, (w, h), interpolation=cv2.INTER_LINEAR)
                cam_ = colormap(cam_)

                image_ = cv2.addWeighted(image, 0.5, cam_, 0.5, 0)[..., ::-1]
                image_ = image_.astype(np.float32) / 255.

                axes[2, j + 1].imshow(image_)
                axes[2, j + 1].axis('off')
            # axes[1, label.size(0) + 2].axis('off')
            # if gt_masks is not None:
            #     axes[0, label.size(0) + 2].imshow(gt_masks[n])

        plt.savefig("{}/{}/step_{}_{}.png".format(save_path, phase, step, n), bbox_inches='tight')
        plt.close()

def visualization21(step, images, cams, labels, save_path, phase='train', gt_masks=None, cam_actmaps=None):
    # print(labels.size())
    for n in range(labels.size(0)):
        image = get_numpy_from_tensor(images[n])
        image = denormalize(image, imagenet_mean, imagenet_std)  # [..., ::-1]
        h, w, c = image.shape
        cam = cams[n, labels[n].view(21) == 1, :, :]
        # cam_act = cam_actmaps[n, labels[n].view(21) == 1, :, :]
        label = torch.nonzero(labels[n].view(21) == 1, as_tuple=False).squeeze(1)
        # print(label.size())
        if cam_actmaps is not None:
            fig, axes = plt.subplots(3, cam.size(0) + 1)
        else:
            fig, axes = plt.subplots(2, cam.size(0) + 1)
        fig.tight_layout()  # 调整整体空白
        plt.subplots_adjust(wspace=0, hspace=0)  # 调整子图间距
        axes[0, 0].imshow(image)
        axes[0, 0].axis('off')
        # axes[0, 0].set_title(pack[0][batch_index])
        axes[1, 0].axis('off')
        # axes[2, 0].axis('off')
        for j in range(label.size(0)):
            cam_ = get_numpy_from_tensor(cam[j])
            #cam_ = (cam_ * 255).astype(np.uint8)
            #cam_ = cv2.resize(cam_, (w, h), interpolation=cv2.INTER_LINEAR)
            #cam_ = colormap(cam_)

            #cam_ = cam_ + 0.4 * image
            #cam_ = cam_ / np.max(cam_)
            #cam_ = np.uint8(cam_ * 255)

            axes[0, j + 1].imshow(cam_)
            axes[0, j + 1].set_title(VOC2012_CAT_21[label[j]])
            axes[0, j + 1].axis('off')
            axes[1, j + 1].hist(
                cam[j].detach().cpu().numpy().ravel(), bins=50, range=(0, 1),
                color='pink', edgecolor='k', density=True)
            # axes[1, j + 1].set_title(VOC2012_CAT[label[j]])
            # axes[1, j + 1].axis('off')
            # axes[2, j + 1].axis('off')
            if cam_actmaps is not None:
                cam_ = get_numpy_from_tensor(cam_act[j])
                # print(cam_.shape)
                cam_ = (cam_ * 255).astype(np.uint8)
                cam_ = cv2.resize(cam_, (w, h), interpolation=cv2.INTER_LINEAR)
                cam_ = colormap(cam_)

                image_ = cv2.addWeighted(image, 0.5, cam_, 0.5, 0)[..., ::-1]
                image_ = image_.astype(np.float32) / 255.

                axes[2, j + 1].imshow(image_)
                axes[2, j + 1].axis('off')
            # axes[1, label.size(0) + 2].axis('off')
            # if gt_masks is not None:
            #     axes[0, label.size(0) + 2].imshow(gt_masks[n])

        plt.savefig("{}/{}/step_{}_{}.png".format(save_path, phase, step, n), bbox_inches='tight')
        plt.close()

def single_activation_map_visualization(step, images, cams, labels, save_path, phase='train', gt_masks=None, cam_actmaps=None):

    for n in range(labels.size(0)):
        image = get_numpy_from_tensor(images[n])
        image = denormalize(image, imagenet_mean, imagenet_std)  # [..., ::-1]
        h, w, c = image.shape
        # cam = cams[n, labels[n].view(20) == 1, :, :]
        # label = torch.nonzero(labels[n].view(21) == 1, as_tuple=False).squeeze()
        if cam_actmaps is None:
            fig, axes = plt.subplots(1, 2, figsize=(15, 15))
        else:
            fig, axes = plt.subplots(1, 4, figsize=(15, 15))
        fig.tight_layout()  # 调整整体空白
        plt.subplots_adjust(wspace=0, hspace=0)  # 调整子图间距
        axes[0].imshow(image)
        axes[0].axis('off')
        # axes[0, 0].set_title(pack[0][batch_index])
        # axes[1].axis('off')

        # for j in range(label.size(0)):
        cam = get_numpy_from_tensor(cams[n, 0])
        # cam = (cam * 255).astype(np.uint8)
        # cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        # cam = colormap(cam)

        # cam = cam + 0.4 * image
        # cam = cam / np.max(cam)
        # cam = np.uint8(cam * 255)

        axes[1].imshow(cam)
        # axes[0, j + 1].set_title()
        axes[1].axis('off')

        # axes[2].hist(
        #     cams[n, 0].detach().cpu().numpy().ravel(), bins=50, range=(0, 1),
        #     color='pink', edgecolor='k', density=True)
        # axes[1, j + 1].set_title(VOC2012_CAT[label[j]])
        # axes[1, j + 1].axis('off')

        # print(cam_actmaps.shape)

        if cam_actmaps is not None:
            axes[2].imshow(gt_masks[n])
            cam = get_numpy_from_tensor(cam_actmaps[n])
            cam = (cam * 255).astype(np.uint8)
            cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
            cam = colormap(cam)

            image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)[..., ::-1]
            image = image.astype(np.float32) / 255.

            axes[3].imshow(image)
            axes[3].axis('off')

        plt.savefig("{}/{}/step_{}_{}.png".format(save_path, phase, step, n), bbox_inches='tight')
        plt.close()

def model_info(model, log_func=print):
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    log_func('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        log_func('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    log_func('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))

def accuracy(output, target, topk=(1,)):
    """Computes the acc@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def visualization_debug(experiments, images, attmaps, epoch, cnt, phase='train', bboxes=None, gt_bboxes=None, seg=True):
    n = int(np.sqrt(images.shape[0]))
    utils.save_image(images, 'images/{}/{}/epoch-{}-{}-pri.jpg'.format(experiments, phase, epoch, cnt), nrow=n,
                     normalize=True)
    _, c, h, w = images.shape
    fig, axes = plt.subplots(n, n, figsize=(21, 21))
    for j in range(axes.shape[0]):
        for k in range(axes.shape[1]):
            temp = attmaps[j * n + k, 0, :, :]
            temp = temp.cpu().detach().numpy()
            if temp.shape[0] != h:
                temp = cv2.resize(temp, (w, h), interpolation=cv2.INTER_CUBIC)
            axes[j, k].imshow(temp)
            axes[j, k].axis('off')
    plt.savefig('images/{}/{}/epoch-{}-{}-att.jpg'.format(experiments, phase, epoch, cnt))
    plt.close()

    fig, axes = plt.subplots(n, n, figsize=(21, 21))
    for j in range(axes.shape[0]):
        for k in range(axes.shape[1]):
            temp = attmaps[j * n + k, 0, :, :]
            temp = temp.cpu().detach().numpy()
            axes[j, k].hist(temp.ravel(), bins=50, range=(0, 1), color='cornflowerblue')
            axes[j, k].set_xlabel('Intensity')
            axes[j, k].set_ylabel('Density')
            # axes[j, k].axis('off')
    plt.savefig('images/{}/{}/epoch-{}-{}-hist.jpg'.format(experiments, phase, epoch, cnt))
    plt.close()

    attmaps = attmaps.squeeze().to('cpu').detach().numpy()

    for i in range(images.shape[0]):
        attmap = attmaps[i]
        attmap = attmap / np.max(attmap)
        attmap = np.uint8(attmap * 255)
        colormap = cv2.applyColorMap(cv2.resize(attmap, (w, h)), cv2.COLORMAP_JET)

        grid = utils.make_grid(images[i].unsqueeze(0), nrow=1, padding=0, pad_value=0,
                         normalize=True, range=None)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        image = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()[..., ::-1]
        # print(image.shape, colormap.shape)
        cam = colormap + 0.4 * image
        cam = cam / np.max(cam)
        cam = np.uint8(cam * 255).copy()

        if phase == 'test' and bboxes is not None:
            box = bboxes[i][0]
            cv2.rectangle(cam, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)  # BGR

            if isinstance(gt_bboxes, list):
                for j in range(gt_bboxes[i].shape[0]):
                    gtbox = gt_bboxes[i][j]
                    cv2.rectangle(cam, (int(gtbox[1]), int(gtbox[2])), (int(gtbox[3]), int(gtbox[4])), (255, 0, 0), 2)
            else:
                gtbox = gt_bboxes[i]
                cv2.rectangle(cam, (int(gtbox[1]), int(gtbox[2])), (int(gtbox[3]), int(gtbox[4])), (255, 0, 0), 2)
        # print('debug/images/{}/{}/colormaps/epoch-{}-{}-colormap.jpg'.format(experiments, phase, epoch, cnt))
        cv2.imwrite('images/{}/{}/colormaps/epoch-{}-{}-image.jpg'.format(experiments, phase, cnt, i), image)
        cv2.imwrite('images/{}/{}/colormaps/epoch-{}-{}-colormap.jpg'.format(experiments, phase, cnt, i), cam)

        if seg:
            h, w, c = image.shape
            attmap = cv2.resize(attmap, (h, w), interpolation=cv2.INTER_LINEAR)
            # print(attmap.max(), attmap.min())
            mask = np.where(attmap > 128, 1, 0).reshape(h, w, 1)
            temp = np.uint8(image*mask)
            cv2.imwrite(
                'debug/images/{}/{}/colormaps/{}-{}-seg.jpg'.format(experiments, phase, cnt, epoch), temp)