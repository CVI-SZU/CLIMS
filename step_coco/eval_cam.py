import numpy as np
import os.path as osp
import mscoco.dataloader
from torch.utils.data import DataLoader
from chainercv.evaluations import calc_semantic_segmentation_confusion
from clip_utils import category_dict


def print_iou(iou, dname='coco'):
    iou_dict = {}
    for i in range(len(iou) - 1):
        iou_dict[category_dict[dname][i]] = iou[i + 1]
    print(iou_dict)
    return iou_dict

def run(args):
    dataset = mscoco.dataloader.COCOSegmentationDataset(image_dir=osp.join(args.mscoco_root, 'train2014/'),
                                                        anno_path=osp.join(args.mscoco_root,
                                                                           'annotations/instances_train2014.json'),
                                                        masks_path=osp.join(args.mscoco_root, 'mask/train2014'),
                                                        crop_size=512)
    preds = []
    labels = []
    n_images = 0
    num = len(dataset)
    for i, pack in enumerate(dataset):
        if i % 1000 == 0:
            print(i, '/', num)
        # if i == 9999:
        #     break
        filename = pack['name'].split('.')[0]

        n_images += 1
        cam_dict = np.load(osp.join(args.cam_out_dir, filename + '.npy'), allow_pickle=True).item()

        cams = cam_dict['high_res']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels].astype(np.uint8)
        preds.append(cls_labels.copy())

        label = dataset.get_label_by_name(filename)
        labels.append(label)

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    # print(iou)
    iou_dict = print_iou(iou, 'coco')

    print("threshold:", args.cam_eval_thres, 'miou:', np.nanmean(iou), "i_imgs", n_images)
    print('among_predfg_bg', float((resj[1:].sum() - confusion[1:, 1:].sum()) / (resj[1:].sum())))

    hyper = [float(h) for h in args.hyper.split(',')]
    name = args.clims_weights_name + f'{hyper[0]}_{hyper[1]}_{hyper[2]}_{hyper[3]})_ep({args.clims_num_epoches})_lr({args.clims_learning_rate}).pth'
    with open(args.work_space + '/eval_result.txt', 'a') as file:
        file.write(name + f' {args.clip} th: {args.cam_eval_thres}, mIoU: {np.nanmean(iou)} {iou_dict} \n')

    return np.nanmean(iou)
