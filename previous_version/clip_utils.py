import torch

category_dict = {
    'voc': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
    'coco': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
             'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
             'teddy bear', 'hair drier', 'toothbrush']
}

background_dict = {
    'fg': ['a photo of aeroplane', 'a photo of bicycle', 'a photo of bird', 'a photo of boat', 'a photo of bottle',
           'a photo of bus', 'a photo of car', 'a photo of cat', 'a photo of chair', 'a photo of cow',
           'a photo of diningtable', 'a photo of dog', 'a photo of horse', 'a photo of motorbike', 'a photo of person',
           'a photo of pottedplant', 'a photo of sheep', 'a photo of sofa', 'a photo of train', 'a photo of tvmonitor'],
    'bg': {'boat': ['a photo of river', 'a photo of water', 'a photo of lake', 'a photo of sea', 'a photo of building'],
            'train': ['a photo of railroad', 'a photo of railway', 'a photo of branches', 'a photo of tree'],
            },
}


template = ['a photo of {}', 'a photo of {} and {}', 'a photo of {}, {}, and {}', 'a photo of {}, {}, {}, and {}',
            'a photo of {}, {}, {}, {}, and {}', 'a photo of {}, {}, {}, {}, {} and {}']


def to_text(labels, dataset='voc'):

    _d = category_dict[dataset]

    text = []
    for i in range(labels.size(0)):
        idx = torch.nonzero(labels[i], as_tuple=False).squeeze()
        if torch.sum(labels[i]) == 1:
            idx = idx.unsqueeze(0)
        cnt = idx.shape[0] - 1
        if cnt == -1:
            text.append('background')
        elif cnt == 0:
            text.append(template[cnt].format(_d[idx[0]]))
        elif cnt == 1:
            text.append(template[cnt].format(_d[idx[0]], _d[idx[1]]))
        elif cnt == 2:
            text.append(template[cnt].format(_d[idx[0]], _d[idx[1]], _d[idx[2]]))
        elif cnt == 3:
            text.append(template[cnt].format(_d[idx[0]], _d[idx[1]], _d[idx[2]], _d[idx[3]]))
        elif cnt == 4:
            text.append(template[cnt].format(_d[idx[0]], _d[idx[1]], _d[idx[2]], _d[idx[3]], _d[idx[4]]))
        elif cnt == 5:
            text.append(template[cnt].format(_d[idx[0]], _d[idx[1]], _d[idx[2]], _d[idx[3]], _d[idx[4]], _d[idx[5]]))
        else:
            raise NotImplementedError
    return text

import clip
def clip_forward(clip_model, images, labels):

    texts = to_text(labels, 'voc')
    texts = clip.tokenize(texts).cuda()

    image_features = clip_model.encode_image(images)
    text_features = clip_model.encode_text(texts)

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    N, C = image_features.size()
    image_features = image_features.reshape(N, 1, C)
    text_features = text_features.reshape(N, C, 1)

    similarity = torch.matmul(image_features, text_features)

    return similarity

valid = ['a photo of sofa', 'a photo of diningtable', 'a photo of chair']
valid_for_cel = ['a photo of sofa', 'a photo of diningtable', 'a photo of chair', 'a photo of bird', 'a photo of boat']
def cal_weights(labels):
    text = to_text(labels, 'voc')
    weights = torch.ones(labels.size(0), 1, 1).cuda()
    for i in range(len(text)):
        if text[i] in valid:
            weights[i] = 3
    return weights