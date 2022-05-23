import torch

category_dict = {
    'voc': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
            'dog',
            'horse', 'motorbike', 'player', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor'],
    'coco': ['player', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
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
    'voc': ['a photo of tree.', 'a photo of river.',
            'a photo of sea.', 'a photo of lake.', 'a photo of water.',
            'a photo of railway.', 'a photo of railroad.', 'a photo of track.',
            'a photo of stone.', 'a photo of rocks.'],
    'coco': ['a photo of tree.', 'a photo of river.',
             'a photo of sea.', 'a photo of lake.', 'a photo of water.',
             'a photo of railway.', 'a photo of railroad.', 'a photo of track.',
             'a photo of stone.', 'a photo of rocks.', 'a photo of playground.', 'a photo of spray.'],
}

prompt_dict = ['a photo of {}.']


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
            text.append(prompt_dict[cnt].format(_d[idx[0]]))
        elif cnt == 1:
            text.append(prompt_dict[cnt].format(_d[idx[0]], _d[idx[1]]))
        elif cnt == 2:
            text.append(prompt_dict[cnt].format(_d[idx[0]], _d[idx[1]], _d[idx[2]]))
        elif cnt == 3:
            text.append(prompt_dict[cnt].format(_d[idx[0]], _d[idx[1]], _d[idx[2]], _d[idx[3]]))
        elif cnt == 4:
            text.append(prompt_dict[cnt].format(_d[idx[0]], _d[idx[1]], _d[idx[2]], _d[idx[3]], _d[idx[4]]))
        else:
            raise NotImplementedError
    return text


import clip
def clip_forward(clip_model, images, labels, dname='coco'):
    texts = to_text(labels, dname)
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
