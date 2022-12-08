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
    'coco': ['a photo of street sign.', 'a photo of mountain.', 'a photo of video game.', 'a photo of men.',
             'a photo of track.', 'a photo of bus stop.', 'a photo of cabinet.', 'a photo of tray.',
             'a photo of plate.', 'a photo of shirt.', 'a photo of city street.', 'a photo of runway.',
             'a photo of tower.', 'a photo of ramp.', 'a photo of grass.', 'a photo of pillow.',
             'a photo of urinal.', 'a photo of lake.', 'a photo of brick.', 'a photo of fence.',
             'a photo of shower.', 'a photo of airport.', 'a photo of animal.', 'a photo of shower curtain.',
             'a photo of road.', 'a photo of mirror.', 'a photo of jacket.', 'a photo of church.', 'a photo of snow.',
             'a photo of fruit.', 'a photo of hay.', 'a photo of floor.', 'a photo of field.', 'a photo of street.',
             'a photo of mouth.', 'a photo of steam engine.', 'a photo of cheese.', 'a photo of river.',
             'a photo of tree branch.', 'a photo of suit.', 'a photo of child.', 'a photo of soup.', 'a photo of desk.',
             'a photo of tub.', 'a photo of tennis court.', 'a photo of teeth.', 'a photo of bridge.',
             'a photo of sky.', 'a photo of officer.', 'a photo of sidewalk.', 'a photo of dock.',
             'a photo of tree.', 'a photo of court.', 'a photo of rock.', 'a photo of board.',
             'a photo of branch.', 'a photo of pan.', 'a photo of box.', 'a photo of body.',
             'a photo of salad.', 'a photo of dirt.', 'a photo of leaf.', 'a photo of hand.',
             'a photo of highway.', 'a photo of vegetable.', 'a photo of computer monitor.',
             'a photo of door.', 'a photo of meat.', 'a photo of pair.', 'a photo of beach.',
             'a photo of harbor.', 'a photo of ocean.', 'a photo of baseball player.', 'a photo of girl.',
             'a photo of market.', 'a photo of window.', 'a photo of blanket.', 'a photo of boy.', 'a photo of woman.',
             'a photo of bat.', 'a photo of baby.', 'a photo of flower.', 'a photo of wall.', 'a photo of bath tub.',
             'a photo of tarmac.', 'a photo of tennis ball.', 'a photo of roll.', 'a photo of park.'],
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
