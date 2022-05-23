import torch
import clip
from clip_utils import background_dict, category_dict

class SimMaxLoss(torch.nn.Module):

    def __init__(self, margin=0):
        super(SimMaxLoss, self).__init__()
        self.margin = margin

    def forward(self, x, weights):
        return -(torch.log(x + self.margin) * weights).mean()

class SimMinLoss(torch.nn.Module):

    def __init__(self, margin=0):
        super(SimMinLoss, self).__init__()
        self.margin = margin

    def forward(self, x, weights):
        return -(torch.log(1 - x + self.margin) * weights).mean()

class BackgroundSuppressionLoss(torch.nn.Module):

    def __init__(self, margin=0):
        super(BackgroundSuppressionLoss, self).__init__()
        self.margin = margin

    def forward(self, clip_model, images, labels):

        bg_dict = background_dict['bg']
        bg_dict_keys = bg_dict.keys()
        image_features = clip_model.encode_image(images)
        loss = torch.tensor(.0, requires_grad=True, device='cuda:0')
        for i in range(labels.size(0)):
            idx = torch.nonzero(labels[i], as_tuple=False).squeeze()
            cate = category_dict['voc'][idx]
            if cate not in bg_dict_keys:
                continue
            text = bg_dict[cate]
            text = clip.tokenize(text).cuda()

            text_features = clip_model.encode_text(text)

            image_feature = image_features[i].reshape(1, -1)
            # normalized features
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # image_features = image_features.reshape(N, C)
            text_features = text_features.permute(1, 0)

            x = torch.matmul(image_feature, text_features)
            loss = loss + (-(torch.log(1 - x + self.margin)).mean())

        return loss