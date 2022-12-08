import torch
import clip
from clip_utils import background_dict, category_dict

# maximize similarity
class SimMaxLoss(torch.nn.Module):

    def __init__(self, margin=0):
        super(SimMaxLoss, self).__init__()
        self.margin = margin

    def forward(self, x, weights):
        x = x.clamp(0.0001, 0.9999)
        return -(torch.log(x + self.margin) * weights).mean()

# minimize similarity
class SimMinLoss(torch.nn.Module):

    def __init__(self, margin=0):
        super(SimMinLoss, self).__init__()
        self.margin = margin

    def forward(self, x, weights):
        x = x.clamp(0.0001, 0.9999)
        return -(torch.log(1 - x + self.margin) * weights).mean()

# suppress background activation
class BackgroundSuppressionLoss(torch.nn.Module):
    """
    based on threshold
    """

    def __init__(self, threshold=0.26, dname='coco'):
        super(BackgroundSuppressionLoss, self).__init__()
        self.dname = dname
        self.background = background_dict[dname]
        self.threshold = threshold
        print(f'Use CBSLoss! threshold: {threshold}')

    def forward(self, clip_model, images, eps=0.0001):
        image_features = clip_model.encode_image(images)  # [N1, C]
        text_features = clip_model.encode_text(clip.tokenize(self.background).cuda())  # [N2, C]

        # normalization
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits_per_image = (image_features @ text_features.t())  # [N1, N2]
        mask = torch.zeros_like(logits_per_image)
        mask = torch.where(logits_per_image > self.threshold, torch.ones_like(mask), torch.zeros_like(mask))

        return -(torch.log(1 - logits_per_image) * mask).sum()

