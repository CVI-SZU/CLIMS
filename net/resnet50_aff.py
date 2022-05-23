import torch
import torch.nn as nn
import torch.nn.functional as F
from net import resnet50
"""
affinity net
"""

class AffinityNet(nn.Module):

    def __init__(self):
        super(AffinityNet, self).__init__()

        # backbone
        self.resnet50 = resnet50.resnet50(pretrained=True, strides=[2, 2, 2, 1])

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage2 = nn.Sequential(self.resnet50.layer1)
        self.stage3 = nn.Sequential(self.resnet50.layer2)
        self.stage4 = nn.Sequential(self.resnet50.layer3)
        self.stage5 = nn.Sequential(self.resnet50.layer4)

        # branch: class boundary detection
        self.fc_edge1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge5 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        self.edge_layers = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6])

    def forward(self, x):
        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3).detach()
        x5 = self.stage5(x4).detach()

        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
        edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]
        edge_out = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))

        dp1 = self.fc_dp1(x1)
        dp2 = self.fc_dp2(x2)
        dp3 = self.fc_dp3(x3)
        dp4 = self.fc_dp4(x4)[..., :dp3.size(2), :dp3.size(3)]
        dp5 = self.fc_dp5(x5)[..., :dp3.size(2), :dp3.size(3)]

        dp_up3 = self.fc_dp6(torch.cat([dp3, dp4, dp5], dim=1))[..., :dp2.size(2), :dp2.size(3)]
        dp_out = self.fc_dp7(torch.cat([dp1, dp2, dp_up3], dim=1))

        return edge_out, dp_out

    def trainable_parameters(self):
        return (tuple(self.edge_layers.parameters()),
                tuple(self.dp_layers.parameters()))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()

    def get_edge(self, x, image_size=512, stride=4):
        feat_size = (x.size(2) - 1) // stride + 1, (x.size(3) - 1) // stride + 1

        x = F.pad(x, [0, image_size - x.size(3), 0, image_size - x.size(2)])
        edge_out = self.forward(x)
        edge_out = edge_out[..., :feat_size[0], :feat_size[1]]
        edge_out = torch.sigmoid(edge_out[0] / 2 + edge_out[1].flip(-1) / 2)

        return edge_out

    """
    aff = self.to_affinity(torch.sigmoid(edge_out))
    pos_aff_loss = (-1) * torch.log(aff + 1e-5)
    neg_aff_loss = (-1) * torch.log(1. + 1e-5 - aff)
    """

    def to_affinity(self, edge):
        aff_list = []
        edge = edge.view(edge.size(0), -1)

        for i in range(self.n_path_lengths):
            ind = self._buffers["path_indices_" + str(i)]
            ind_flat = ind.view(-1)
            dist = torch.index_select(edge, dim=-1, index=ind_flat)
            dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
            aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
            aff_list.append(aff)
        aff_cat = torch.cat(aff_list, dim=1)
        return aff_cat