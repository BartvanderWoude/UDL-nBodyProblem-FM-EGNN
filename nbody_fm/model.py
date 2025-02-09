import torch

from egnn_pytorch import EGNN
from flow_matching.utils import ModelWrapper


class EGNN_network(torch.nn.Module):
    def __init__(self):
        super(EGNN_network, self).__init__()

        self.layer1 = EGNN(dim=1, update_vel=True)
        self.layer2 = EGNN(dim=1, update_vel=True)
        self.layer3 = EGNN(dim=1, update_vel=True)

    def forward(self, t, coors, vel):
        t, coors, vel = self.layer1(feats=t, coors=coors, vel=vel)
        t, coors, vel = self.layer2(feats=t, coors=coors, vel=vel)
        t, coors, vel = self.layer3(feats=t, coors=coors, vel=vel)
        return t, coors, vel


class CoorsWrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        vel = extras["vel"]
        t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, x.shape[1], 1)
        pred_t, pred_x, pred_vel = self.model(t=t, coors=x, vel=vel)

        return pred_x


class VelWrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        coors = extras["coors"]
        t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, x.shape[1], 1)
        pred_t, pred_x, pred_vel = self.model(t=t, coors=coors, vel=x)
        return pred_vel
