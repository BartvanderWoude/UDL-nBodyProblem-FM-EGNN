import torch

from egnn_pytorch import EGNN
from flow_matching.utils import ModelWrapper


class EGNN_network(torch.nn.Module):
    def __init__(self, number_of_layers=3, use_time_embedding=False, feature_dim=3):
        super(EGNN_network, self).__init__()
        self.number_of_layers = number_of_layers
        self.use_time_embedding = use_time_embedding
        self.time_dim = 1
        self.feature_dim = feature_dim

        if self.use_time_embedding:
            self.time_embedding = torch.nn.Sequential(
                torch.nn.Linear(self.time_dim, 2*feature_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2*feature_dim, feature_dim)
            )
        GNN_layers = [EGNN_layer(feature_dim=feature_dim) for _ in range(number_of_layers)]
        self.layers = torch.nn.ModuleList(GNN_layers)

    def forward(self, t, coors, vel):
        if self.use_time_embedding:
            t = self.time_embedding(t)
        elif self.time_dim == 1:
            t = t.repeat(1, 1, self.feature_dim)
        elif self.time_dim > 1:
            raise ValueError("Cannot handle time_dim > 1 without time_embedding")

        for layer in self.layers:
            _, coors, vel = layer(feats=t, coors=coors, vel=vel)

        return coors, vel


class EGNN_layer(torch.nn.Module):
    def __init__(self, feature_dim=3):
        super(EGNN_layer, self).__init__()
        self.egnn = EGNN(dim=feature_dim, update_vel=True)
        self.activation = torch.nn.ReLU()

    def forward(self, feats, coors, vel):
        t, coors, vel = self.egnn(feats=feats, coors=coors, vel=vel)
        # coors = self.activation(coors)
        # vel = self.activation(vel)
        return t, coors, vel


class CoorsWrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        vel = extras["vel"]
        t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, x.shape[1], self.model.time_dim)
        pred_x, pred_vel = self.model(t=t, coors=x, vel=vel)

        return pred_x


class VelWrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        coors = extras["coors"]
        t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, x.shape[1], self.model.time_dim)
        pred_x, pred_vel = self.model(t=t, coors=coors, vel=x)
        return pred_vel
