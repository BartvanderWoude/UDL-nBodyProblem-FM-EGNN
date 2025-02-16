import torch

from egnn_pytorch import EGNN
from flow_matching.utils import ModelWrapper


class EGNN_network(torch.nn.Module):
    def __init__(self, number_of_layers=2, use_time_embedding=False, feature_dim=3, number_of_nodes=3):
        super(EGNN_network, self).__init__()
        self.number_of_layers = number_of_layers
        self.use_time_embedding = use_time_embedding
        self.time_dim = 1
        self.feature_dim = feature_dim

        self.base_edges = torch.ones(1, number_of_nodes, number_of_nodes, 1)
        self.edges = None

        if self.use_time_embedding:
            self.time_embedding = torch.nn.Sequential(
                torch.nn.Linear(self.time_dim, 2 * feature_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(2 * feature_dim, feature_dim)
            )
        GNN_layers = [EGNN_layer(feature_dim=feature_dim) for _ in range(number_of_layers)]
        self.layers = torch.nn.ModuleList(GNN_layers)

        # and this
        self.conv1 = torch.nn.Conv1d(4, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 4, 1)

    def forward(self, t, coors, vel):
        if self.use_time_embedding:
            t = self.time_embedding(t)
        elif self.time_dim == 1:
            t = t.repeat(1, 1, self.feature_dim)
        elif self.time_dim > 1:
            raise ValueError("Cannot handle time_dim > 1 without time_embedding")

        if self.edges is None or coors.shape[0] != self.edges.shape[0]:
            self.edges = self.base_edges.repeat(coors.shape[0], 1, 1, 1)

        for layer in self.layers:
            _, coors, vel = layer(feats=t, coors=coors, vel=vel, edges=self.edges)

        # make this an optional thing with argument probably

        coors = torch.swapaxes(coors, 1, 2)
        vel = torch.swapaxes(vel, 1, 2)

        out = torch.concat([coors, vel], dim=1)
        out = torch.nn.SELU()(out)
        out = self.conv1(out)
        out = torch.nn.SELU()(out)
        out = self.conv2(out)

        coors, vel = out[:, 0:2, :], out[:, 2:4, :]

        coors = torch.swapaxes(coors, 1, 2)
        vel = torch.swapaxes(vel, 1, 2)

        ###
        return coors, vel


class EGNN_layer(torch.nn.Module):
    def __init__(self, feature_dim=3, edge_dim=1):
        super(EGNN_layer, self).__init__()
        self.egnn = EGNN(dim=feature_dim, edge_dim=edge_dim, update_vel=True)

        self.activation = torch.nn.SELU()

        # self.linear_coors = torch.nn.Linear(2, 2)
        # self.linear_vel = torch.nn.Linear(2, 2)

    def forward(self, feats, coors, vel, edges):
        t, coors, vel = self.egnn(feats=feats, coors=coors, vel=vel, edges=edges)

        coors = self.activation(coors)
        vel = self.activation(vel)

        # coors = self.linear_coors(coors)
        # vel = self.linear_vel(vel)

        return t, coors, vel


class CoorsWrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        vel = extras["vel"]
        t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, x.shape[1], self.model.time_dim)
        pred_x, _ = self.model(t=t, coors=x, vel=vel)

        return pred_x


class VelWrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        coors = extras["coors"]
        t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, x.shape[1], self.model.time_dim)
        _, pred_vel = self.model(t=t, coors=coors, vel=x)
        return pred_vel


class SolverWrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, x.shape[1], self.model.time_dim)
        pred_x, pred_vel = self.model(t=t, coors=x[0], vel=x[1])
        return torch.vstack((pred_x, pred_vel))
