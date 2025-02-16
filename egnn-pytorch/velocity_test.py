import torch
from egnn_pytorch.egnn_pytorch import EGNN

model = EGNN(dim=512, edge_dim=1, update_vel=True)

feats = torch.randn(1, 16, 512)
coors = torch.randn(1, 16, 3, requires_grad=True) * 10.0
edges = torch.randn(1, 16, 16, 1)
vel = torch.randn(1, 16, 3, requires_grad=True) * 5.0

coors.retain_grad()
out_feats, out_coors, out_vel = model(feats, coors, edges, vel=vel)
print('out_feats', out_feats)
print('out_coors', out_coors)
print('out_vel', vel, out_vel)


