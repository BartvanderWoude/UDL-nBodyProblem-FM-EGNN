import torch

from tqdm import tqdm
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath


def train(vf, dataloader, loss_fn, nepochs, lr):
    path = AffineProbPath(scheduler=CondOTScheduler())
    optim = torch.optim.Adam(vf.parameters(), lr=lr)

    for epoch in range(nepochs):
        total_loss = 0.0
        for data in tqdm(dataloader):
            x_0, vel_0, x_1, vel_1 = data

            optim.zero_grad()

            t = torch.rand(x_0.shape[0])

            coors_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            vel_sample = path.sample(t=t, x_0=vel_0, x_1=vel_1)
            t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, x_0.shape[1], 1)

            pred_t, pred_x, pred_vel = vf(t=t, coors=coors_sample.x_t, vel=vel_sample.x_t)
            loss = loss_fn(pred_x, coors_sample.dx_t) + loss_fn(pred_vel, vel_sample.dx_t)

            loss.backward()
            optim.step()

            total_loss += loss.item()

        print(f"Epoch: {epoch}, Loss: {total_loss / len(dataloader)}")

    return vf
