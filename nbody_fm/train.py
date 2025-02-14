import torch

from tqdm import tqdm
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath


def train(vf, traindataloader, valdataloader, loss_fn, nepochs, lr, beta, loss_file_name, val_file_name):
    path = AffineProbPath(scheduler=CondOTScheduler())
    optim = torch.optim.Adam(vf.parameters(), lr=lr)

    loss_file = open("losses/" + loss_file_name, "w", encoding="utf-8")
    val_file = open("losses/" + val_file_name, "w", encoding="utf-8")

    for epoch in range(nepochs):
        total_loss = 0.0
        pos_loss = 0.0
        vel_loss = 0.0

        for data in tqdm(traindataloader):
            x_0, vel_0, x_1, vel_1 = data

            optim.zero_grad()

            t = torch.rand(x_0.shape[0])

            coors_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            vel_sample = path.sample(t=t, x_0=vel_0, x_1=vel_1)
            t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, x_0.shape[1], 1)

            pred_x, pred_vel = vf(t=t, coors=coors_sample.x_t, vel=vel_sample.x_t)
            loss_pos = loss_fn(pred_x, coors_sample.dx_t)
            loss_vel = loss_fn(pred_vel, vel_sample.dx_t)
            loss = beta * loss_pos + (1 - beta) * loss_vel

            loss.backward()
            optim.step()

            pos_loss += loss_pos.item()
            vel_loss += loss_vel.item()
            total_loss += loss.item()

        loss_file.write(
            f"{epoch},{pos_loss / len(traindataloader)},{vel_loss / len(traindataloader)},{total_loss / len(traindataloader)}\n")
        print(f"Epoch: {epoch}, Pos Loss: {pos_loss / len(traindataloader)}, Vel Loss: {vel_loss / len(traindataloader)}, Total loss: {total_loss / len(traindataloader)}")

        total_loss = 0.0
        pos_loss = 0.0
        vel_loss = 0.0

        with torch.no_grad():
            for data in valdataloader:
                x_0, vel_0, x_1, vel_1 = data

                t = torch.rand(x_0.shape[0])

                coors_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
                vel_sample = path.sample(t=t, x_0=vel_0, x_1=vel_1)
                t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, x_0.shape[1], 1)

                pred_x, pred_vel = vf(t=t, coors=coors_sample.x_t, vel=vel_sample.x_t)
                loss_pos = loss_fn(pred_x, coors_sample.dx_t)
                loss_vel = loss_fn(pred_vel, vel_sample.dx_t)
                loss = beta * loss_pos + (1 - beta) * loss_vel

                pos_loss += loss_pos.item()
                vel_loss += loss_vel.item()
                total_loss += loss.item()

        val_file.write(
            f"{epoch},{pos_loss / len(valdataloader)},{vel_loss / len(valdataloader)},{total_loss / len(valdataloader)}\n")
        print(
            f"Validation epoch: {epoch}, Pos Loss: {pos_loss / len(valdataloader)}, Vel Loss: {vel_loss / len(valdataloader)}, Total loss: {total_loss / len(valdataloader)}")

    loss_file.close()
    val_file.close()

    return vf
