import torch
from nbody_fm.model import CoorsWrappedModel, VelWrappedModel

from tqdm import tqdm
from flow_matching.solver import ODESolver


def infer(vf, dataset, inference_method, inference_steps, output_file, loss_file, loss_fn, step_size=0.01, look_ahead=1, max_dist=10.0):
    if inference_method in ["dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun"]:
        step_size = None

    coors_model = CoorsWrappedModel(model=vf)
    vel_model = VelWrappedModel(model=vf)

    coors_solver = ODESolver(velocity_model=coors_model)
    vel_solver = ODESolver(velocity_model=vel_model)

    inferred = open("infer/" + output_file, "w", encoding="utf-8")
    loss_file = open("infer/" + loss_file, "w", encoding="utf-8")
    idx = 0

    for i in tqdm(range(min(inference_steps, len(dataset)))):
        GTx_0, GTvel_0, GTx_1, GTvel_1 = dataset[i]
        GTx_0 = GTx_0.unsqueeze(0)
        GTvel_0 = GTvel_0.unsqueeze(0)
        GTx_1 = GTx_1.unsqueeze(0)
        GTvel_1 = GTvel_1.unsqueeze(0)

        if i % look_ahead == 0:
            x_0 = GTx_0
            vel_0 = GTvel_0

        out_x = coors_solver.sample(x_init=x_0, step_size=step_size, method=inference_method, vel=vel_0)
        out_vel = vel_solver.sample(x_init=vel_0, step_size=step_size, method=inference_method, coors=x_0)

        pos_loss = loss_fn(out_x, GTx_1)
        vel_loss = loss_fn(out_vel, GTvel_1)
        loss = pos_loss + vel_loss

        loss_file.write(f"{i},{pos_loss.item()},{vel_loss.item()},{loss.item()}\n")

        x_0 = out_x
        vel_0 = out_vel

        output = f"{x_0[0, 0, 0].item()},{x_0[0, 0, 1].item()},"
        output += f"{x_0[0, 1, 0].item()},{x_0[0, 1, 1].item()},"
        output += f"{x_0[0, 2, 0].item()},{x_0[0, 2, 1].item()}\n"
        inferred.write(output)

        # Allow for possible early stopping if there is no reset using look_ahead
        if inference_steps == look_ahead:
            # Get coordinates of individual nodes
            x_node_0 = x_0[0, 0, :]
            x_node_1 = x_0[0, 1, :]
            x_node_2 = x_0[0, 2, :]

            # Average coordinates and velocity
            x_avg = (x_node_0 + x_node_1 + x_node_2) / 3
            # vel_avg = (vel_0[0, 0, :] + vel_0[0, 1, :] + vel_0[0, 2, :]) / 3

            # Magnitude of the vector between the nodes
            dist_01 = torch.norm(x_node_0 - x_node_1)
            dist_02 = torch.norm(x_node_0 - x_node_2)
            dist_12 = torch.norm(x_node_1 - x_node_2)

            # Distance origin and average position/ velocity
            dist_avg = torch.norm(x_avg)
            # dist_vel = torch.norm(vel_avg)

            if dist_01 > max_dist or dist_02 > max_dist or dist_12 > max_dist or dist_avg > max_dist:
                print(f"Unstable configuration at step {i}")
                break

        idx += 1


def infer_unified(vf, dataset, inference_method, inference_steps, output_file, step_size=0.01, look_ahead=1):
    if inference_method in ["dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun"]:
        step_size = None

    solver = ODESolver(velocity_model=vf)

    inferred = open("infer/" + output_file, "w", encoding="utf-8")
    idx = 0

    for i in tqdm(range(inference_steps)):
        if i % look_ahead == 0:
            x_0, vel_0, _, _ = dataset[i]
            x_0 = x_0.unsqueeze(0)
            vel_0 = vel_0.unsqueeze(0)

        out = solver.sample(x_init=torch.vstack((x_0, vel_0)), step_size=step_size, method=inference_method)

        x_0 = out[0]
        vel_0 = out[1]

        output = f"{x_0[0, 0, 0].item()},{x_0[0, 0, 1].item()},"
        output += f"{x_0[0, 1, 0].item()},{x_0[0, 1, 1].item()},"
        output += f"{x_0[0, 2, 0].item()},{x_0[0, 2, 1].item()}\n"
        inferred.write(output)

        idx += 1
