from nbody_fm import CoorsWrappedModel, VelWrappedModel

from tqdm import tqdm
from flow_matching.solver import ODESolver


def infer(vf, dataset, inference_method, inference_steps, output_file, step_size=0.01, look_ahead=1):
    if inference_method in ["dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun"]:
        step_size = None

    coors_model = CoorsWrappedModel(model=vf)
    vel_model = VelWrappedModel(model=vf)

    coors_solver = ODESolver(velocity_model=coors_model)
    vel_solver = ODESolver(velocity_model=vel_model)

    inferred = open(output_file, "w")
    idx = 0

    for i in tqdm(range(inference_steps)):
        if i % look_ahead == 0:
            x_0, vel_0, x_1, vel_1 = dataset[i]
            x_0 = x_0.unsqueeze(0)
            vel_0 = vel_0.unsqueeze(0)

        x_0 = coors_solver.sample(x_init=x_0, step_size=step_size, method=inference_method, vel=vel_0)
        vel_0 = vel_solver.sample(x_init=vel_0, step_size=step_size, method=inference_method, coors=x_0)

        output = f"{x_0[0, 0, 0].item()},{x_0[0, 0, 1].item()},"
        output += f"{x_0[0, 1, 0].item()},{x_0[0, 1, 1].item()},"
        output += f"{x_0[0, 2, 0].item()},{x_0[0, 2, 1].item()}\n"
        inferred.write(output)

        idx += 1
