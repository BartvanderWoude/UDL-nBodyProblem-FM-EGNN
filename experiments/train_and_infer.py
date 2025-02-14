from nbody_fm import NBodyData, EGNN_network, train, infer

import torch

from torch.utils.data import DataLoader


NEPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DATAFILE = "3body_2d_data.csv"

NUMBER_OF_LAYERS = 2
USE_TIME_EMBEDDING = True

# Non-adaptive: "euler", "rk2", "rk4" || Adaptive: "dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun"
INFERENCE_METHOD = "dopri5"
SOLVER_STEP_SIZE = 0.01  # Only necessary when ODE solver is not adaptive
INFERENCE_STEPS = 7000
LOOK_AHEAD = 7000

OUTPUT_FILE = "inferred.csv"
SAVED_MODEL = "model.pth"


def run():
    dataset = NBodyData(DATAFILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = torch.nn.MSELoss()
    vf = EGNN_network(number_of_layers=NUMBER_OF_LAYERS,
                      use_time_embedding=USE_TIME_EMBEDDING,
                      feature_dim=3)

    vf = train(vf=vf,
               dataloader=dataloader,
               loss_fn=loss_fn,
               nepochs=NEPOCHS,
               lr=LEARNING_RATE)

    torch.save(vf.state_dict(), SAVED_MODEL)

    infer(vf=vf,
          dataset=dataset,
          inference_method=INFERENCE_METHOD,
          inference_steps=INFERENCE_STEPS,
          output_file=OUTPUT_FILE,
          step_size=SOLVER_STEP_SIZE,
          look_ahead=LOOK_AHEAD)


if __name__ == "__main__":
    run()
