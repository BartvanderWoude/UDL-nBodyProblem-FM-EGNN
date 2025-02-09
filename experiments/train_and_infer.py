from nbody_fm import NBodyData, EGNN_network, train, infer

import torch

from torch.utils.data import DataLoader


NEPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
DATAFILE = "3body_2d_data.csv"

INFERENCE_STEPS = 50
SOLVER_STEP_SIZE = 0.01
LOOK_AHEAD = 1
OUTPUT_FILE = "inferred.csv"


def run():
    dataset = NBodyData(DATAFILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = torch.nn.MSELoss()
    vf = EGNN_network()

    vf = train(vf, dataloader, loss_fn, NEPOCHS, LEARNING_RATE)

    torch.save(vf.state_dict(), "model.pth")

    infer(vf, dataset, INFERENCE_STEPS, SOLVER_STEP_SIZE, OUTPUT_FILE, look_ahead=LOOK_AHEAD)


if __name__ == "__main__":
    run()
