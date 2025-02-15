from nbody_fm import NBodyData, EGNN_network, infer

import torch
import argparse
import os

from torch.utils.data import DataLoader, Subset


if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("losses"):
    os.makedirs("losses")
if not os.path.exists("infer"):
    os.makedirs("infer")

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True)
parser.add_argument("--infermethod", type=str, required=True)
parser.add_argument("--solverstepsize", type=float, required=False, default=0.01)
parser.add_argument("--lookahead", type=int, required=True)
parser.add_argument("--inferencesteps", type=int, required=True)
parser.add_argument("--fulldata", type=bool, required=False, default=False)

args = parser.parse_args()

# Variable hyperparameters
MODEL_FILE = args.model
INFERENCE_METHOD = args.infermethod  # Non-adaptive: "euler", "rk2", "rk4" || # Adaptive: "dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun"
SOLVER_STEP_SIZE = args.solverstepsize  # Only necessary when ODE solver is not adaptive
LOOK_AHEAD = args.lookahead
INFERENCE_STEPS = args.inferencesteps
FULL_DATA = args.fulldata

print(f"MODEL_FILE: {MODEL_FILE}")
print(f"INFERENCE_METHOD: {INFERENCE_METHOD}")
print(f"SOLVER_STEP_SIZE: {SOLVER_STEP_SIZE}")
print(f"LOOK_AHEAD: {LOOK_AHEAD}")
print(f"INFERENCE_STEPS: {INFERENCE_STEPS}")
print(f"FULL_DATA: {FULL_DATA}")

INFER_FILE = f"infer_M{INFERENCE_METHOD}_SS{SOLVER_STEP_SIZE}_LA{LOOK_AHEAD}_IS{INFERENCE_STEPS}_FD{FULL_DATA}.csv"
INFER_LOSS = f"inferloss_M{INFERENCE_METHOD}_SS{SOLVER_STEP_SIZE}_LA{LOOK_AHEAD}_IS{INFERENCE_STEPS}_FD{FULL_DATA}.csv"

# Constant hyperparameters
DATAFILE = "3body_2d_data.csv"
NUMBER_OF_LAYERS = 2
USE_TIME_EMBEDDING = True

# Extract hyperparameters from MODEL_FILE N20_batch32_lr0.0001_fd4_b0.5
hyperparameters = MODEL_FILE.replace(".pth", "").split("_")
NEPOCHS = [int(x.replace("N", "")) for x in hyperparameters if "N" in x][0]
BATCH_SIZE = [int(x.replace("batch", "")) for x in hyperparameters if "batch" in x][0]
LEARNING_RATE = [float(x.replace("lr", "")) for x in hyperparameters if "lr" in x][0]
FEATURE_DIM = [int(x.replace("fd", "")) for x in hyperparameters if "fd" in x][0]
BETA = [float(x.replace("b", "")) for x in hyperparameters if "b" in x and "batch" not in x][0]

dataset = NBodyData(DATAFILE)
traindata_index = int(0.7 * len(dataset))
valdata_index = int(0.85 * len(dataset))

traindataset = Subset(dataset, range(traindata_index))
valdataset = Subset(dataset, range(traindata_index, valdata_index))
testdataset = Subset(dataset, range(valdata_index, len(dataset)))

traindataloader = DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True)
valdataloader = DataLoader(valdataset, batch_size=BATCH_SIZE, shuffle=True)
testdataloader = DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=True)

if not FULL_DATA:
    dataset = testdataset

loss_fn = torch.nn.MSELoss()

vf = EGNN_network(number_of_layers=NUMBER_OF_LAYERS,
                  use_time_embedding=USE_TIME_EMBEDDING,
                  feature_dim=FEATURE_DIM)
vf.load_state_dict(torch.load("models/" + MODEL_FILE))

infer(vf=vf,
      dataset=dataset,
      inference_method=INFERENCE_METHOD,
      inference_steps=INFERENCE_STEPS,
      output_file=INFER_FILE,
      loss_file=INFER_LOSS,
      loss_fn=loss_fn,
      step_size=SOLVER_STEP_SIZE,
      look_ahead=LOOK_AHEAD)
