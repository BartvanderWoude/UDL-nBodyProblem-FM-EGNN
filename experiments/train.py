from nbody_fm import NBodyData, EGNN_network, train

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

parser.add_argument("--nepochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--feature_dim", type=int)
parser.add_argument("--beta", type=float)

args = parser.parse_args()

# Variable hyperparameters
NEPOCHS = args.nepochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
FEATURE_DIM = args.feature_dim
BETA = args.beta

LOSS_FILE = f"loss_N{NEPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_fd{FEATURE_DIM}_b{BETA}.csv"
VAL_FILE = f"val_N{NEPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_fd{FEATURE_DIM}_b{BETA}.csv"
MODEL_FILE = f"model_N{NEPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_fd{FEATURE_DIM}_b{BETA}.pth"
INFER_FILE = f"infer_N{NEPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_fd{FEATURE_DIM}_b{BETA}.csv"

# Constant hyperparameters
DATAFILE = "3body_2d_data.csv"
NUMBER_OF_LAYERS = 2
USE_TIME_EMBEDDING = True

# Inference hyperparameters
INFERENCE_METHOD = "dopri5"  # Non-adaptive: "euler", "rk2", "rk4" || # Adaptive: "dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun"
SOLVER_STEP_SIZE = 0.01  # Only necessary when ODE solver is not adaptive
INFERENCE_STEPS = 7000
LOOK_AHEAD = 7000

dataset = NBodyData(DATAFILE)
traindata_index = int(0.7 * len(dataset))
valdata_index = int(0.85 * len(dataset))

traindataset = Subset(dataset, range(traindata_index))
valdataset = Subset(dataset, range(traindata_index, valdata_index))
testdataset = Subset(dataset, range(valdata_index, len(dataset)))

traindataloader = DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True)
valdataloader = DataLoader(valdataset, batch_size=BATCH_SIZE, shuffle=True)
testdataloader = DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=True)

loss_fn = torch.nn.MSELoss()
vf = EGNN_network(number_of_layers=NUMBER_OF_LAYERS,
                  use_time_embedding=USE_TIME_EMBEDDING,
                  feature_dim=FEATURE_DIM)

vf = train(vf=vf,
           traindataloader=traindataloader,
           valdataloader=valdataloader,
           loss_fn=loss_fn,
           nepochs=NEPOCHS,
           lr=LEARNING_RATE,
           beta=BETA,
           loss_file_name=LOSS_FILE,
           val_file_name=VAL_FILE)

torch.save(vf.state_dict(), "models/" + MODEL_FILE)
