import os
import pandas as pd


file = open("plots/loss_table.txt", "w", encoding='utf-8')
gridIDs = open("plots/gridIDs_table.txt", "w", encoding='utf-8')

file.write("\\begin{table}[H]\n")
file.write("\\centering\n")
file.write("\\caption{Table showing the minimum training and validation loss of the gridsearch. Model ID corresponds to Table \\ref{tab:modelID}. \\textit{Pos. loss} is the MSE of the position of the nodes, \\textit{Vel. loss} is the MSE of the velocities of the nodes, \\textit{Loss} is the $\beta$ combination of the \\textit{Pos. loss} and \\textit{Vel. loss}.}\n")
file.write("\\label{tab:gridsearch_losses}\n")
file.write("\\begin{tabular}{|c|c c c|c c c|}\n")
file.write("\\hline\n")
file.write("& & Training & & & Validation & \\\\ \n")
file.write("\\hline\n")
file.write("ID & Pos. Loss & Vel. Loss & Loss & Pos. Loss & Vel. Loss & Loss \\\\ \n")
file.write("\\hline\n")

gridIDs.write("\\begin{table}[H]\n")
gridIDs.write("\\centering\n")
gridIDs.write("\\caption{Legend for the gridsearch.}\n")
gridIDs.write("\\label{tab:modelID}\n")
gridIDs.write("\\begin{tabular}{|c|c|c|c|c|}\n")
gridIDs.write("\\hline\n")
gridIDs.write("ID & Epochs & Batch size & Learning rate & Beta \\\\ \n")
gridIDs.write("\\hline\n")


files = [f for f in os.listdir('losses') if 'loss' in f and "N35" not in f and "N70" not in f]
files.sort()

print("Legend:")
for i, f in enumerate(files):
    if "N35" in f or "N70" in f:
        continue
    x = f.replace('loss_', '').replace('.csv', '')
    print(f"{i}: {x}")

    hyperparameters = x.split("_")
    NEPOCHS = [int(x.replace("N", "")) for x in hyperparameters if "N" in x][0]
    BATCH_SIZE = [int(x.replace("batch", "")) for x in hyperparameters if "batch" in x][0]
    LEARNING_RATE = [float(x.replace("lr", "")) for x in hyperparameters if "lr" in x][0]
    FEATURE_DIM = [int(x.replace("fd", "")) for x in hyperparameters if "fd" in x][0]
    BETA = [float(x.replace("b", "")) for x in hyperparameters if "b" in x and "batch" not in x][0]

    out = f"{i} & {NEPOCHS} & {BATCH_SIZE} & {LEARNING_RATE} & {BETA} \\\\ \n"
    gridIDs.write(out)
    gridIDs.write("\\hline\n")

for i, f in enumerate(files):
    traindata = pd.read_csv('losses/' + f, sep=',', names=['epoch', 'pos_loss', 'vel_loss', 'loss'])
    valdata = pd.read_csv('losses/' + f.replace("loss_", "val_"), sep=',', names=['epoch', 'pos_loss', 'vel_loss', 'loss'])
    out = f"{i} & {traindata['pos_loss'].min():.6f} & {traindata['vel_loss'].min():.6f} & {traindata['loss'].min():.6f}"
    out += f" & {valdata['pos_loss'].min():.6f} & {valdata['vel_loss'].min():.6f} & {valdata['loss'].min():.6f} \\\\ \n"
    file.write(out)
    file.write("\\hline\n")


file.write("\\end{tabular}\n")
file.write("\\end{table}\n")
file.close()

gridIDs.write("\\end{tabular}\n")
gridIDs.write("\\end{table}\n")
gridIDs.close()
