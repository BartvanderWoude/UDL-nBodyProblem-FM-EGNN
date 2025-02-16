import os
import pandas as pd


file = open("plots/infer_table.txt", "w", encoding='utf-8')
inferIDs = open("plots/inferIDs_table.txt", "w", encoding='utf-8')

file.write("\\begin{table}[H]\n")
file.write("\\centering\n")
file.write("\\caption{Table showing the average MSE loss between expected values and predicted values of the inference. \\textit{Pos. loss} is the MSE of the position of the nodes, \\textit{Vel. loss} is the MSE of the velocities of the nodes, \\textit{Loss} is the combination of the \\textit{Pos. loss} and \\textit{Vel. loss}.}\n")
file.write("\\label{tab:inference_losses}\n")
file.write("\\begin{tabular}{|c|c c c|c c c|}\n")
file.write("\\hline\n")
file.write("& & Full data & & & Test data & \\\\ \n")
file.write("\\hline\n")
file.write("ID & Pos. Loss & Vel. Loss & Loss & Pos. Loss & Vel. Loss & Loss \\\\ \n")
file.write("\\hline\n")

inferIDs.write("\\begin{table}[H]\n")
inferIDs.write("\\centering\n")
inferIDs.write("\\caption{Legend for the inference hyperparameters.}\n")
inferIDs.write("\\label{tab:inferID}\n")
inferIDs.write("\\begin{tabular}{|c|c|c|c|c|}\n")
inferIDs.write("\\hline\n")
inferIDs.write("ID & Solver method & Solver step size & Look ahead & Inference steps \\\\ \n")
inferIDs.write("\\hline\n")


files = [f for f in os.listdir('infer') if 'inferloss_' in f]
files.sort()

already_done = []

print("Legend:")
i = 0
for f in files:
    if f in already_done:
        continue

    x = f.replace('inferloss_', '').replace('.csv', '')
    print(f"{i}: {x}")

    hyperparameters = x.split("_")
    METHOD = [x.replace("M", "") for x in hyperparameters if "M" in x][0]
    SOLVER_STEP_SIZE = [float(x.replace("SS", "")) for x in hyperparameters if "SS" in x][0]
    LOOK_AHEAD = [int(x.replace("LA", "")) for x in hyperparameters if "LA" in x][0]
    INFERENCE_STEPS = [int(x.replace("IS", "")) for x in hyperparameters if "IS" in x][0]
    FULL_DATA = [bool(x.replace("FD", "")) for x in hyperparameters if "FD" in x][0]

    fulldata = f.replace("False", "True")
    testdata = f.replace("True", "False")
    already_done.append(fulldata)
    already_done.append(testdata)

    out = f"{i} & {METHOD} & {SOLVER_STEP_SIZE} & {LOOK_AHEAD} & {INFERENCE_STEPS} \\\\ \n"
    inferIDs.write(out)
    inferIDs.write("\\hline\n")

    i += 1

already_done = []
i = 0
for f in files:
    if f in already_done:
        continue

    fulldata = f.replace("False", "True")
    testdata = f.replace("True", "False")

    fulldataframe = pd.read_csv('infer/' + fulldata, sep=',', names=['epoch', 'pos_loss', 'vel_loss', 'loss'])
    testdataframe = pd.read_csv('infer/' + testdata, sep=',', names=['epoch', 'pos_loss', 'vel_loss', 'loss'])
    out = f"{i} & {fulldataframe['pos_loss'].mean():.6f} & {fulldataframe['vel_loss'].mean():.6f} & {fulldataframe['loss'].mean():.6f}"
    out += f" & {testdataframe['pos_loss'].mean():.6f} & {testdataframe['vel_loss'].mean():.6f} & {testdataframe['loss'].mean():.6f} \\\\ \n"
    file.write(out)
    file.write("\\hline\n")

    already_done.append(fulldata)
    already_done.append(testdata)
    i += 1


file.write("\\end{tabular}\n")
file.write("\\end{table}\n")
file.close()

inferIDs.write("\\end{tabular}\n")
inferIDs.write("\\end{table}\n")
inferIDs.close()
