import os
import pandas as pd


file = open("plots/loss_table.txt", "w", encoding='utf-8')

file.write("\\begin{table}[H]\n")
file.write("\\centering\n")
file.write("\\begin{tabular}{|c|c|c|c|c|c|c|}\n")
file.write("\\hline\n")
file.write("Model & Pos Loss & Vel Loss & Loss & Val Pos Loss & Val Vel Loss & Val Loss \\\\ \n")

files = [f for f in os.listdir('losses') if 'loss' in f and "N35" not in f and "N70" not in f]
files.sort()

print("Legend:")
for i, f in enumerate(files):
    if "N35" in f or "N70" in f:
        continue
    x = f.replace('loss_', '').replace('.csv', '')
    print(f"{i}: {x}")

for i, f in enumerate(files):
    traindata = pd.read_csv('losses/' + f, sep=',', names=['epoch', 'pos_loss', 'vel_loss', 'loss'])
    valdata = pd.read_csv('losses/' + f.replace("loss_", "val_"), sep=',', names=['epoch', 'pos_loss', 'vel_loss', 'loss'])
    out = f"{i} & {traindata['pos_loss'].mean():.4f} & {traindata['vel_loss'].mean():.4f} & {traindata['loss'].mean():.4f}"
    out += f" & {valdata['pos_loss'].mean():.4f} & {valdata['vel_loss'].mean():.4f} & {valdata['loss'].mean():.4f} \\\\ \n"
    file.write(out)
    file.write("\\hline\n")


file.write("\\end{tabular}\n")
file.write("\\end{table}\n")
file.close()