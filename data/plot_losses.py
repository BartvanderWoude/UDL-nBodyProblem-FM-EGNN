import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_losses(losses, y_lim, title, file_name, xlabel='Epoch', ylabel='Loss', legend=None):
    # 8 colors to use for the plot
    colors = plt.get_cmap('tab10').colors[:8]

    plt.figure(figsize=(12, 8))
    for key in losses.keys():
        if legend is None:
            plt.plot(losses[key]['epoch'], losses[key]['loss'], label=str(key), color=colors[key % 8])
        else:
            plt.plot(losses[key]['epoch'], losses[key]['loss'], label=legend[key], color=colors[key % 8])
    plt.ylim(0, y_lim)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.title(title, fontsize=28)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=20)
    plt.tight_layout()
    # plt.show()

    plt.savefig(file_name)


# Get all loss files
train_loss_files = [f for f in os.listdir('losses') if 'loss' in f]
val_loss_files = [f for f in os.listdir('losses') if 'val' in f]

# Sort the files
train_loss_files.sort()
val_loss_files.sort()

print("Legend:")
for i, f in enumerate(train_loss_files):
    if "N35" in f or "N70" in f:
        continue
    x = f.replace('loss_', '').replace('.csv', '')
    print(f"{i}: {x}")

# Create a dictionary to store all training losses
trainlosses = {}
for i, f in enumerate(train_loss_files):
    if "N35" in f or "N70" in f:
        continue
    trainlosses[i] = pd.read_csv('losses/' + f, sep=',', names=['epoch', 'pos_loss', 'vel_loss', 'loss'])

# Create a dictionary to store all validation losses
vallosses = {}
for i, f in enumerate(val_loss_files):
    if "N35" in f or "N70" in f:
        continue
    vallosses[i] = pd.read_csv('losses/' + f, sep=',', names=['epoch', 'pos_loss', 'vel_loss', 'loss'])

# Create a dictionary to store the main training/ val loss
mainlosses = {}
mainlosses[0] = pd.read_csv('losses/loss_N70_batch32_lr0.0001_fd4_b0.5.csv', sep=',', names=['epoch', 'pos_loss', 'vel_loss', 'loss'])
mainlosses[1] = pd.read_csv('losses/val_N70_batch32_lr0.0001_fd4_b0.5.csv', sep=',', names=['epoch', 'pos_loss', 'vel_loss', 'loss'])

# Set epoch columns to integer
for key in trainlosses.keys():
    trainlosses[key]['epoch'] = trainlosses[key]['epoch'].astype(int)
for key in vallosses.keys():
    vallosses[key]['epoch'] = vallosses[key]['epoch'].astype(int)
for key in mainlosses.keys():
    mainlosses[key]['epoch'] = mainlosses[key]['epoch'].astype(int)

plot_losses(trainlosses, 1.0, 'Training losses', 'plots/training_losses_large.png')
plot_losses(trainlosses, 0.0025, 'Training losses', 'plots/training_losses_small.png')
plot_losses(vallosses, 0.3, 'Validation losses', 'plots/validation_losses_large.png')
plot_losses(vallosses, 0.005, 'Validation losses', 'plots/validation_losses_small.png')
plot_losses(vallosses, 0.0001, 'Validation losses', 'plots/validation_losses_extrasmall.png')

plot_losses(mainlosses, 0.55, 'Training and Validation losses', 'plots/main_losses_large.png', legend=['Training', 'Validation'])
plot_losses(mainlosses, 0.0025, 'Training and Validation losses', 'plots/main_losses_small.png', legend=['Training', 'Validation'])
