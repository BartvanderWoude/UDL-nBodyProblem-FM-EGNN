import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_losses(losses, y_lim, title, file_name, xlabel='Epoch', ylabel='Loss'):
    # 8 colors to use for the plot
    colors = plt.get_cmap('tab10').colors[:8]

    plt.figure(figsize=(12, 8))
    for key in losses.keys():
        plt.plot(losses[key]['epoch'], losses[key]['loss'], label=str(key), color=colors[key % 8])
    plt.ylim(0, y_lim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
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
    x = f.replace('loss_', '').replace('.csv', '')
    print(f"{i}: {x}")

# Create a dictionary to store all training losses
trainlosses = {}
for i, f in enumerate(train_loss_files):
    trainlosses[i] = pd.read_csv('losses/' + f, sep=',', names=['epoch', 'pos_loss', 'vel_loss', 'loss'])

# Create a dictionary to store all validation losses
vallosses = {}
for i, f in enumerate(val_loss_files):
    vallosses[i] = pd.read_csv('losses/' + f, sep=',', names=['epoch', 'pos_loss', 'vel_loss', 'loss'])

plot_losses(trainlosses, 1.0, 'Training losses', 'training_losses_large.png')
plot_losses(trainlosses, 0.0025, 'Training losses', 'training_losses_small.png')
plot_losses(vallosses, 0.3, 'Validation losses', 'validation_losses_large.png')
plot_losses(vallosses, 0.005, 'Validation losses', 'validation_losses_small.png')
plot_losses(vallosses, 0.0001, 'Validation losses', 'validation_losses_extrasmall.png')