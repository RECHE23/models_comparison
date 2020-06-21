# -*- coding: utf-8 -*-

import os.path
from IPython.display import Video

# Import Numpy:
import numpy as np
np.random.seed(42)

# Import PyTorch:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
torch.manual_seed(42)

# Setting the device to use:
device = None
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    print("Running on GPU 0")
else:
    print("Running on CPU")

# Import MatPlotLib:
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.animation import FuncAnimation, FFMpegWriter
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
matplotlib.rcParams['animation.html'] = 'html5'
matplotlib.rcParams["animation.writer"] = 'imagemagick'
if os.path.isfile('/usr/bin/ffmpeg'):
    matplotlib.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
norm = cm.colors.Normalize(vmin=-1.0, vmax=1.0)
plt.tight_layout()

# Suppress useless warnings:
import warnings
warnings.filterwarnings("ignore")

# Import SKLearn datasets:
import sklearn.datasets

# Import ffmpeg-python:
import ffmpeg


# Define the accuracy function:
def accuracy(y_pred, y):
    pred_y_copy = y_pred.clone()
    pred_y_copy[pred_y_copy < 0] = -1
    pred_y_copy[pred_y_copy >= 0] = 1
    acc_pred = (pred_y_copy == y).float().sum() / y.shape[0]
    return acc_pred * 100


# Define the Hinge loss function:
def hinge_loss(input, target, reduction='mean'):
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(),
                                                                      input.size()),
                      stacklevel=2)
    ret = 1 - input * target
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


# Define the Hinge loss Module:
class HingeLoss(nn.Module):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        self.reduction = reduction
        super(HingeLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return hinge_loss(input, target, reduction=self.reduction)


# Custom Dataset class:
class MyDataset(Dataset):
    def __init__(self, ds):
        self.X, self.y = ds

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return [self.X[index], self.y[index]]


def convert_to_pytorch_dataset(X, y):
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y.reshape(-1, 1)).float()
    X_tensor = X_tensor.to(device)
    y_tensor = y_tensor.to(device)
    return MyDataset((X_tensor, y_tensor))


# Define the function that generates the animated gif:
def generate_animated_plot(epochs=400, datasets=None, models=None,
                           optimizers=None, criterion=None,
                           filename="animation", batch_size=32, shuffle=True):
    # Define the plotting function:
    def plot_decision_boundaries(fig):
        fig.clf()
        h = .02  # Step size in the mesh.
        i = 1

        # Plot the points:
        def plot_points(ax, X, y):
            plt.scatter(X.cpu()[:, 0], X.cpu()[:, 1], s=40,
                        c=y.flatten().cpu(), edgecolors='k', alpha=0.75)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())

        # Iterate over datasets:
        for ds_index, ds in enumerate(datasets):
            # Separates the characteristics from the labels:
            X, y = ds.X, ds.y

            # Creates a mesh grid from the characteristics space:
            x_span = np.linspace(min(X.cpu()[:, 0]), max(X.cpu()[:, 0]))
            y_span = np.linspace(min(X.cpu()[:, 1]), max(X.cpu()[:, 1]))
            xx, yy = np.meshgrid(x_span, y_span)
            grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])

            # Initialize the subplot:
            ax = fig.add_subplot(len(datasets), len(models[ds_index]) + 1, i)

            # Adds a title:
            if ds_index == 0:
                ax.set_title("Input data", fontdict={'fontsize': 16,
                                                     'fontweight': 'medium'})

            # Plot the points:
            plot_points(ax, X, y)
            i += 1

            # Iterate over models:
            for model_name, model in models[ds_index].items():
                ax = plt.subplot(len(datasets), len(models[ds_index]) + 1, i)

                # Evaluate the accuracy:
                y_pred = model.forward(X)
                score = accuracy(y_pred, y)

                # Evaluate the loss:
                if isinstance(criterion, dict):
                    loss = criterion[model_name](y_pred, y)
                else:
                    loss = criterion(y_pred, y)

                # Plot the decision boundary:
                pred_func = model.forward(grid.to(device))
                z = pred_func.cpu().view(xx.shape).detach().numpy()

                # Put the result into a color plot:
                ax.contourf(xx, yy, z, zorder=0, vmin=-1.0, vmax=1.0,
                            norm=norm, alpha=.95)
                plt.contour(xx, yy, z, levels=[-.0001, 0.0001], colors='r')

                # Plot the points:
                plot_points(ax, X, y)

                # Adds a title:
                if ds_index == 0:
                    ax.set_title(model_name, fontdict={'fontsize': 16,
                                                       'fontweight': 'medium'})

                # Adds the score:
                txt1 = ax.text(0.04, 0.03, ('Loss: %.2f' % loss).lstrip('0'),
                               horizontalalignment='left',
                               verticalalignment='bottom',
                               fontsize=14, color='w', weight='bold',
                               transform=ax.transAxes)
                txt2 = ax.text(0.96, 0.03, ('Acc: %.2f' % score).lstrip('0'),
                               horizontalalignment='right',
                               verticalalignment='bottom',
                               fontsize=14, color='w', weight='bold',
                               transform=ax.transAxes)
                txt1.set_path_effects(
                    [PathEffects.withStroke(linewidth=1, foreground='k')])
                txt2.set_path_effects(
                    [PathEffects.withStroke(linewidth=1, foreground='k')])
                i += 1

    # Define the update function:
    def update(i):
        label = f'Epoch {i} / {epochs}'
        display_progress_bar(i, epochs,
                             prefix='Working:   ', suffix='', length=90)

        # Iterate over datasets:
        for ds_index, ds in enumerate(datasets):
            train_loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
            for X_batch, y_batch in train_loader:
                # Iterate over models:
                for model_name, model in models[ds_index].items():
                    model.to(device)
                    model.train()
                    def closure():
                        # Do a forward pass:
                        y_pred = model.forward(X_batch)

                        # Evaluate the loss:
                        if isinstance(criterion, dict):
                            loss = criterion[model_name](y_pred, y_batch)
                        else:
                            loss = criterion(y_pred, y_batch)

                        # Do the backward pass:
                        optimizers[ds_index][model_name].zero_grad()
                        loss.backward()
                        return loss

                    optimizers[ds_index][model_name].step(closure)

        # Update the plot:
        plot_decision_boundaries(fig)
        fig.tight_layout(pad=5.00)
        fig.subplots_adjust(bottom=0.1, top=0.9)

        # Update the title:
        fig.suptitle(label, fontsize=25, y=0.075, x=0.978, ha='right')
        if i == epochs:
            print()
            print(u'\x1b[?25l', end='')
            print("Generating the animation... Please wait!", end=u"\r")

    fig = plt.figure(figsize=(32, 18), dpi=80)  # WQHD Resolution: 2560 x 1440
    anim = FuncAnimation(fig, update, frames=(epochs + 1), interval=1)
    anim.save(f'{filename}.mp4', dpi=80, fps=10, writer='ffmpeg')
    print(u'\r' + ' ' * 90, end=u'\r')
    print(u'\x1b[?25h', end='')
    plt.close()
    return Video(f'./{filename}.mp4', embed=True, width=960, height=540, html_attributes="loop autoplay")


def display_progress_bar(iteration, total, prefix='', suffix='', decimals=1,
                         length=100, fill=u'\u2588', print_end=u"\r"):
    if iteration == 0:
        print(u'\x1b[?25l', end='')
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    line = u'\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix)
    print(line, end=print_end)
    if iteration == total:
        print(u'\r' + ' ' * len(line), end=u'\r')
        print(u'\x1b[?25h', end='')
