import torch
import time
import os

import numpy as np

from visdom import Visdom
import random

"""
    Deals with all the logging and plotting
    requirements during the training
"""
class Plotter:
    def __init__(self, experiment_name, logdir):
        self.experiment_name = experiment_name
        self.logdir = logdir
        self.create_logdir()

        self.vis = Visdom()
        self.window_map = {
            "accuracy_plot" : self.experiment_name + "::Acc",
            "loss_plot" : self.experiment_name + "::loss",
        }

        self.history = {
            "accuracy" : {
                "train": {
                    "top_1": [],
                    "top_5": []
                },
                "val": {
                    "top_1": [],
                    "top_5": []
                }
            },
            "loss" : {
                "train" : [],
                "val" : []
            }
        }

        self.init_accuracy_plot()
        self.init_loss_plot()

    def init_loss_plot(self):
        self.loss_plot_opts = dict(
            legend = [
                "train_loss",
                "val_loss",
                ],
            showlegend = True,
            title = self.window_map["loss_plot"],
            xlabel = "Epochs",
            ylabel = "Loss",
            xtickmin = 0,
            xtickmax = 100,
            marginbottom = 50,
            marginleft = 50
        )

        self.vis.line(
            X = np.zeros(1),
            Y = np.zeros((1,2)),
            name = self.window_map["loss_plot"],
            win = self.window_map["loss_plot"],
            opts = self.loss_plot_opts
        )

    def init_accuracy_plot(self):
        self.accuracy_plot_opts = dict(
            legend = [
                "train_top_1",
                "train_top_5",
                "val_top_1",
                "val_top_5",
                ],
            showlegend = True,
            title = self.window_map["accuracy_plot"],
            xlabel = "Epochs",
            ylabel = "Accuracy",
            xtickmin = 0,
            xtickmax = 100,
            ytickmin = 0,
            ytickmax = 100,
            marginbottom = 50,
            marginleft = 50
        )
        self.vis.line(
            X = np.zeros(1),
            Y = np.zeros((1,4)),
            name = self.window_map["accuracy_plot"],
            win = self.window_map["accuracy_plot"],
            opts = self.accuracy_plot_opts
        )

    def update_loss(self, epoch, loss, train=True):
        y = np.zeros((1,2))
        y[:] = np.nan
        x = np.zeros((1,2))
        x[:] = epoch
        if train:
            y[0][0] = loss
        else:
            y[0][1] = loss

        self.vis.line(
            X = x,
            Y = y,
            win = self.window_map["loss_plot"],
            update = "append",
            opts = self.loss_plot_opts
        )

    def update_accuracy(self, epoch, top_1, top_5, train=True):
        y = np.zeros((1,4))
        y[:] = np.nan
        x = np.zeros((1,4))
        x[:] = epoch
        if train:
            y[0][0] = top_1
            y[0][1] = top_5
        else:
            y[0][2] = top_1
            y[0][3] = top_5

        print("Updating : ", x, y)
        self.vis.line(
            X = x,
            Y = y,
            win = self.window_map["accuracy_plot"],
            update = "append",
            opts = self.accuracy_plot_opts
        )

    def create_logdir(self):
        if not os.path.exists(self.logdir):
            os.mkdir(self.logdir)

if __name__ == "__main__":
    print("Testing")

    # experiment_name = "exp1"
    plotter = Plotter(experiment_name="exp1", logdir="experiments/exp1")
    # plotter.update_accuracy(10, 11)


    # vis = Visdom()
    #
    for k in range(1000):
        print("Update : ", k)
        plotter.update_accuracy(k*1.0/10, random.randint(0,100), random.randint(0,100))
        plotter.update_loss(k*1.0/10, random.randint(0, 100)*1.0/100)
        time.sleep(1.0/10)
