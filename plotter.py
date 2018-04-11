import torch
import time
import os
import torchvision.utils as vutils
import tensorflow as tf

import numpy as np
import tfplot
import matplotlib

from visdom import Visdom
from tensorboardX import SummaryWriter
import random


def plot_confusion_matrix(XY, tensor_name, classes):

    np.set_printoptions(precision=2)

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(XY, cmap='Blues')

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=20)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=14, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=20)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=14, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return summary


class VisdomTest:
    def __init__(self, server='localhost', port=8097):
        self.server = server
        self.port = port
        try:
            vis = Visdom(server="http://"+self.server, port=self.port)
            vis.win_exists("test")#dummy call to make sure that a connection to
            # visdom can be established
        except:
            print("\n\n\TortillaError :: Unable to connect to Visdom Server even when you have plotting on. \n\
Are you sure that you have visdom server running at : \
http://{}:{} ? \n \
If not, please start visdom by running : \n\npython -m visdom.server \n\n\n \
Or...disable plotting by passing the --no-plots option. \
                ".format(server,port))
            exit(0)
"""
    Deals with all the logging and plotting
    requirements during the training
"""
class TortillaBasePlotter:
    def __init__(self,  experiment_name=None, fields=None, win=None,
                        opts={}, platform="tensorboard", port=8097, server='localhost',
                        debug=False):
        self.experiment_name = experiment_name
        self.fields = fields
        self.win = win
        self.env = self.experiment_name
        self.opts = opts
        self.default_opts = {}
        self.platform = platform
        self.port = port
        self.server = server
        self.debug = debug

        self.init_server()
        self.plot_initalised = False
        self.log_dir = os.path.join("experiments",self.experiment_name,"tb_logs")

    def init_server(self):
        if self.platform == "visdom":
            self.vis = Visdom(server="http://"+self.server, port=self.port)
        elif self.platform == "tensorboard":
            self.writer = SummaryWriter(self.log_dir)

    def update_opts(self):
        self._opts = self.default_opts.copy()
        self._opts.update(self.opts)
        self.opts = self._opts

class TortillaLinePlotter(TortillaBasePlotter):
    def __init__(   self, experiment_name=None, fields=None,
                    title=None, opts={}, platform="tensorboard", port=8097,
                    server='localhost', debug=False):
        super(TortillaLinePlotter, self).__init__(
                    experiment_name=experiment_name, fields=fields,
                    win=title, opts=opts, platform=platform, port=port,
                    server=server, debug=debug)

        self.default_opts = dict(
            legend = self.fields,
            showlegend = True,
            title = self.win,
            marginbottom = 50,
            marginleft = 50,
            connectgaps=True
        )
        self.update_opts() #merge supplied opts into default_opts

    def append_plot(self, y, t):
        """
        Args:
            y : An array or 1D np-array of size 1 x number-of-fields
            t : A floating point number representing the location along
                time-axis
        """

        y = np.array(y).reshape((1,len(self.fields)))
        t = np.array([t])

        if self.platform == "tensorboard":
            dictionary =  dict(zip(self.fields,y[0].tolist()))
            self.writer.add_scalars(self.win, dictionary, t)

        elif self.platform == "visdom":
            if self.plot_initalised:
                win = self.vis.line(
                    Y = y,
                    X = t,
                    win = self.win,
                    env=self.env,
                    update = "append",
                    opts = self.opts
                )

            else:
                # Instantiate
                win = self.vis.line(
                    Y = y,
                    X = t,
                    env=self.env,
                    win = self.win,
                    opts = self.opts
                )
                self.plot_initalised = True

    def append_plot_with_dict(self, d, t):
        """
        Args:
            d:  A dictionary containing scalar values keyed by field names
                (as specified by self.fields)
            t : As floating point number representing the location along
                time-axis
        """
        payload = np.zeros((1, len(self.fields)))
        payload[:] = np.nan
        for _key in d.keys():
            _index = self.fields.index(_key)
            if _index > -1:
                payload[0, _index] = d[_key]
        self.append_plot(payload, t)

class TortillaHeatMapPlotter(TortillaBasePlotter):
    def __init__(   self, experiment_name=None, fields=None,
                    title=None, opts={}, platform="tensorboard", port=8097,
                    server='localhost', debug=False):
        super(TortillaHeatMapPlotter, self).__init__(
                    experiment_name=experiment_name, fields=fields,
                    win=title, opts=opts, platform=platform, port=port,
                    server=server, debug=debug)

        self.default_opts = dict(
            legend = self.fields,
            showlegend = True,
            title = self.win,
            marginbottom = 100,
            marginleft = 100,
            xlabel="Predicted Labels",
            ylabel="True Labels",
            connectgaps=True
        )
        self.update_opts() #merge supplied opts into default_opts

    def update_plot(self, XY):
        """
        Args:
            XY : A 2D array representing a confusion matrix
        """
        if self.platform == "tensorboard":
            sess = tf.InteractiveSession()

            img_d_summary_writer = tf.summary.FileWriter(os.path.join(self.log_dir,"confusion_matrix"), sess.graph)
            img_d_summary = plot_confusion_matrix(XY=XY, tensor_name='Confusion matrix', classes = self.fields)
            img_d_summary_writer.add_summary(img_d_summary)

        elif self.platform == "visdom":
            print("Updating Plot : ", XY.shape)
            if self.plot_initalised:
                win = self.vis.heatmap(
                    X = XY,
                    win = self.win,
                    env=self.env,
                    opts = self.opts
                )
            else:
                # Instantiate
                win = self.vis.heatmap(
                    X = XY,
                    env=self.env,
                    win = self.win,
                    opts = self.opts
                )
                self.plot_initalised = True


class TortillaImagesPlotter(TortillaBasePlotter):
    def __init__(   self, experiment_name=None, fields=None,
                    title=None, opts={}, platform="tensorboard", port=8097,
                    server='localhost', debug=False):
        super(TortillaImagesPlotter, self).__init__(
                    experiment_name=experiment_name, fields=fields,
                    win=title, opts=opts, platform=platform, port=port,
                    server=server, debug=debug)

        self.default_opts = dict(
            legend = self.fields,
            showlegend = True,
            title = self.win,
            nrow = 3,
            marginbottom = 50,
            marginleft = 50,
        )
        self.update_opts() #merge supplied opts into default_opts

    def update_images(self, images):
        """
        Args:
            images : A 4D tensor representing a B x C x H x W tensor or a list of images
        """
        if self.platform == "tensorboard":
            x = vutils.make_grid(images, normalize=True, scale_each=True)
            self.writer.add_image('Images', x)

        elif self.platform == "visdom":
            if self.plot_initalised:
                win = self.vis.images(
                    tensor = images,
                    win = self.win,
                    env=self.env,
                    opts = self.opts
                )
            else:
                # Instantiate
                win = self.vis.images(
                    tensor = images,
                    env=self.env,
                    win = self.win,
                    opts = self.opts
                )
                self.plot_initalised = True


if __name__ == "__main__":
    # opts = dict(
    #     xlabel = "accuracy",
    #     ylabel = "epochs",
    # )
    # fields = ['top-1', 'top-2', 'top-3']
    # plotter = TortillaLinePlotter(
    #                     experiment_name="test-experiment",
    #                     fields=fields,
    #                     title='test-plot',
    #                     opts = opts
    #                     )
    # # Example of call for direct update
    # # for _idx, _t in enumerate(range(100)):
    # #     plotter.append_plot(np.random.randn(len(fields)), _t)
    #
    # for _idx, _t in enumerate(range(100)):
    #     _d = {}
    #     _d["top-1"] = np.random.randn(1)[0]
    #     _d["top-2"] = np.random.randn(1)[0]
    #     _d["top-3"] = np.random.randn(1)[0]
    #     plotter.append_plot_with_dict(_d, _t)
    XY = np.random.randn(4, 4)
    plotter = TortillaHeatMapPlotter(experiment_name="test", fields=['A','B','C','D'], title="mohanty")
    plotter.update_plot(XY)
    #X = (20,2)
    #print(X)
    #plot=TortillaLinePlotter(experiment_name="test", fields = ["Train", "test"], title= "Camille")
    #plot.append_plot(X, t=2)
