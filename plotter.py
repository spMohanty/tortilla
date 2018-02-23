import torch
import time
import os

import numpy as np

from visdom import Visdom
import random


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
                        opts={}, port=8097, server='localhost',
                        debug=False):
        self.experiment_name = experiment_name
        self.fields = fields
        self.win = win
        self.env = self.experiment_name
        self.opts = opts
        self.default_opts = {}
        self.port = port
        self.server = server
        self.debug = debug

        self.init_visdom_server()
        self.plot_initalised = False

    def init_visdom_server(self):
        self.vis = Visdom(server="http://"+self.server, port=self.port)

    def update_opts(self):
        self._opts = self.default_opts.copy()
        self._opts.update(self.opts)
        self.opts = self._opts

class TortillaLinePlotter(TortillaBasePlotter):
    def __init__(   self, experiment_name=None, fields=None,
                    title=None, opts={}, port=8097, server='localhost',
                    debug=False):
        super(TortillaLinePlotter, self).__init__(
                    experiment_name=experiment_name, fields=fields,
                    win=title, opts=opts, port=port, server=server,
                    debug=debug)

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
                    title=None, opts={}, port=8097, server='localhost',
                    debug=False):
        super(TortillaHeatMapPlotter, self).__init__(
                    experiment_name=experiment_name, fields=fields,
                    win=title, opts=opts, port=port, server=server,
                    debug=debug)

        self.default_opts = dict(
            legend = self.fields,
            showlegend = True,
            title = self.win,
            marginbottom = 50,
            marginleft = 50,
            connectgaps=True
        )
        self.update_opts() #merge supplied opts into default_opts

    def update_plot(self, XY):
        """
        Args:
            XY : A 2D array representing a confusion matrix
        """
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
                    title=None, opts={}, port=8097, server='localhost',
                    debug=False):
        super(TortillaImagesPlotter, self).__init__(
                    experiment_name=experiment_name, fields=fields,
                    win=title, opts=opts, port=port, server=server,
                    debug=debug)

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

    XY = np.random.randn(10, 10)
    plotter = TortillaHeatMapPlotter(experiment_name="test", title="mohanty")
    plotter.update_plot(XY)
