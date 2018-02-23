#!/usr/bin/env python
from utils import accuracy
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from datastream import TortillaDataStream
from plotter import TortillaLinePlotter, TortillaHeatMapPlotter

class TortillaMonitor:
    """
    Monitors the
    - (top-k) train accuracy
    - (top-k) val accuracy
    - train loss
    - val loss
    - Train Confusion Matrix
    - Val confusion Matrix
    """
    def __init__(self,  experiment_name, plot=True, topk=(1,5),
                        classes=[], config=None, use_gpu=False):
        self.experiment_name = experiment_name
        self.plot = plot
        self.topk = topk
        self.classes = classes
        self.config = config
        self.use_gpu = use_gpu

        self._init_data_gatherers()
        if self.plot:
            self._init_plotters()

    def _init_plotters(self):
        topklabels = ["top-"+str(x) for x in self.topk]
        self.train_accuracy_plotter = TortillaLinePlotter(
                            experiment_name=self.experiment_name,
                            fields=topklabels,
                            title='train-accuracy',
                            opts=dict(
                                        xtickmin = 0,
                                        xtickmax = self.config.epochs,
                                        ytickmin = 0,
                                        ytickmax = 100,
                                        xlabel="Epochs",
                                        ylabel="Accuracy"
                                )
                            )

        self.val_accuracy_plotter = TortillaLinePlotter(
                            experiment_name=self.experiment_name,
                            fields=topklabels,
                            title='val-accuracy',
                            opts = dict(
                                        xtickmin = 0,
                                        xtickmax = self.config.epochs,
                                        ytickmin = 0,
                                        ytickmax = 100,
                                        xlabel="Epochs",
                                        ylabel="Accuracy",
                                        markers=True
                                )
                            )

        self.loss_plotter = TortillaLinePlotter(
                            experiment_name=self.experiment_name,
                            fields=['train_loss', 'val_loss'],
                            title='Loss',
                            opts = dict(
                                        xtickmin = 0,
                                        xtickmax = self.config.epochs,
                                        xlabel="Epochs",
                                        ylabel="Loss",
                                        markers=True
                                )
                            )
        self.val_confusion_matrix_plotter = TortillaHeatMapPlotter(
                            experiment_name=self.experiment_name,
                            fields=self.classes,
                            title='Latest Validation Confusion Matrix',
                            opts = dict(
                                    rownames=self.classes,
                                    columnnames=self.classes
                                )
                            )


    def _init_data_gatherers(self):
        topklabels = ["top-"+str(x) for x in self.topk]
        self.train_accuracy = TortillaDataStream(
                                name="train-accuracy",
                                column_names=topklabels
                                )
        self.val_accuracy = TortillaDataStream(
                                name="val-accuracy",
                                column_names=topklabels
                                )

        self.train_epochs = TortillaDataStream(
                                name="train-epochs",
                                column_names=['epochs']
                                )
        self.val_epochs = TortillaDataStream(
                                name="val-epochs",
                                column_names=['epochs']
                                )

        self.train_loss = TortillaDataStream(
                                name="train-loss",
                                column_names=['loss']
                                )
        self.val_loss = TortillaDataStream(
                                name="val-loss",
                                column_names=['loss']
                                )

        self.train_confusion_matrix = TortillaDataStream(
                                name="train-confusion_matrix",
                                merge_mode="sum"
                                )
        self.val_confusion_matrix = TortillaDataStream(
                                name="val-confusion_matrix",
                                merge_mode="sum"
                                )

    def compute_confusion_matrix(self, outputs, labels):
        _, pred_top_1 = outputs.topk(1, 1, True, True)#compute top-1 predictions
        pred_top_1 = pred_top_1.t()
        if self.use_gpu:
            _labels = labels.data.cpu().numpy()
            _preds =  pred_top_1.data.cpu().numpy()[0]
        else:
            _labels = labels.data.numpy()
            _preds =  pred_top_1.data.numpy()[0]

        _batch_confusion_matrix = confusion_matrix(_labels, _preds, labels=range(len(self.classes)))
        return _batch_confusion_matrix

    def _compute_and_register_stats(self, epoch, outputs, labels, loss, train=True):
        _accuracy = accuracy(outputs, labels, topk=self.topk)
        _accuracy = np.array([x.data[0] for x in _accuracy])
        _batch_confusion_matrix = self.compute_confusion_matrix(outputs, labels)

        if train:
            accuracy_stream = self.train_accuracy
            epoch_stream = self.train_epochs
            loss_stream = self.train_loss
            confusion_matrix_stream = self.train_confusion_matrix
        else:
            accuracy_stream = self.val_accuracy
            epoch_stream = self.val_epochs
            loss_stream = self.val_loss
            confusion_matrix_stream = self.val_confusion_matrix

        accuracy_stream.add_to_buffer(_accuracy)
        epoch_stream.add_to_buffer(epoch)
        loss_stream.add_to_buffer(loss.data[0])
        confusion_matrix_stream.add_to_buffer(_batch_confusion_matrix)


    def _flush_stats(self, train=True):
        """
            Flush all the data from the datastream_buffers into the actual
            datastreams
        """
        if train:
            self.train_accuracy.flush_buffer()
            self.train_epochs.flush_buffer()
            self.train_loss.flush_buffer()
            self.train_confusion_matrix.flush_buffer()
        else:
            self.val_accuracy.flush_buffer()
            self.val_epochs.flush_buffer()
            self.val_loss.flush_buffer()
            self.val_confusion_matrix.flush_buffer()

        if self.plot:
            self._plot(train=train)

    def _dump_states(self, train=True):
        """
        Pickles and saves all the datastreams
        """
        prefix = self.config.experiment_dir_name+"/datastreams/"
        try:
            os.mkdir(prefix)
        except:
            pass

        prefix += "{}.pickle"

        if train:
            self.train_accuracy.dump(prefix.format("train_accuracy"))
            self.train_epochs.dump(prefix.format("train_epochs"))
            self.train_loss.dump(prefix.format("train_loss"))
            self.train_confusion_matrix.dump(prefix.format("train_confusion_matrix"))
        else:
            self.val_accuracy.dump(prefix.format("val_accuracy"))
            self.val_epochs.dump(prefix.format("val_epochs"))
            self.val_loss.dump(prefix.format("val_loss"))
            self.val_confusion_matrix.dump(prefix.format("val_confusion_matrix"))


    def _plot(self, train=True):
        #The actual plot happens on every buffer flush
        if train:
            # A check to ensure that it doesnt throw errors when
            # the buffer is empty
            if self.train_epochs.get_last() == None:
                print("Empty Buffer in Train. Ignoring...")
                return

            self.train_accuracy_plotter.append_plot(
                self.train_accuracy.get_last(),
                self.train_epochs.get_last()
            )
            _payload = {}
            _payload["train_loss"] = self.train_loss.get_last()
            self.loss_plotter.append_plot_with_dict(
                _payload,
                self.train_epochs.get_last()
            )
        else:
            # A check to ensure that it doesnt throw errors when
            # the buffer is empty
            if self.val_epochs.get_last() == None:
                print("Empty Buffer in Val. Ignoring...")
                return

            self.val_accuracy_plotter.append_plot(
                self.val_accuracy.get_last(),
                self.val_epochs.get_last()
            )
            _payload = {}
            _payload["val_loss"] = self.val_loss.get_last()
            self.loss_plotter.append_plot_with_dict(
                _payload,
                self.val_epochs.get_last()
            )
            last_confusion_matrix = self.val_confusion_matrix.get_last()
            # Normalize confusion matrix
            if self.config.normalize_confusion_matrix:
                #last_confusion_matrix = last_confusion_matrix.astype('float')/last_confusion_matrix.sum(axis=1)
                last_confusion_matrix = normalize(last_confusion_matrix)
            self.val_confusion_matrix_plotter.update_plot(
                last_confusion_matrix
            )


def main():
    monitor = TortillaMonitor(topk=(1,5,), plotter=None, classes=[])

if __name__ == "__main__":
    main()
