#!/usr/bin/env python
from utils import accuracy
import numpy as np
from sklearn.metrics import confusion_matrix
from datastream import TortillaDataStream

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
    def __init__(self, plotter=None, topk=(1,5), classes=[], use_gpu=False):
        self.plotter = plotter
        self.topk = topk
        self.classes = classes
        self.use_gpu = use_gpu

        self._init_data_gatherers()

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
                                name="train-confusion_matrix"
                                )
        self.val_confusion_matrix = TortillaDataStream(
                                name="val-confusion_matrix"
                                )

    def compute_confusion_matrix(self, outputs, labels):
        _, pred_top_1 = outputs.topk(1, 1, True, True)#compute top-1 predictions
        pred_top_1 = pred_top_1.t()
        pred_top_1 = pred_top_1.eq(labels.view((1,-1)).expand_as(pred_top_1))
        pred_top_1 = pred_top_1.squeeze(0)
        if self.use_gpu:
            _labels = labels.data.cpu().numpy()
            _preds =  pred_top_1.data.cpu().numpy()
        else:
            _labels = labels.data.numpy()
            _preds =  pred_top_1.data.numpy()

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


def main():
    monitor = TortillaMonitor(topk=(1,5,), plotter=None, classes=[])

if __name__ == "__main__":
    main()
