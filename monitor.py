#!/usr/bin/env python
from utils import accuracy
import numpy as np
from sklearn.metrics import confusion_matrix

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
    def __init__(self, plotter=None, topk=(1,5), classes=[]):
        self.plotter = plotter
        self.topk = topk
        self.classes = classes

        self._init_data_gatherers()

    def _init_data_gatherers(self):
        self.train_accuracy = []
        self.val_accuracy = []

        self.train_epochs = []
        self.val_epochs = []

        self.train_loss = []
        self.val_loss = []

        self.train_loss_buffer = []
        self.val_loss_buffer = []

        self.train_accuracy_buffer = []
        self.val_accuracy_buffer = []

        self.train_epoch_buffer = []
        self.val_epoch_buffer = []

        class_len = len(self.classes)
        self.train_confusion_matrix_buffer = np.zeros((class_len, class_len))
        self.val_confusion_matrix_buffer = np.zeros((class_len, class_len))

    def _compute_and_register_stats(self, epoch, outputs, labels, loss, train=True):
        _accuracy = accuracy(outputs, labels, topk=self.topk)
        _accuracy = np.array([x.data[0] for x in _accuracy])

        if train:
            accuracy_buffer = self.train_accuracy_buffer
            epoch_buffer = self.train_epoch_buffer
            loss_buffer = self.train_loss_buffer
        else:
            accuracy_buffer = self.val_accuracy_buffer
            epoch_buffer = self.val_epoch_buffer
            loss_buffer = self.val_loss_buffer

        accuracy_buffer.append(_accuracy)
        epoch_buffer.append(epoch)
        loss_buffer.append(loss.data[0])

        #TODO: Add plotter function calls
        outputs = outputs.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        outputs = outputs.argmax(axis=1)

        _batch_confusion_matrix = confusion_matrix(labels, outputs, labels=range(len(self.classes)))

        self.train_confusion_matrix_buffer += _batch_confusion_matrix

        print(self.train_confusion_matrix_buffer, self.train_confusion_matrix_buffer.mean())

    def _flush_stats(self, train=True):
        """
            Processes all the stats in the buffer
            and adds them to the main log
        """
         buffer_mean = np.mean(_buffer, axis=1)
         # Append buffer to main datalog
         # Plot diff




def main():
    monitor = TortillaMonitor(topk=(1,5,), plotter=None, classes=[])

if __name__ == "__main__":
    main()
