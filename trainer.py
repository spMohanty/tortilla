import math
import torchvision
from torchvision import datasets, models, transforms
import torch

class TortillaTrainer:
    def __init__(self,  dataset, model, loss,
                        optimizer=None, monitor=None,
                        config=None, verbose=True):
        """
            A wrapper class for all the training requirements of tortilla.
        """
        self.dataset = dataset
        self.train_epochs = 0
        self.val_epochs = 0

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.monitor = monitor
        self.config = config
        self.verbose = verbose

        self.train_step_counter = 0

    def _compute_and_register_stats(self, outputs, labels, loss, train=True):
        epoch_pointer = self.dataset.percent_complete(train=train)
        if train:
            epoch_pointer += self.train_epochs
        else:
            # Use a integral epoch value in case of
            # val set
            epoch_pointer = float(self.val_epochs)

        if self.monitor:
            self.monitor._compute_and_register_stats(
                epoch_pointer,
                outputs,
                labels,
                loss,
                train=train)
            if train:
                # Flush every `train_flush_frequency` steps in case of training
                flush_frequency = \
                    max(1, \
                        math.floor((float(self.dataset.train_dataset.total_images) \
                        /self.config.batch_size) \
                        /self.config.train_flush_per_epoch))

                if self.train_step_counter % flush_frequency == 0:
                    self.monitor._flush_stats(train=train)
                    self.train_step_counter = 0
                self.train_step_counter += 1
        else:
            raise("TortillaMonitor not defined")

    def _predict(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def _step(self, train=True, use_gpu=False):
        """
            Do a single step of training/validation
        """
        images, labels, end_of_epoch = self.dataset.get_next_batch(
                                            train=train,
                                            use_gpu=use_gpu
                                            )
        if end_of_epoch:
            if train:
                self.train_epochs += 1
            else:
                self.val_epochs += 1

            # Flush at the end of the epoch in any case
            self.monitor._flush_stats(train=train)
            self.monitor._dump_states(train=train)
            return (False, False, False, False, 100, end_of_epoch)

        # Predict
        outputs = self._predict(images)

        # Compute Loss
        _loss = self.loss(outputs, labels)

        # Compute and log/plot stats
        self._compute_and_register_stats(outputs, labels, _loss, train=train)

        # DEBUG
        im = images[0:5]
        # MEAN = [0.485, 0.456, 0.406]
        # STD = [0.229, 0.224, 0.225]
        # for t in range(3):
        #     im[:,t,:,:] = im[:,t,:,:]*STD[t] + MEAN[t]
        if use_gpu:
            _im = im.data.cpu()
        else:
            _im = im.data
        self.monitor.images_plotter.update_images(_im)

        if train:
            # Adjust weights
            self.model.zero_grad()
            _loss.backward()
            self.optimizer.step()


        percent_complete = self.dataset.percent_complete(train=train)
        return  (
                    _loss,
                    images,
                    labels,
                    outputs,
                    percent_complete,
                    end_of_epoch
                )

    def train_step(self, use_gpu=False):
        return self._step(train=True, use_gpu=use_gpu)

    def val_step(self, use_gpu=False):
        return self._step(train=False, use_gpu=use_gpu)
