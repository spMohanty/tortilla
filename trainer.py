
class TortillaTrainer:
    def __init__(self,  dataset, model, loss,
                        optimizer=None, monitor=None,
                        verbose=True):
        """
            A wrapper class for all the training requirements of tortilla.
        """
        self.dataset = dataset
        self.epochs = 0

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.monitor = monitor
        self.verbose = verbose

    def _compute_and_register_stats(self, outputs, labels, loss, train=True):
        epoch_pointer = self.dataset.get_current_pointer(train=train)
        if train:
            epoch_pointer += self.epochs

        if self.monitor:
            self.monitor._compute_and_register_stats(
                epoch_pointer,
                outputs,
                labels,
                loss,
                train=train)
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
            return (False, False, False, False, end_of_epoch)

        # Predict
        outputs = self._predict(images)

        # Compute Loss
        _loss = self.loss(outputs, labels)

        # Compute and log/plot stats
        self._compute_and_register_stats(outputs, labels, _loss, train=train)

        if train:
            # Adjust weights
            self.model.zero_grad()
            _loss.backward()
            self.optimizer.step()

        return  (
                    _loss,
                    images,
                    labels,
                    outputs,
                    end_of_epoch
                )

    def train_step(self, use_gpu=False):
        return self._step(train=True, use_gpu=use_gpu)

    def val_step(self, use_gpu=False):
        return self._step(train=False, use_gpu=use_gpu)
