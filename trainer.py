
from utils import accuracy, append_val

class TortillaTrainer:
    def __init__(self,  dataset, model, loss,
                        optimizer=None, plotter=None,
                        verbose=True):
        """
            A wrapper class for all the training requirements of tortilla.
        """
        self.dataset = dataset
        self.epochs = 0

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.plotter = plotter
        self.verbose = verbose

        self.running_stats = {
            "train" : {
            },
            "val" : {
            }
        }

    def _predict(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def _compute_and_log_stats(self, outputs, labels, loss, train=True):
        epoch_pointer = self.dataset.get_current_pointer(train=train)
        if train:
            epoch_pointer += self.epochs

        _accuracy = accuracy(outputs, labels, topk=(1,5,))
        _accuracy = [x.data[0] for x in _accuracy]

        if train:
            stats = self.running_stats["train"]
        else:
            stats = self.running_stats["val"]

        stats = append_val(stats, "top_1", _accuracy[0])
        stats = append_val(stats, "top_5", _accuracy[1])
        stats = append_val(stats, "loss", loss.data[0])

        print(stats)

    def _step(self, train=True, use_gpu=False):
        """
            Do a single step of training/validation
        """
        images, labels, end_of_epoch = self.dataset.get_next_batch(
                                            train=train,
                                            use_gpu=use_gpu
                                            )
        # Predict
        outputs = self._predict(images)

        # Compute Loss
        _loss = self.loss(outputs, labels)

        # Compute and log/plot stats
        self._compute_and_log_stats(outputs, labels, _loss, train=train)

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
