
class TortillaTrainer:
    def __init__(self, dataset, model, loss, optimizer=None, plotter=None):
        """
            A wrapper class for all the training requirements of tortilla.
        """
        self.dataset = dataset
        self.epochs = 0

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.plotter = plotter


    def train_step(self):
        """
            Do a single step of training
        """
        images, labels, end_of_epoch = self.dataset.get_next_batch(train=True)
