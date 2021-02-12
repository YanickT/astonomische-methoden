import torch
import torch.nn as nn
import torch.nn.functional as F


class Network:

    """
     Simple neural network for the estimation of the redshift for astronomical objects (quasar and galaxies)
    """

    def __init__(self, lr=1e-3):
        """
        Initalize Classifier network.
        :param lr: float = learning rate
        :param momentum: float = momentum for the optimizer
        """
        self.lr = lr
        self.f_loss = F.mse_loss

        self.network = nn.Sequential(
            nn.Linear(5, 5, bias=True),
            nn.LeakyReLU(),
            nn.Linear(5, 15, bias=True),
            nn.Sigmoid(),
            nn.Linear(15, 30, bias=True),
            nn.Sigmoid(),
            nn.Linear(30, 15, bias=True),
            nn.Sigmoid(),
            nn.Linear(15, 1))

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)

    def train(self, eps, train_data, test_data, patience=5):
        """
        Train the Network.
        :param patience: int = Patience of the early stopping
        :param eps: int = Number of training episodes
        :param train_data: torch.Dataloader = Dataloader for the training data
        :param test_data:torch.Dataloader = Dataloader for the test data
        :return: List[float] = List of accuracy(episode)
        """
        losses = []
        for ep in range(eps):
            print(ep)
            self.network.train()
            for input_data, output_expected in train_data:
                self.calc_loss(input_data, output_expected, self.optimizer)

            self.network.eval()
            with torch.no_grad():
                loss = sum([self.calc_loss(input_data, output_expected) for input_data, output_expected in test_data])

            losses.append(loss)
            if self.early_stop(losses[-patience:]):
                return losses
        return losses

    def calc_loss(self, input_data, output_expected, opt=None):
        """
        Calculate the total loss for the prediction
        :param input_data: torch.Tensor = Data to test
        :param output_expected: torch.Tensor = Correct answers for data to test
        :param opt: torch.optim = Optimizer to train network
        :return: float = loss of the data
        """
        loss = self.f_loss(self.network(input_data), output_expected)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item()

    def predict(self, input_data):
        self.network.eval()
        with torch.no_grad():
            return self.network(input_data)

    @staticmethod
    def early_stop(losses):
        return all([losses[0] <= loss for loss in losses]) and len(losses) > 1


