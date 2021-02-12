import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_SIMPLE_CNN(nn.Module):
    """
    Hand-tuned architecture for ColoredMNISTEasier.
    """
    n_outputs = 180

    def __init__(self, input_shape, out_dim):
        super(MNIST_SIMPLE_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 5, kernel_size = 4, stride=(2, 2))
        self.conv2 = nn.Conv2d(5, 5, kernel_size = 3, stride=(2, 2))
        self.classifier_out_dim = out_dim
        if out_dim is not None:
            self.d_out = out_dim
            self.final = nn.Linear(self.n_outputs, self.d_out)
        else:
            self.d_out = self.n_outputs

    def features(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(-1, self.n_outputs)
        return x

    def forward(self, x):
        x = self.features(x)
        if self.classifier_out_dim is not None:
            x = self.final(x)

        return x