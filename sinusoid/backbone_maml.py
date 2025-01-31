import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pdb


# defining our sine-net
class SineNet(nn.Module):
    def __init__(self):
        super(SineNet, self).__init__()
        self.net = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(1, 40)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(40, 40)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(40, 1))
        ]))

    # I implemented argforward() so that I could use a set of custom weights for evaluation.
    # This is important for the "inner loop" in MAML where you temporarily update the weights
    # of the network for a task to calculate the meta-loss and then reset them for the next meta-task.

    def argforward(self, x, weights):
        x = F.linear(x, weights[0], weights[1])
        x = F.relu(x)
        x = F.linear(x, weights[2], weights[3])
        x = F.relu(x)
        x = F.linear(x, weights[4], weights[5])
        return x

