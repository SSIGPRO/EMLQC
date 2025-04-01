from torch import nn

class NN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self) 
        self.nn = nn.Sequential(
                nn.Linear(4, 10),
                nn.ReLU(),
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.Linear(10, 4),
                )

    def forward(self, x):
        return self.nn(x)

