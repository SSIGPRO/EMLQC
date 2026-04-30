import gymnasium as gym
from time import sleep
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader 
from numpy import sin, cos
from math import ceil
import matplotlib.pyplot as plt

class MemBuffer(Dataset):
    def __init__(self, size=0, max_size=1e3, dropout_size=None):
        self.size = size
        self.max_size = max_size
        if dropout_size == None:
            self.dropout = ceil(max_size/10)
        else:
            self.dropout = dropout_size

        self.x = []
        self.y = []

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def add(self, x, y):
        if self.size == self.max_size:
            self.x = self.x[self.dropout:]
            self.y = self.y[self.dropout:]
        else:
            self.size += 1

        self.x.append(x)
        self.y.append(y)
        return


def get_angular(env):
    return torch.tensor(env.unwrapped.state, dtype=torch.float)

def get_cartesian(env):
    s = env.unwrapped.state
    p1 = [
        -env.unwrapped.LINK_LENGTH_1 * cos(s[0]),
        env.unwrapped.LINK_LENGTH_1 * sin(s[0]),
        ]
    p2 = [
        p1[0] - env.unwrapped.LINK_LENGTH_2 * cos(s[0] + s[1]),
        p1[1] + env.unwrapped.LINK_LENGTH_2 * sin(s[0] + s[1]),
        ]

    return torch.tensor(p1+p2, dtype=torch.float)

# add parameters for number of layers, layer size
class NN(nn.Module):
    def __init__(self, env):
        nn.Module.__init__(self)
        # add env
        self.env = env

        # add n_layers, layer_size, lr, n_iter, max_epochs
        self.n_layers = n_layers
        self.layer_size = layer_size

        # NN 
        self.nn = nn.Sequential(
                nn.Linear(4, 5),
                nn.ReLU(),
                nn.Linear(5, 4),
                nn.ReLU(),
                )

        return

    def forward(self, x):
        return self.nn(x)

# -----------------------------------------------------
# training functions

# add train iter
def train_iter(self):
    return 

# add train epoch
def train_epoch(self):
    return

# add fn to print training and val losses
def plot_losses(self, losses):

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Andamento della loss")
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    return


if __name__ == '__main__':
    # control params 
    max_it = 100

    # mem buffer
    bs = 2
    buffer = MemBuffer(max_size=5)
    
    print('\n-------------- filling buffer')
    for i in range(7):
        buffer.add(i, i+100)
        print(buffer.x, buffer.y)
    
    dl = DataLoader(dataset=buffer, batch_size=bs, shuffle=True, generator=torch.Generator(device='cpu'))

    print('\n-------------- dataloading')
    for i in range(3):
        data = next(iter(dl))
        print('data: ', data)



    env = gym.make('Acrobot-v1', render_mode='human')
    observation, info = env.reset()

    episode_over = False
    it = 0
    while (not episode_over) and (it < max_it):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        angles = get_angular(env)
        goal = get_cartesian(env)

        episode_over = terminated or truncated
        it += 1
        sleep(0.15)
    env.close()
