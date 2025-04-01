import gymnasium as gym
from time import sleep
from torch import nn
import torch
from numpy import sin, cos
from math import ceil
import matplotlib.pyplot as plt

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
def plot_losses(self):
    return


if __name__ == '__main__':
    # control params 
    max_it = 100

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
