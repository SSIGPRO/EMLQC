from math import ceil
from time import sleep

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from matplotlib.pylab import shuffle
from numpy import cos, sin
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from mem_buffer import MemBuffer, data


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

    return torch.tensor(p1 + p2, dtype=torch.float)


# add parameters for number of layers, layer size
class NN(nn.Module):
    def __init__(self, env, n_layers, layer_size):
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
    def train_iter(self, dataset, batch_size, learning_rate, num_epochs):
        optimizer = Adam(self.parameters(), lr=learning_rate)
        for i in range(num_epochs):
            self.train_epoch(dataset, batch_size, optimizer)
        return

    # add train epoch
    def train_epoch(self, dataset, batch_size, optimizer):
        self.train()
        total_loss = 0
        criterion = nn.MSELoss()
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator(device="cuda"),
        )
        for batch in data_loader:
            output = self.forward(batch[0].to_device("cuda"))
            loss = criterion(output, batch[1].to_device("cuda"))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return

    # add fn to print training and val losses
    def plot_losses(self):
        return


if __name__ == "__main__":
    # control params
    max_it = 100

    env = gym.make("Acrobot-v1", render_mode="human")
    observation, info = env.reset()

    episode_over = False
    it = 0
    buffer = MemBuffer(max_size=max_it)
    batch_size = 10
    while (not episode_over) and (it < max_it):
        action = (
            env.action_space.sample()
        )  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        angles = get_angular(env)
        goal = get_cartesian(env)
        buffer.add(angles, goal)

        episode_over = terminated or truncated
        it += 1
        sleep(0.15)
    env.close()
