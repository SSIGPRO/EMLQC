from itertools import product
from math import ceil
from time import sleep

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from numpy import cos, sin
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from mem_buffer import MemBuffer


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


class NN(nn.Module):
    def __init__(self, env, n_layers, layer_size):
        nn.Module.__init__(self)
        self.env = env
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

    def train_iter(self, dataset, batch_size, learning_rate, num_epochs):
        optimizer = Adam(self.parameters(), lr=learning_rate)
        for i in range(num_epochs):
            self.train_epoch(dataset, batch_size, optimizer)
        return

    def train_epoch(self, dataset, batch_size, optimizer):
        self.train()
        total_loss = 0
        criterion = nn.MSELoss()
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator(device="cpu"),
        )
        for batch in data_loader:
            output = self.forward(batch[0])
            loss = criterion(output, batch[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return

    def plot_losses(self):
        return

    # add tuning function
    def tune_hyperparameters(self, dataset, param_grid=None):
        # default grid
        if param_grid is None:
            param_grid = {
                'learning_rate': [0.01, 0.001, 0.0001],
                'batch_size': [8, 16, 32],
                'num_epochs': [5, 10],
            }
        best_loss = float('inf')
        best_params = {}
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        # try all combinations
        for combo in product(*values):
            params = dict(zip(keys, combo))
            self._reset_weights()
            self.train_iter(
                dataset=dataset,
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                num_epochs=params['num_epochs'],
            )
            val_loss = self._compute_loss(dataset)
            print('params:', params, 'loss:', val_loss)
            # keep best
            if val_loss < best_loss:
                best_loss = val_loss
                best_params = params
        return best_params, best_loss

    #add reset weights
    def _reset_weights(self):
        for layer in self.nn:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        return

    #add compute loss
    def _compute_loss(self, dataset):
        self.eval()
        criterion = nn.MSELoss()
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=32,
            shuffle=False,
            generator=torch.Generator(device="cpu"),
        )
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                output = self.forward(batch[0])
                total_loss += criterion(output, batch[1]).item()
        return total_loss / len(data_loader)


if __name__ == '__main__':
    # control params
    max_it = 100
    env = gym.make('Acrobot-v1', render_mode='human')
    observation, info = env.reset()
    episode_over = False
    it = 0
    buffer = MemBuffer(max_size=max_it)
    batch_size = 10
    while (not episode_over) and (it < max_it):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        angles = get_angular(env)
        goal = get_cartesian(env)
        buffer.add(angles, goal)
        episode_over = terminated or truncated
        it += 1
    model = NN(env, n_layers=2, layer_size=5)
    best_params, best_loss = model.tune_hyperparameters(buffer)