import gymnasium as gym
from time import sleep
import torch
import torch.nn as nn
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
    def __init__(self, env, n_layers=2, layer_size=32, lr=1e-3, n_iter=100, max_epochs=100):
        super().__init__()

        # environment and hyperparameters
        self.env = env
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.lr = lr
        self.n_iter = n_iter
        self.max_epochs = max_epochs

        # input / output sizes from environment spaces
        input_dim = int(torch.tensor(env.observation_space.shape).prod().item())
        if isinstance(env.action_space, gym.spaces.Discrete):
            output_dim = env.action_space.n
        else:
            output_dim = int(torch.tensor(env.action_space.shape).prod().item())

        # build dynamic hidden layers
        layers = [nn.Linear(input_dim, self.layer_size), nn.ReLU()]
        for _ in range(max(0, self.n_layers - 1)):
            layers.append(nn.Linear(self.layer_size, self.layer_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.layer_size, output_dim))

        self.nn = nn.Sequential(*layers)

        # define loss and optimizer for training
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

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

    # instantiate a parametric neural network
    model = NN(env, n_layers=3, layer_size=32, lr=1e-3, n_iter=100, max_epochs=100)
    print('Created NN with {} layers, layer size {}, lr {}'.format(model.n_layers, model.layer_size, model.lr))

    # example forward pass using the current environment state
    sample_input = get_angular(env).unsqueeze(0)
    sample_output = model(sample_input)
    print('Sample network output:', sample_output)

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
