from itertools import product

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from mem_buffer import MemBuffer
from numpy import cos, sin
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader


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
    def __init__(self, n_layers, layer_size, device):
        nn.Module.__init__(self)

        # NN
        # LdSR: input should be 2, the velocity is useless. -1 project
        layers = [nn.Linear(4, layer_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(layer_size, layer_size), nn.ReLU()]
        layers.append(nn.Linear(layer_size, 4))
        self.nn = nn.Sequential(*layers)
        self.to(device)

        return

    def forward(self, x):
        return self.nn(x)

    # LdSR: this will break, depending on where are the parameters, -1 code quality
    @property
    def device(self):
        # Sempre in sync, anche dopo .to(device)
        return next(self.parameters()).device

    # -----------------------------------------------------
    # training functions

    # add train iter
    def train_iter(self, train_set, val_set, batch_size, learning_rate, num_epochs):
        # LdSR: creating the optimizer in each training iteration, it wont save the states and backprop correctly. -1 code quality 
        optimizer = Adam(self.parameters(), lr=learning_rate)
        train_losses = []
        val_losses = []
        for _ in range(num_epochs):
            train_loss = self.train_epoch(train_set, batch_size, optimizer)
            val_loss = self.val_epoch(val_set, batch_size)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        return train_losses, val_losses

    # add train epoch
    def train_epoch(self, dataset, batch_size, optimizer):
        self.train()
        total_loss = 0
        criterion = nn.MSELoss()
        # LdSR: Creating a dataloader and iterating on the whole dataset each epoch. Iteration and epoch are inverter. -1 code quality
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator(device="cpu"),
        )
        for batch in data_loader:
            output = self.forward(batch[0].to(self.device))
            loss = criterion(output, batch[1].to(self.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)

    def val_epoch(self, dataset, batch_size):
        # Imposta il modello in modalità valutazione (disabilita Dropout/BatchNorm se presenti)
        self.eval()
        total_loss = 0
        criterion = nn.MSELoss()

        # Durante la validazione non è necessario mescolare i dati (shuffle=False)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            generator=torch.Generator(device="cpu"),
        )

        # Disabilita il calcolo dei gradienti per risparmiare memoria e tempo di calcolo
        with torch.no_grad():
            for batch in data_loader:
                output = self.forward(batch[0].to(self.device))
                loss = criterion(output, batch[1].to(self.device))
                total_loss += loss.item()

        # Restituisce la perdita media di validazione
        return total_loss / len(data_loader)


def plot_losses(train_losses, val_losses):
    # Genera il grafico per confrontare le curve di apprendimento
    plt.figure()
    plt.plot(train_losses, label="Perdita di Addestramento (Training)")
    plt.plot(val_losses, label="Perdita di Validazione (Validation)")
    plt.xlabel("Epoca (Epoch)")
    plt.ylabel("Perdita (MSE Loss)")
    plt.title("Addestramento vs Validazione")
    plt.legend()
    plt.show()


def tune_hyperparameters(train_set, val_set, device, param_grid=None):
    # default grid
    if param_grid is None:
        param_grid = {
            "learning_rate": [0.01, 0.001, 0.0001],
            "batch_size": [8, 16, 32],
            "num_epochs": [5, 10, 20],
            "n_layers": [2, 3, 4, 5],
            "layer_size": [16, 32, 64],
        }
    best_loss = float("inf")
    best_params = {}
    # LdSR: from dict, to list, to product, to dict. this is messy. -1 code quality
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    # try all combinations
    for combo in product(*values):
        params = dict(zip(keys, combo))
        model = NN(params["n_layers"], params["layer_size"], device)
        _, val_losses = model.train_iter(
            train_set=train_set,
            val_set=val_set,
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            num_epochs=params["num_epochs"],
        )
        print("params:", params, "loss:", val_losses[-1])
        # keep best
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            best_params = params
    return best_params, best_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # control params
    max_it = 1000
    human = False

    env = (
        gym.make("Acrobot-v1", render_mode="human") if human else gym.make("Acrobot-v1")
    )

    observation, info = env.reset()

    # LdSR: buffer should be fed every epoch. -1 project
    # LdSR: No policy, uncovered state space. -2 project
    episode_over = False
    it = 0
    train_set = MemBuffer(max_size=max_it)
    val_set = MemBuffer(max_size=max_it / 10)
    while (not episode_over) and (it < max_it):
        action = (
            env.action_space.sample()
        )  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        angles = get_angular(env)
        goal = get_cartesian(env)
        train_set.add(angles, goal)

        episode_over = terminated or truncated
        it += 1
    env.reset()
    episode_over = False
    it = 0
    while (not episode_over) and (it < max_it):
        action = (
            env.action_space.sample()
        )  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        angles = get_angular(env)
        goal = get_cartesian(env)
        val_set.add(angles, goal)

        episode_over = terminated or truncated
        it += 1
    env.close()
    best_params, best_loss = tune_hyperparameters(train_set, val_set, device)
    print(f"Best params: {best_params}")
    print(f"Best loss: {best_loss}")
    # Best params: {'learning_rate': 0.01, 'batch_size': 8, 'num_epochs': 10, 'n_layers': 2, 'layer_size': 64}
    # Best loss: 0.005225589606337822
    model = NN(best_params["n_layers"], best_params["layer_size"], device)
    train_losses, val_losses = model.train_iter(
        train_set,
        val_set,
        best_params["batch_size"],
        best_params["learning_rate"],
        best_params["num_epochs"],
    )
    plot_losses(train_losses, val_losses)
