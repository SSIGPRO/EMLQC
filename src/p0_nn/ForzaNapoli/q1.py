import gymnasium as gym
from time import sleep
import torch
from torch.utils.data import Dataset, DataLoader 
import torch.nn as nn
from numpy import sin, cos
from math import ceil
import matplotlib.pyplot as plt
import torch.optim as optim

# LdSR: copy past instead of import. -1 code quality 
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
    def __init__(self, env, n_layers=2, layer_size=32, lr=1e-3, n_iter=100, max_epochs=100):
        super().__init__()

        # LdSR: env, losses, opt and loss should not be part of the model. -1 code quality
        # environment and hyperparameters
        self.env = env
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.lr = lr
        self.n_iter = n_iter
        self.max_epochs = max_epochs
        self.losses = []
        self.val_losses= []

        # LdSR: using ang vel for model. -1 project
        input_dim = 4
        output_dim = 4
        # build dynamic hidden layers
        layers = [nn.Linear(input_dim, self.layer_size), nn.ReLU()]
        for _ in range(max(0, self.n_layers - 1)):
            layers.append(nn.Linear(self.layer_size, self.layer_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.layer_size, output_dim))

        self.nn = nn.Sequential(*layers)

        # define loss and optimizer for training
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.nn(x)

# -----------------------------------------------------
# training functions

# train iter
def train_iter(model, x, y):

    # LdSR: optim step should be done each epoch not every iteration. -1 project 
    model.optimizer.zero_grad()
    outputs = model(x)
    loss= model.criterion(outputs, y)
    loss.backward()
    model.optimizer.step()

    return loss.item()

# train epoch
def train_epoch(model, dataloader):
    model.train()
    total_loss = 0

    for x, y in dataloader:
        loss =  train_iter(model, x, y)
        total_loss += loss

    average_loss= total_loss/len(dataloader)
    model.losses.append(average_loss)
    return average_loss

# fn to print training and val losses
def plot_losses(val_losses, train_losses):

    plt.plot(train_losses, label= 'Validation loss')
    plt.plot(val_losses, label= 'Training loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Andamento delle loss")
    plt.yscale('log')
    plt.grid(True, which= 'both')
    plt.legend()
    plt.show()

    return


if __name__ == '__main__':
    # LdSR: delivery delay. -2 project
    # control params 
    max_it = 100

    # mem buffer
    bs = 2
    buffer = MemBuffer(max_size= 5)

    env = gym.make('Acrobot-v1', render_mode='human')
    observation, info = env.reset()

    # instantiate a parametric neural network
    model = NN(env, n_layers=4, layer_size=128, lr=1e-1, n_iter=100, max_epochs=100)
    print('Created NN with {} layers, layer size {}, lr {}'.format(model.n_layers, model.layer_size, model.lr))

    episode_over = False
    it = 0
    # LdSR: Mem buffer should be fed every epoch. -1 project 
    # LdSR: feeding buffer based on max_it. -1 code quality
    while (not episode_over) and (it < max_it):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        angles = get_angular(env)
        goal = get_cartesian(env)

        print('\n-------------- filling buffer')
        # LdSR: no validation buffe. -1 project
        buffer.add(angles, goal)
        print(buffer.x, buffer.y)

        episode_over = terminated or truncated
        it += 1
        sleep(0.15)

    #Preparing for the training loop
    dl = DataLoader(dataset=buffer, batch_size=bs, shuffle=True, generator=torch.Generator(device='cpu'))

    print('\n-------------- dataloading')
    for i in range(3):
        data = next(iter(dl))
        print('data: ', data)

    # LdSR: no hyperparam exploration. -2 project
    #Training loop
    for epoch in range(model.max_epochs):
        train_loss = train_epoch(model, dl)

    #validation loss
    model.eval() # Modalità valutazione
    total_val_loss = 0
    with torch.no_grad(): 
        for x, y in dl:
            outputs = model(x)
            loss = model.criterion(outputs, y)
            total_val_loss += loss.item()
    v_loss = total_val_loss / len(dl)
    model.val_losses.append(v_loss)

    #validation loss and train loss visualization
    plot_losses(model.val_losses, model.losses)
    env.close()
