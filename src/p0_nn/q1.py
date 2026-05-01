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
        
        # track losses for plotting
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        return self.nn(x)

# -----------------------------------------------------
# training functions

def train_iter(self, x, y):
    """Single training iteration: forward pass, loss computation, backward, step"""
    self.train()
    self.optimizer.zero_grad()
    
    # forward pass
    output = self(x)
    
    # compute loss
    loss = self.loss_fn(output, y)
    
    # backward pass and optimization step
    loss.backward()
    self.optimizer.step()
    
    return loss.item()


def eval_iter(self, x, y):
    """Evaluation iteration without weight update"""
    self.eval()
    with torch.no_grad():
        output = self(x)
        loss = self.loss_fn(output, y)
    
    return loss.item()


def train_epoch(self, train_loader, val_loader=None):
    """Train for one complete epoch"""
    # training phase
    train_loss_sum = 0.0
    for _ in range(self.n_iter):
        x_batch, y_batch = next(iter(train_loader))
        loss = self.train_iter(x_batch, y_batch)
        train_loss_sum += loss
    
    avg_train_loss = train_loss_sum / self.n_iter
    self.train_losses.append(avg_train_loss)
    
    # validation phase
    if val_loader is not None:
        val_loss_sum = 0.0
        num_batches = 0
        for x_batch, y_batch in val_loader:
            loss = self.eval_iter(x_batch, y_batch)
            val_loss_sum += loss
            num_batches += 1
        
        avg_val_loss = val_loss_sum / num_batches
        self.val_losses.append(avg_val_loss)
        
        return avg_train_loss, avg_val_loss
    
    return avg_train_loss, None


def plot_losses(self):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    
    if self.train_losses:
        plt.plot(self.train_losses, label='Train Loss', marker='o')
    
    if self.val_losses:
        plt.plot(self.val_losses, label='Validation Loss', marker='s')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


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

    # collect training data from environment
    print('\nCollecting training data...')
    train_data_x = []
    train_data_y = []
    
    episode_over = False
    it = 0
    while (not episode_over) and (it < max_it):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        angles = get_angular(env)
        goal = get_cartesian(env)
        
        train_data_x.append(angles)
        train_data_y.append(goal)

        episode_over = terminated or truncated
        it += 1
        sleep(0.15)
    
    # convert to tensors
    train_x = torch.stack(train_data_x)
    train_y = torch.stack(train_data_y)
    print(f'Collected {len(train_x)} training samples')
    
    # training loop
    print('\nTraining...')
    for epoch in range(model.max_epochs):
        total_loss = 0.0
        for i in range(0, len(train_x), 8):
            batch_x = train_x[i:i+8]
            batch_y = train_y[i:i+8]
            loss = model.train_iter(batch_x, batch_y)
            total_loss += loss
        
        avg_loss = total_loss / ceil(len(train_x) / 8)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{model.max_epochs}, Loss: {avg_loss:.6f}')
    
    print('Training completed!')
    print(f'Final train loss: {model.train_losses[-1]:.6f}')
    
    # plot losses
    model.plot_losses()
    
    env.close()
