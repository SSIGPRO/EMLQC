import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from numpy import sin, cos


# Funzione per ottenere le posizioni angolari (input)
def get_angular(env):
    return torch.tensor(env.unwrapped.state, dtype=torch.float)


# Funzione per ottenere le coordinate cartesiane (output)
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


# Definizione della rete neurale
class ForwardKinematicsNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ForwardKinematicsNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),  # Primo strato (input → hidden)
            nn.ReLU(),
            nn.Linear(64, 64),  # Secondo strato (hidden → hidden)
            nn.ReLU(),
            nn.Linear(64, output_size)  # Strato finale (hidden → output)
        )

    def forward(self, x):
        return self.model(x)


# Generazione del dataset
def generate_dataset(env, samples=1000):
    dataset_x = []
    dataset_y = []
    for _ in range(samples):
        env.reset()
        angles = get_angular(env)
        coords = get_cartesian(env)
        dataset_x.append(angles)
        dataset_y.append(coords)
    return torch.stack(dataset_x), torch.stack(dataset_y)


# Addestramento del modello
def train_model(model, dataloader, optimizer, loss_fn, epochs=500):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")


# Test del modello
def test_model(model, env, steps=10):
    model.eval()
    for _ in range(steps):
        env.reset()
        angles = get_angular(env)
        goal = get_cartesian(env)
        pos = model(angles)
        loss = torch.norm(pos - goal).item()
        print(f"Angoli: {angles.numpy()}, Previsto: {pos.detach().numpy()}, Reale: {goal.numpy()}, Loss: {loss}")


if __name__ == "__main__":
    # Inizializza l'ambiente simulato
    env = gym.make('Acrobot-v1')

    # Genera il dataset
    print("Generazione del dataset...")
    X, Y = generate_dataset(env, samples=1000)
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Inizializza il modello
    input_size = X.size(1)  # Dimensione input (numero di angoli)
    output_size = Y.size(1)  # Dimensione output (coordinate cartesiane)
    model = ForwardKinematicsNN(input_size, output_size)

    # Configura l'ottimizzatore e la funzione di perdita
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Addestramento
    print("Inizio addestramento...")
    train_model(model, dataloader, optimizer, loss_fn, epochs=500)

    # Test del modello
    print("Test del modello...")
    test_model(model, env)

    env.close()

