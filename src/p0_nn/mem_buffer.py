from math import ceil
import torch
from torch.utils.data import Dataset, DataLoader

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

if __name__ == "__main__":
    bs = 2
    buffer = MemBuffer(max_size=5)
    print('\n-------------- filling buffer')
    for i in range(7):
        buffer.add(i, i+100)
        print(buffer.x, buffer.y)
    dl = DataLoader(
        dataset=buffer,
        batch_size=bs,
        shuffle=True,
        generator=torch.Generator(device='cpu'),
    )
    print('\n-------------- dataloading')
    for i in range(3):
        data = next(iter(dl))
        print('data: ', data)
