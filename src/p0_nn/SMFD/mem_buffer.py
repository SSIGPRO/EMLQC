from math import ceil

from torch.utils.data import Dataset


class MemBuffer(Dataset):
    def __init__(self, size=0, max_size=1e3, dropout_size=None):
        self.size = size
        self.max_size = max_size
        if dropout_size is None:
            self.dropout = ceil(max_size / 10)
        else:
            self.dropout = dropout_size

        self.x = []
        self.y = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def add(self, x, y):
        if self.size == self.max_size:
            self.x = self.x[self.dropout :]
            self.y = self.y[self.dropout :]
            self.size -= self.dropout

        self.x.append(x)
        self.y.append(y)
        self.size += 1
        return
