import numpy as np


class FakeDataset(object):
    def __init__(self):
        self.rand_state = np.random.RandomState()

    def next_batch(self, batch_size):
        batch_xs = self.rand_state.rand(batch_size, 784).astype(np.float32)
        labels = self.rand_state.randint(low=0, high=9, size=batch_size)
        batch_ys = np.eye(10)[labels, :]
        return (batch_xs, batch_ys)


class FakeMNIST(object):
    def __init__(self, train_dir=None, random_seed=None):
        self.train = FakeDataset()
        if random_seed is not None:
            self.reset(random_seed)

    def reset(self, random_seed):
        self.train.rand_state.seed(random_seed)
