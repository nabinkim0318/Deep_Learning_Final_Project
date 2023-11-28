import torch

class GRUDdataset(torch.utils.data.Dataset):
    def __init__(self, X, delta, M, last_observation, empirical_mean, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.delta = torch.tensor(delta, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.float32)
        self.last_observation = torch.tensor(last_observation, dtype=torch.float32)
        self.empirical_mean = torch.tensor(empirical_mean, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.delta[idx], self.M[idx], self.last_observation[idx], self.empirical_mean[idx], self.Y[idx]
    
class GRUDFdataset(GRUDdataset):
    def __init__(self, X, delta, delta_future, M, last_observation, next_observation, empirical_mean, Y):
        super().__init__(X, delta, M, last_observation, empirical_mean, Y)
        self.delta_future = torch.tensor(delta_future, dtype=torch.float32)
        self.next_observation = torch.tensor(next_observation, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.delta[idx], self.delta_future[idx], self.M[idx], self.last_observation[idx], self.next_observation[idx], \
            self.empirical_mean[idx], self.Y[idx]