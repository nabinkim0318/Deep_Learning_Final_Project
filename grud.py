import torch
import torch.nn as nn
import numpy as np

class GRU_D_Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU_D_Cell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weights for the update gate
        self.weight_zx = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_zh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_zm = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.bias_z = nn.Parameter(torch.Tensor(hidden_size))

        # Weights for the reset gate
        self.weight_rx = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_rh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_rm = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.bias_r = nn.Parameter(torch.Tensor(hidden_size))

        # Weights for the new gate
        self.weight_nx = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_nh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_nm = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.bias_n = nn.Parameter(torch.Tensor(hidden_size))

        # Weights for the decay terms
        self.decay_Wx = nn.Parameter(torch.Tensor(input_size, input_size))
        self.decay_bx = nn.Parameter(torch.Tensor(input_size))
        self.decay_Wh = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.decay_bh = nn.Parameter(torch.Tensor(hidden_size))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        # Initialization method (Xavier for this example)
        nn.init.xavier_uniform_(self.weight_zx)
        nn.init.xavier_uniform_(self.weight_zh)
        nn.init.xavier_uniform_(self.weight_zm)
        nn.init.xavier_uniform_(self.weight_rx)
        nn.init.xavier_uniform_(self.weight_rh)
        nn.init.xavier_uniform_(self.weight_rm)
        nn.init.xavier_uniform_(self.weight_nx)
        nn.init.xavier_uniform_(self.weight_nh)
        nn.init.xavier_uniform_(self.weight_nm)
        nn.init.xavier_uniform_(self.decay_Wx)
        nn.init.xavier_uniform_(self.decay_Wh)

        self.bias_z.data.fill_(0)
        self.bias_r.data.fill_(0)
        self.bias_n.data.fill_(0)
        self.decay_bx.data.fill_(0)
        self.decay_bh.data.fill_(0)

    def forward(self, x, delta, m, h_prev, x_last_observed, empirical_mean):
        gamma_x = torch.exp(-torch.max(torch.zeros_like(delta), torch.matmul(delta, self.decay_Wx.t()) + self.decay_bx))
        gamma_h = torch.exp(-torch.max(torch.zeros_like(h_prev), torch.matmul(delta, self.decay_Wh.t()) + self.decay_bh))

        x_nan_to_num = torch.nan_to_num(x)
        x_hat = m * x_nan_to_num + (1 - m) * (gamma_x * x_last_observed + (1 - gamma_x) * empirical_mean)
        h_hat = gamma_h * h_prev

        z = self.sigmoid(torch.matmul(x_hat, self.weight_zx.t()) + torch.matmul(h_hat, self.weight_zh.t()) + torch.matmul(m, self.weight_zm.t()) + self.bias_z)
        r = self.sigmoid(torch.matmul(x_hat, self.weight_rx.t()) + torch.matmul(h_hat, self.weight_rh.t()) + torch.matmul(m, self.weight_rm.t()) + self.bias_r)
        n = self.tanh(torch.matmul(x_hat, self.weight_nx.t()) + torch.matmul(r * h_hat, self.weight_nh.t()) + torch.matmul(m, self.weight_nm.t()) + self.bias_n)
        h_next = (1 - z) * h_hat + z * n

        return h_next
    
class GRU_D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(GRU_D, self).__init__()
        self.gru_d_cell = GRU_D_Cell(input_size, hidden_size)

        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, delta, M, last_observation, empirical_mean):
        h_t = torch.zeros(X.size(0), self.gru_d_cell.hidden_size, device=X.device)
        for t in range(X.size(1)):
            h_t = self.gru_d_cell(X[:, t, :], delta[:, t, :], M[:, t, :], h_t, last_observation[:, t, :], empirical_mean)
        output = self.sigmoid(self.output_layer(h_t))
        return output.squeeze()
    
def preprocess_dataset(X):
    num_samples, num_timepoints, num_variables = X.shape[0], X.shape[1], X.shape[2]
    M = (~np.isnan(X)).astype(int)
    delta = np.zeros_like(X)
    last_observation = np.full_like(X, np.nan)
    for sample in range(num_samples):
        for var in range(num_variables):
            for time in range(num_timepoints):
                if (time == 0) or (np.isnan(last_observation[sample, time - 1, var])):
                    future_values = X[sample, time:, var]
                    next_obs = future_values[~np.isnan(future_values)]
                    if next_obs.size > 0:
                        last_observation[sample, time, var] = next_obs[0]
                else:
                    last_observation[sample, time, var] = X[sample, time, var] if M[sample, time, var] == 1 else last_observation[sample, time - 1, var]

                if time == 0:
                    delta[sample, time, var] = 0
                else:
                    if M[sample, time - 1, var] == 1:
                        delta[sample, time, var] = 1
                    else:
                        delta[sample, time, var] = 1 + delta[sample, time - 1, var]
    empirical_mean = np.nanmean(X, axis=1)
    means = np.nanmean(empirical_mean, axis=0)
    for i in range(empirical_mean.shape[1]):
        mask = np.isnan(empirical_mean[:, i])
        empirical_mean[:, i][mask] = means[i]
        last_observation[:, :, i][mask] = means[i]
    return X, delta, M, last_observation, empirical_mean
    
