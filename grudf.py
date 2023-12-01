import torch
import torch.nn as nn
import numpy as np
import grud
import os

class GRU_DF_Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU_DF_Cell, self).__init__()

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
        self.decay_Wx_diag = nn.Parameter(torch.Tensor(input_size))
        self.decay_bx = nn.Parameter(torch.Tensor(input_size))
        self.decay_Wh = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.decay_bh = nn.Parameter(torch.Tensor(hidden_size))

        #Weights for the future decay terms
        self.decay_future_Wx_diag = nn.Parameter(torch.Tensor(input_size))
        self.decay_future_bx = nn.Parameter(torch.Tensor(input_size))

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
        nn.init.uniform_(self.decay_Wx_diag, a=-0.01, b=0.01)
        nn.init.xavier_uniform_(self.decay_Wh)
        nn.init.uniform_(self.decay_future_Wx_diag, a=-0.01, b=0.01)

        self.bias_z.data.fill_(0)
        self.bias_r.data.fill_(0)
        self.bias_n.data.fill_(0)
        self.decay_bx.data.fill_(0)
        self.decay_bh.data.fill_(0)
        self.decay_future_bx.data.fill_(0)

    def forward(self, x, delta, delta_future, m, h_prev, x_last_observed, x_next_observed, empirical_mean):
        decay_Wx = torch.diag(self.decay_Wx_diag)
        decay_future_Wx = torch.diag(self.decay_future_Wx_diag)
        gamma_x = torch.exp(-torch.max(torch.zeros_like(delta), torch.matmul(delta, decay_Wx.t()) + self.decay_bx))
        gamma_h = torch.exp(-torch.max(torch.zeros_like(h_prev), torch.matmul(delta, self.decay_Wh.t()) + self.decay_bh))
        gamma_x_future = torch.exp(-torch.max(torch.zeros_like(delta_future), 
                                              torch.matmul(delta_future, decay_future_Wx.t()) + self.decay_future_bx))
        
        x_nan_to_num = torch.nan_to_num(x)
        x_hat = m * x_nan_to_num + (1 - m) * (gamma_x * x_last_observed + gamma_x_future * x_next_observed + (1 - gamma_x - gamma_x_future) * empirical_mean)
        h_hat = gamma_h * h_prev

        z = self.sigmoid(torch.matmul(x_hat, self.weight_zx.t()) + torch.matmul(h_hat, self.weight_zh.t()) + torch.matmul(m, self.weight_zm.t()) + self.bias_z)
        r = self.sigmoid(torch.matmul(x_hat, self.weight_rx.t()) + torch.matmul(h_hat, self.weight_rh.t()) + torch.matmul(m, self.weight_rm.t()) + self.bias_r)
        n = self.tanh(torch.matmul(x_hat, self.weight_nx.t()) + torch.matmul(r * h_hat, self.weight_nh.t()) + torch.matmul(m, self.weight_nm.t()) + self.bias_n)
        h_next = (1 - z) * h_hat + z * n

        return h_next
    
class GRU_DF(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(GRU_DF, self).__init__()
        self.gru_d_cell = GRU_DF_Cell(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, delta, delta_future, M, last_observation, next_observation, empirical_mean):
        h_t = torch.zeros(X.size(0), self.gru_d_cell.hidden_size, device=X.device)
        for t in range(X.size(1)):
            h_t = self.gru_d_cell(X[:, t, :], delta[:, t, :], delta_future[:, t, :], M[:, t, :], h_t, last_observation[:, t, :], next_observation[:, t, :], 
                                  empirical_mean)
        output = self.sigmoid(self.output_layer(h_t))
        return output.squeeze()
    
def preprocess_dataset(X):
    num_samples, num_timepoints, num_variables = X.shape[0], X.shape[1], X.shape[2]
    X, delta, M, last_observation, empirical_mean = grud.preprocess_dataset(X)
    delta_future = np.zeros_like(X)
    next_observation = np.full_like(X, np.nan)
    for sample in range(num_samples):
        for var in range(num_variables):
            for time in range(num_timepoints-1, -1, -1):
                if (time == num_timepoints - 1) or (np.isnan(next_observation[sample, time + 1, var])):
                    past_values = X[sample, :time+1, var]
                    past_obs = past_values[~np.isnan(past_values)]
                    if past_obs.size > 0:
                        next_observation[sample, time, var] = past_obs[-1]
                else:
                    next_observation[sample, time, var] = X[sample, time, var] if M[sample, time, var] == 1 else next_observation[sample, time + 1, var]

                if time == num_timepoints - 1:
                    delta_future[sample, time, var] = 0
                elif M[sample, time + 1, var] == 1:
                    delta_future[sample, time, var] = 1
                else:
                    delta_future[sample, time, var] = 1 + delta[sample, time + 1, var]
    empirical_mean_1 = np.nanmean(X, axis=1)
    means = np.nanmean(empirical_mean_1, axis=0)
    for i in range(empirical_mean_1.shape[1]):
        mask = np.isnan(empirical_mean_1[:, i])
        next_observation[:, :, i][mask] = means[i]

    return X, delta, delta_future, M, last_observation, next_observation, empirical_mean

class GRUDF_Weighted_Loss(nn.Module):
    def __init__(self, pos_weight, neg_weight):
        super(GRUDF_Weighted_Loss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, X, y_pred, y_true):
        missing_ratio = torch.isnan(X).view(X.size(0), -1).sum(1)/(X.size(1) * X.size(2))
        loss = -(1 - missing_ratio) * (self.pos_weight * y_true * torch.log(y_pred) + 
                    self.neg_weight * (1 - y_true) * torch.log(1 - y_pred))
        return torch.mean(loss)