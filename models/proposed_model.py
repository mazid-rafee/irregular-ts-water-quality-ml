import torch
import torch.nn as nn
from torchdiffeq import odeint

class nODEBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, t=torch.tensor([0, 1], dtype=torch.float32)):
        super(nODEBiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.odefunc = ODEFunc(hidden_dim * 2)
        self.t = t
        self.fc = nn.Linear(hidden_dim * 2 + 2, output_dim)

    def forward(self, x, t):
        h_0 = torch.zeros(2 * self.bilstm.num_layers, x.size(0), self.bilstm.hidden_size).to(x.device)
        c_0 = torch.zeros(2 * self.bilstm.num_layers, x.size(0), self.bilstm.hidden_size).to(x.device)
        bilstm_out, _ = self.bilstm(x, (h_0, c_0))
        bilstm_out_last_step = bilstm_out[:, -1, :]
        self.t = self.t.to(x.device).float()
        ode_out = odeint(self.odefunc, bilstm_out_last_step, self.t)
        ode_out_last = ode_out[-1]
        t = t.float()
        combined_input = torch.cat((ode_out_last, t), dim=-1)
        return self.fc(combined_input)

class nODELSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, t=torch.tensor([0, 1], dtype=torch.float32)):
        super(nODELSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.odefunc = ODEFunc(hidden_dim)
        self.t = t
        self.fc = nn.Linear(hidden_dim + 2, output_dim)

    def forward(self, x, t):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        lstm_out_last_step = lstm_out[:, -1, :]
        self.t = self.t.to(x.device).float()
        ode_out = odeint(self.odefunc, lstm_out_last_step, self.t)
        ode_out_last = ode_out[-1]
        t = t.float()
        combined_input = torch.cat((ode_out_last, t), dim=-1)
        return self.fc(combined_input)


class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, t, h):
        return self.relu(self.fc(h))