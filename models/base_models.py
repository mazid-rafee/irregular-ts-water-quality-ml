import torch
import torch.nn as nn
from torchdiffeq import odeint

# ODE function
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 2)
        )

    def forward(self, t, y):
        y_with_time = torch.cat([y, t.unsqueeze(-1)], dim=-1)
        return self.net(y_with_time)

# Neural ODE model
class NeuralODEModel(nn.Module):
    def __init__(self, ode_func, seq_length):
        super(NeuralODEModel, self).__init__()
        self.seq_length = seq_length
        self.ode_func = ode_func
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        batch_size = x.size(0)
        out = []
        for i in range(batch_size):
            x_i = x[i, :, :2]
            t_i = x[i, :, 2]
            for j in range(1, len(t_i)):
                if t_i[j] <= t_i[j - 1]:
                    t_i[j] = t_i[j - 1] + 1e-6
            ode_sol = odeint(self.ode_func, x_i[0], t_i)
            out.append(self.fc(ode_sol[-1]))
        return torch.stack(out)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), 50)
        c_0 = torch.zeros(1, x.size(0), 50)
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        return self.fc(lstm_out[:, -1, :])

# BiLSTM model
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 50)
        c_0 = torch.zeros(2, x.size(0), 50)
        bilstm_out, _ = self.bilstm(x, (h_0, c_0))
        return self.fc(bilstm_out[:, -1, :])

# Temporal Convolutional Network (TCN) model
class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation_size, padding=(kernel_size-1) * dilation_size),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.network(x)
        return self.fc(x[:, :, -1])

# Transformer model
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, seq_length, output_dim, nhead=1, num_layers=2, dim_feedforward=128):
        super(TransformerTimeSeries, self).__init__()
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_length, input_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])
