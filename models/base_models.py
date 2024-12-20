import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import random


# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim + 2, output_dim)

    def forward(self, x, t):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)       
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        lstm_out_last_step = lstm_out[:, -1, :] 
        combined_input = torch.cat((lstm_out_last_step, t), dim=-1)
        
        return self.fc(combined_input)


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2 + 2, output_dim)

    def forward(self, x, t):
        h_0 = torch.zeros(2 * self.bilstm.num_layers, x.size(0), self.bilstm.hidden_size).to(x.device)
        c_0 = torch.zeros(2 * self.bilstm.num_layers, x.size(0), self.bilstm.hidden_size).to(x.device)
        bilstm_out, _ = self.bilstm(x, (h_0, c_0))
        bilstm_out_last_step = bilstm_out[:, -1, :] 
        combined_input = torch.cat((bilstm_out_last_step, t), dim=-1)
    
        return self.fc(combined_input)


class LayerNormLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        super(LayerNormLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim + 2, output_dim)

    def forward(self, x, t):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        lstm_out_last_step = self.layer_norm(lstm_out[:, -1, :]) 
        combined_input = torch.cat((lstm_out_last_step, t), dim=-1)
        
        return self.fc(combined_input)

class LayerNormBiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        super(LayerNormBiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.layer_norm = nn.LayerNorm(2 * hidden_dim)
        self.fc = nn.Linear(2 * hidden_dim + 2, output_dim)

    def forward(self, x, t):
        h_0 = torch.zeros(2 * self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(2 * self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        lstm_out_last_step = lstm_out[:, -1, :]
        lstm_out_last_step = self.layer_norm(lstm_out_last_step)
        combined_input = torch.cat((lstm_out_last_step, t), dim=-1)
        
        return self.fc(combined_input)


class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, t, h):
        return self.relu(self.fc(h))

class NeuralODEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, t=torch.linspace(0, 1, steps=10)):
        super(NeuralODEModel, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.odefunc = ODEFunc(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.default_t = t

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        h0 = self.fc_in(x.view(-1, input_dim))
        h0 = h0.view(batch_size * seq_len, -1)
        t = self.default_t.to(x.device).float()
        ode_out = odeint(self.odefunc, h0, t)
        ode_out_last = ode_out[-1].view(batch_size, seq_len, -1)[:, -1, :]
        return self.fc_out(ode_out_last)






