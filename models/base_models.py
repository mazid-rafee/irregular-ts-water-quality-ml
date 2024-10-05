import torch
import torch.nn as nn

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
