import torch
import torch.nn as nn
import torch.nn.functional as F

# LSTM + Attention Model
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), 50)
        c_0 = torch.zeros(1, x.size(0), 50)
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attn_output = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(attn_output)

# CNN + LSTM Model
class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(32, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to (batch_size, input_dim, seq_len)
        conv_out = torch.relu(self.conv1(x))
        conv_out = conv_out.permute(0, 2, 1)  # Back to (batch_size, seq_len, features)
        h_0 = torch.zeros(1, conv_out.size(0), 50)
        c_0 = torch.zeros(1, conv_out.size(0), 50)
        lstm_out, _ = self.lstm(conv_out, (h_0, c_0))
        return self.fc(lstm_out[:, -1, :])

# TCN + LSTM Model Hybrid Model
class TCNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(TCNLSTMModel, self).__init__()
        self.tcn = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)  # Replace with actual TCN layer
        self.lstm = nn.LSTM(32, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to (batch_size, input_dim, seq_len)
        tcn_out = torch.relu(self.tcn(x))
        tcn_out = tcn_out.permute(0, 2, 1)  # Back to (batch_size, seq_len, features)
        h_0 = torch.zeros(1, tcn_out.size(0), 50)
        c_0 = torch.zeros(1, tcn_out.size(0), 50)
        lstm_out, _ = self.lstm(tcn_out, (h_0, c_0))
        return self.fc(lstm_out[:, -1, :])

# LSTM + Transformer Hybrid Model
class LSTMTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMTransformerModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.transformer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=1)  # Transformer Encoder Layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), 50)
        c_0 = torch.zeros(1, x.size(0), 50)
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        transformer_out = self.transformer(lstm_out)
        return self.fc(transformer_out[:, -1, :])