import torch
import torch.nn as nn
from torchdiffeq import odeint

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim + 2, output_dim)  # +2 for both timestamp and time difference

    def forward(self, x, t):
        # LSTM expects (batch_size, seq_length, input_dim)
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        
        # Concatenate the LSTM output (from the last time step) with the 101st timestamp and time difference
        lstm_out_last_step = lstm_out[:, -1, :]  # Get the output of the last time step
        combined_input = torch.cat((lstm_out_last_step, t), dim=-1)  # Concatenate with timestamp and time diff (Input 2)
        
        # Feed into the fully connected layer
        return self.fc(combined_input)


# BiLSTM model with additional timestamp and time difference input
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2 + 2, output_dim)  # +2 to include both timestamp and time difference

    def forward(self, x, t):
        # BiLSTM expects (batch_size, seq_length, input_dim)
        h_0 = torch.zeros(2 * self.bilstm.num_layers, x.size(0), self.bilstm.hidden_size).to(x.device)  # 2 for bidirectional
        c_0 = torch.zeros(2 * self.bilstm.num_layers, x.size(0), self.bilstm.hidden_size).to(x.device)
        
        # Pass through BiLSTM
        bilstm_out, _ = self.bilstm(x, (h_0, c_0))
        
        # Extract the last time step output from the BiLSTM
        bilstm_out_last_step = bilstm_out[:, -1, :]  # Get the output of the last time step
        
        # Concatenate the BiLSTM output (from the last time step) with the timestamp and time difference
        combined_input = torch.cat((bilstm_out_last_step, t), dim=-1)  # Concatenate with timestamp and time difference
        
        # Feed the concatenated result into the fully connected layer
        return self.fc(combined_input)

