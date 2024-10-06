import pandas as pd
import numpy as np
import os
from pytz import timezone
from PyEMD import EMD
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def parse_datetime_utc_to_est(row):
    date_str = str(row['Date'])
    time_str = str(row['Time']).split('.')[0].zfill(4)
    dt = pd.to_datetime(date_str + ' ' + time_str, format='%Y%m%d %H%M')
    dt = dt.tz_localize('UTC').tz_convert('US/Eastern')
    return dt

def decompose_signal(signal):
    emd = EMD()
    imfs = emd(signal)
    return imfs

def prepare_lstm_data_with_features(imfs, timestamps, time_diffs, look_back=30):
    X, y = [], []
    for i in range(len(imfs) - look_back):
        X.append(np.column_stack([imfs[i:i + look_back], time_diffs[i:i + look_back], timestamps[i:i + look_back]]))
        y.append(imfs[i + look_back])
    return np.array(X), np.array(y)

class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=64, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

class sMAPELoss(nn.Module):
    def __init__(self):
        super(sMAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-7 
        diff = torch.abs(y_pred - y_true)
        denominator = torch.abs(y_true) + torch.abs(y_pred) + epsilon
        smape = 2.0 * torch.mean(diff / denominator) * 100
        return smape

def train_lstm_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=64, lr=0.001):
    print("Starting training process...")
    criterion = sMAPELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    test_data = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    output = model(batch_X)
                    test_loss += criterion(output, batch_y).item()

            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Test Loss: {test_loss / len(test_loader)}')

def plot_predictions(true_values, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(true_values, label='True Values')
    plt.plot(predictions, label='Predicted Values', color='r')
    plt.title('Dissolved Oxygen (mg/L) Forecast')
    plt.xlabel('Time Steps')
    plt.ylabel('Dissolved Oxygen (mg/L)')
    plt.legend()
    plt.show()

def main(file_path):
    print("Loading data...")
    data = pd.read_csv(file_path, sep='\t')

    print("Filtering data for station SE 03...")
    data = data[data['Station Identifier'] == 'SE 03']

    print("Converting Date and Time to EST...")
    data['Datetime'] = data.apply(parse_datetime_utc_to_est, axis=1)
    data.set_index('Datetime', inplace=True)

    print("Extracting and preparing 'Dissolved Oxygen (mg/L)' values...")
    dissolved_oxygen = data['Dissolved Oxygen (mg/L)'].dropna()

    print("Decomposing the time series using EMD...")
    imfs = decompose_signal(dissolved_oxygen.values)
    decomposed_signal = imfs[0]

    print("Calculating timestamps and time differences...")
    timestamps = data.index.astype(np.int64) // 10**9  # Convert datetime to Unix timestamp (in seconds)
    timestamps = timestamps[:len(decomposed_signal)]  # Align timestamps with decomposed signal
    time_diffs = np.diff(timestamps, prepend=timestamps[0])  # Calculate time differences (in seconds)

    print("Preparing input data with features (IMF, time differences, and timestamps)...")
    look_back = 30
    X, y = prepare_lstm_data_with_features(decomposed_signal, timestamps, time_diffs, look_back=look_back)

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("Converting data to tensors...")
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    y_train = torch.Tensor(y_train).unsqueeze(-1)
    y_test = torch.Tensor(y_test).unsqueeze(-1)

    print("Building the LSTM model...")
    model = LSTMModel(input_size=3, hidden_layer_size=64, output_size=1)

    print("Training the model...")
    train_lstm_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=64)

    print("Making predictions...")
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()

    print("Plotting results...")
    plot_predictions(y_test.numpy(), predictions)

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path_dbhydro = os.path.join(base_dir, 'data', 'DbHydro PhysChem Top 100 - Time Series Research.txt')
    print(f"Starting the process with DbHydro Dataset")
    main(file_path_dbhydro)
    print("Process completed successfully.")