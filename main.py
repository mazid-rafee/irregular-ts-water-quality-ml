import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from models import NeuralODEModel, LSTMModel, BiLSTMModel, TemporalConvNet, TransformerTimeSeries, ODEFunc
from train import train_model
from test import evaluate_model
from plot import plot_model_results

# Data loading and preprocessing
def load_data(file_path, station_id, analyte):
    df = pd.read_csv(file_path, sep='\t', header=0, low_memory=False)
    df_station = df[df['Station Identifier'] == station_id]

    def parse_datetime(row):
        date_str = str(row['Date'])
        time_str = str(row['Time']).split('.')[0].zfill(4)
        return pd.to_datetime(date_str + ' ' + time_str, format='%Y%m%d %H%M')

    df_station.loc[:, 'Datetime'] = df_station.apply(parse_datetime, axis=1)
    df_station = df_station.sort_values('Datetime')
    df_station = df_station.dropna(subset=[analyte, 'Datetime'])

    start_time = df_station['Datetime'].iloc[0]
    df_station['Time_in_seconds'] = (df_station['Datetime'] - start_time).dt.total_seconds()
    time_scaler = MinMaxScaler()
    df_station['Time_normalized'] = time_scaler.fit_transform(df_station[['Time_in_seconds']])
    df_station['TimeDiff'] = df_station['Datetime'].diff().dt.total_seconds().fillna(0)
    
    scaler_analyte = MinMaxScaler()
    scaler_timediff = MinMaxScaler()

    df_station['Analyte_scaled'] = scaler_analyte.fit_transform(df_station[[analyte]])
    df_station['TimeDiff_scaled'] = scaler_timediff.fit_transform(df_station[['TimeDiff']])

    return df_station, scaler_analyte

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def main(args):
    # File paths and data settings
    file_path_usgs = './data/USGS PhysChem Top 100 - Time Series Research.txt'
    file_path_dbhydro = './data/DbHydro PhysChem Top 100 - Time Series Research.txt'
    top_station_id_usgs = '21FLSJWM_WQX'
    top_station_id_dbhydro = 'SE 03'
    analyte_usgs = 'Dissolved Oxygen (mg/L)'
    analyte_dbhydro = 'Sp Conductivity (uS/cm)'

    # Load and preprocess the selected dataset
    if args.dataset == 'USGS':
        df_station, scaler_analyte = load_data(file_path_usgs, top_station_id_usgs, analyte_usgs)
    elif args.dataset == 'DbHydro':
        df_station, scaler_analyte = load_data(file_path_dbhydro, top_station_id_dbhydro, analyte_dbhydro)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Choose between 'USGS' or 'DbHydro'.")

    df_combined_scaled = np.hstack((
        df_station[['Analyte_scaled', 'TimeDiff_scaled']].values,
        df_station[['Time_normalized']].values
    ))

    # Create sequences
    seq_length = 100
    X, y = create_sequences(df_combined_scaled, seq_length)

    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Convert numpy arrays to PyTorch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32)

    # Create data loaders
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    test_dataset = TensorDataset(X_test_torch, y_test_torch)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define loss function and optimizers
    criterion = torch.nn.MSELoss()

    # Initialize models and train based on user choice
    if 'LSTM' in args.models:
        lstm_model = LSTMModel(input_dim=3, hidden_dim=50, output_dim=1, num_layers=1)
        lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
        print("Training LSTM Model...")
        train_model(lstm_model, lstm_optimizer, criterion, train_loader, num_epochs=20)
        lstm_test_preds, lstm_test_actuals = evaluate_model(lstm_model, criterion, test_loader)
        plot_model_results(lstm_test_preds, lstm_test_actuals, scaler_analyte, "LSTM")

    if 'BiLSTM' in args.models:
        bilstm_model = BiLSTMModel(input_dim=3, hidden_dim=50, output_dim=1, num_layers=1)
        bilstm_optimizer = torch.optim.Adam(bilstm_model.parameters(), lr=0.001)
        print("Training Bi-LSTM Model...")
        train_model(bilstm_model, bilstm_optimizer, criterion, train_loader, num_epochs=20)
        bilstm_test_preds, bilstm_test_actuals = evaluate_model(bilstm_model, criterion, test_loader)
        plot_model_results(bilstm_test_preds, bilstm_test_actuals, scaler_analyte, "Bi-LSTM")

    if 'NeuralODE' in args.models:
        ode_func = ODEFunc()
        ode_model = NeuralODEModel(ode_func, seq_length=seq_length)
        ode_optimizer = torch.optim.Adam(ode_model.parameters(), lr=0.001)
        print("Training Neural ODE Model...")
        train_model(ode_model, ode_optimizer, criterion, train_loader, num_epochs=20)
        ode_test_preds, ode_test_actuals = evaluate_model(ode_model, criterion, test_loader)
        plot_model_results(ode_test_preds, ode_test_actuals, scaler_analyte, "Neural ODE")

    if 'TCN' in args.models:
        tcn_model = TemporalConvNet(input_dim=3, output_dim=1, num_channels=[50, 50])
        tcn_optimizer = torch.optim.Adam(tcn_model.parameters(), lr=0.001)
        print("Training TCN Model...")
        train_model(tcn_model, tcn_optimizer, criterion, train_loader, num_epochs=20)
        tcn_test_preds, tcn_test_actuals = evaluate_model(tcn_model, criterion, test_loader)
        plot_model_results(tcn_test_preds, tcn_test_actuals, scaler_analyte, "TCN")

    if 'Transformer' in args.models:
        transformer_model = TransformerTimeSeries(input_dim=3, seq_length=seq_length, output_dim=1)
        transformer_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)
        print("Training Transformer Model...")
        train_model(transformer_model, transformer_optimizer, criterion, train_loader, num_epochs=20)
        transformer_test_preds, transformer_test_actuals = evaluate_model(transformer_model, criterion, test_loader)
        plot_model_results(transformer_test_preds, transformer_test_actuals, scaler_analyte, "Transformer")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test models on selected dataset and models")
    parser.add_argument('--dataset', type=str, choices=['USGS', 'DbHydro'], required=True, help="Choose dataset: 'USGS' or 'DbHydro'")
    parser.add_argument('--models', nargs='+', choices=['LSTM', 'BiLSTM', 'NeuralODE', 'TCN', 'Transformer'], required=True, help="Choose models to train: 'LSTM', 'BiLSTM', 'NeuralODE', 'TCN', 'Transformer'")

    args = parser.parse_args()
    main(args)
