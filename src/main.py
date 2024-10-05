import argparse
import pandas as pd
import numpy as np
import torch
import sys
import os
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.base_models import LSTMModel, BiLSTMModel
from train import train_model
from test import evaluate_model
from utils.plot import plot_model_results
from utils.loss_functions import sMAPELoss
from utils.data_processor import load_data, prepare_data_loaders
from utils.prompt_options import choose_analytes, choose_station


def main(args):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path_usgs = os.path.join(base_dir, 'data', 'USGS PhysChem Top 100 - Time Series Research.txt')
    file_path_dbhydro = os.path.join(base_dir, 'data', 'DbHydro PhysChem Top 100 - Time Series Research.txt')
    
    if args.dataset == 'USGS':
        file_path = file_path_usgs
    elif args.dataset == 'DbHydro':
        file_path = file_path_dbhydro
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Choose between 'USGS' or 'DbHydro'.")

    # Let user choose a station
    station_id = choose_station(args.dataset)

    # Let user choose main analyte and associated analytes
    main_analyte, associated_analytes = choose_analytes(args.dataset)
    df_station, scaler_analyte = load_data(file_path, station_id, main_analyte, associated_analytes)
    train_loader, test_loader, input_dimension = prepare_data_loaders(df_station, main_analyte, associated_analytes, 100, 32, True)

    #criterion = torch.nn.MSELoss()
    criterion = sMAPELoss()

    if 'LSTM' in args.models:
        lstm_model = LSTMModel(input_dim=input_dimension, hidden_dim=50, output_dim=1, num_layers=1)
        lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
        print("Training LSTM Model...")
        train_model(lstm_model, lstm_optimizer, criterion, train_loader, scaler_analyte, num_epochs=20)
        lstm_test_preds, lstm_test_actuals = evaluate_model(lstm_model, criterion, test_loader, scaler_analyte)
        plot_model_results(lstm_test_preds, lstm_test_actuals, scaler_analyte, "LSTM")

    if 'BiLSTM' in args.models:
        bilstm_model = BiLSTMModel(input_dim=input_dimension, hidden_dim=50, output_dim=1, num_layers=1)
        bilstm_optimizer = torch.optim.Adam(bilstm_model.parameters(), lr=0.001)
        print("Training Bi-LSTM Model...")
        train_model(bilstm_model, bilstm_optimizer, criterion, train_loader, scaler_analyte, num_epochs=20)
        bilstm_test_preds, bilstm_test_actuals = evaluate_model(bilstm_model, criterion, test_loader, scaler_analyte)
        plot_model_results(bilstm_test_preds, bilstm_test_actuals, scaler_analyte, "Bi-LSTM")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test models on selected dataset and models")
    parser.add_argument('--dataset', type=str, choices=['USGS', 'DbHydro'], required=True, help="Choose dataset: 'USGS' or 'DbHydro'")
    parser.add_argument('--models', nargs='+', choices=['LSTM', 'BiLSTM', 'NeuralODE', 'TCN', 'Transformer', 'LSTM+Attention', 'CNN+LSTM', 'LSTM+Transformer', 'TCN+LSTM'], required=True, help="Choose models to train: 'LSTM', 'BiLSTM', 'NeuralODE', 'TCN', 'Transformer', 'LSTM+Attention', 'CNN+LSTM', 'LSTM+Transformer', 'TCN+LSTM'")

    args = parser.parse_args()
    main(args)

