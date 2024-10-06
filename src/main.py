import argparse
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.base_models import LSTMModel, BiLSTMModel, LayerNormLSTMModel
from models.hybrid_models import LSTMAttentionModel
from utils.model_trainer_evaluator import train_and_evaluate_model
from utils.loss_functions import sMAPELoss, RMSELoss, MAPELoss
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

    station_id = choose_station(args.dataset)
    main_analyte, associated_analytes = choose_analytes(args.dataset)
    df_station, scaler_analyte = load_data(file_path, station_id, main_analyte, associated_analytes)
    train_loader, test_loader, input_dimension = prepare_data_loaders(df_station, main_analyte, associated_analytes, 100, 32, True)

    
    #criterion = RMSELoss()
    criterion = sMAPELoss()

    if 'LSTM' in args.models:
        train_and_evaluate_model(
            model_class=LSTMModel, model_name="LSTM", input_dimension=input_dimension,
            hidden_dim=50, output_dim=1, num_layers=1, criterion=criterion,
            train_loader=train_loader, test_loader=test_loader, scaler=scaler_analyte, num_epochs=20
        )

    if 'nLSTM' in args.models:
        train_and_evaluate_model(
            model_class=LayerNormLSTMModel, model_name="Layer Normalized LSTM", input_dimension=input_dimension, 
            hidden_dim=50, output_dim=1, num_layers=1, criterion=criterion, 
            train_loader=train_loader, test_loader=test_loader, scaler=scaler_analyte, num_epochs=20
        )

    if 'BiLSTM' in args.models:
        train_and_evaluate_model(
            model_class=BiLSTMModel, model_name="Bi-LSTM", input_dimension=input_dimension,
            hidden_dim=50, output_dim=1, num_layers=1, criterion=criterion,
            train_loader=train_loader, test_loader=test_loader, scaler=scaler_analyte, num_epochs=20
        )

    if 'aLSTM' in args.models:
        train_and_evaluate_model(
            model_class=LSTMAttentionModel, model_name="LSTM with Attention", 
            input_dimension=input_dimension, hidden_dim=50, output_dim=1, num_layers=1, criterion=criterion, 
            train_loader=train_loader, test_loader=test_loader, scaler=scaler_analyte, num_epochs=20
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test models on selected dataset and models")
    parser.add_argument('--dataset', type=str, choices=['USGS', 'DbHydro'], required=True, help="Choose dataset: 'USGS' or 'DbHydro'")
    parser.add_argument('--models', nargs='+', choices=['LSTM', 'nLSTM', 'BiLSTM', 'NeuralODE', 'TCN', 'Transformer', 'aLSTM', 'CNN+LSTM', 'LSTM+Transformer', 'TCN+LSTM'], required=True, help="Choose models to train: 'LSTM', 'nLSTM', 'BiLSTM', 'NeuralODE', 'TCN', 'Transformer', 'aLSTM', 'CNN+LSTM', 'LSTM+Transformer', 'TCN+LSTM'")

    args = parser.parse_args()
    main(args)

