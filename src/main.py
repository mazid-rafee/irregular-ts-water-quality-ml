import argparse
import pandas as pd
import numpy as np
import sys
import os
from torch.nn import MSELoss
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.base_models import LSTMModel, BiLSTMModel, LayerNormLSTMModel, LayerNormBiLSTMModel, NeuralODEModel
from models.hybrid_models import LSTMAttentionModel
from models.proposed_model import nODEBiLSTM
from utils.model_trainer_evaluator import train_and_evaluate_model
from utils.loss_functions import sMAPELoss, RMSELoss, MAPELoss
from utils.data_processor import load_data, prepare_data_loaders
from utils.prompt_options import choose_analytes, choose_station
from stations_and_analytes import usgs_stations, dbhydro_stations, usgs_analytes_with_associated, dbhydro_analytes_with_associated

def main(args):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path_usgs = os.path.join(base_dir, 'data', 'USGS PhysChem Top 5 - Time Series Research.txt')
    file_path_dbhydro = os.path.join(base_dir, 'data', 'DbHydro PhysChem Top 100 - Time Series Research.txt')
    
    if args.dataset == 'USGS':
        file_path = file_path_usgs
        stations, analytes_with_associated = usgs_stations, usgs_analytes_with_associated
    elif args.dataset == 'DbHydro':
        file_path = file_path_dbhydro
        stations, analytes_with_associated = dbhydro_stations, dbhydro_analytes_with_associated
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Choose between 'USGS' or 'DbHydro'.")
    
    criterion = sMAPELoss()
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{args.dataset}_model_results.txt")

    with open(results_file, 'w') as f:
        f.write("Station,Main Analyte,Associated Analytes,Model,sMAPE Loss,RMSE Loss,MAPE Loss,MSE Loss\n")

    for station_id in stations:
        print(f"\nProcessing station: {station_id}")
        
        for analyte_set in analytes_with_associated:
            main_analyte = analyte_set[0].replace('Main Analyte: ', '')
            associated_analytes = [a.replace('Associated: ', '') for a in analyte_set[1:]]
            print(f"  Main Analyte: {main_analyte}")
            print(f"    Associated Analytes: {', '.join(associated_analytes) if associated_analytes else 'None'}")
            
            df_station, scaler_analyte = load_data(file_path, station_id, main_analyte, associated_analytes)
            train_loader, test_loader, input_dimension = prepare_data_loaders(
                df_station, main_analyte, associated_analytes, 100, 32, True
            )
            
            for model_choice in args.models:
                print(f"Training {model_choice} on station {station_id} with main analyte {main_analyte} and associated analytes {associated_analytes}")
                if model_choice == 'LSTM':
                    model, _, _, smape_loss, rmse_loss, mape_loss, mse_loss = train_and_evaluate_model(
                        model_class=LSTMModel, model_name="LSTM", input_dimension=input_dimension,
                        hidden_dim=50, output_dim=1, num_layers=1, criterion=criterion,
                        train_loader=train_loader, test_loader=test_loader, scaler=scaler_analyte, num_epochs=20
                    )
                elif model_choice == 'nLSTM':
                    model, _, _, smape_loss, rmse_loss, mape_loss, mse_loss = train_and_evaluate_model(
                        model_class=LayerNormLSTMModel, model_name="Layer Normalized LSTM", 
                        input_dimension=input_dimension, hidden_dim=50, output_dim=1, num_layers=1, 
                        criterion=criterion, train_loader=train_loader, test_loader=test_loader, 
                        scaler=scaler_analyte, num_epochs=20
                    )
                elif model_choice == 'BiLSTM':
                    model, _, _, smape_loss, rmse_loss, mape_loss, mse_loss = train_and_evaluate_model(
                        model_class=BiLSTMModel, model_name="Bi-LSTM", input_dimension=input_dimension,
                        hidden_dim=50, output_dim=1, num_layers=1, criterion=criterion,
                        train_loader=train_loader, test_loader=test_loader, scaler=scaler_analyte, num_epochs=20
                    )
                elif model_choice == 'aLSTM':
                    model, _, _, smape_loss, rmse_loss, mape_loss, mse_loss = train_and_evaluate_model(
                        model_class=LSTMAttentionModel, model_name="LSTM with Attention", 
                        input_dimension=input_dimension, hidden_dim=50, output_dim=1, num_layers=1, criterion=criterion, 
                        train_loader=train_loader, test_loader=test_loader, scaler=scaler_analyte, num_epochs=20
                    )
                elif model_choice == 'nBiLSTM':
                    model, _, _, smape_loss, rmse_loss, mape_loss, mse_loss = train_and_evaluate_model(
                        model_class=LayerNormBiLSTMModel, model_name="Layer Normalized Bi-LSTM", 
                        input_dimension=input_dimension, hidden_dim=50, output_dim=1, num_layers=1, criterion=criterion, 
                        train_loader=train_loader, test_loader=test_loader, scaler=scaler_analyte, num_epochs=20
                    )
                elif model_choice == 'NeuralODE':
                    model, _, _, smape_loss, rmse_loss, mape_loss, mse_loss = train_and_evaluate_model(
                        model_class=NeuralODEModel, model_name="Neural ODE", 
                        input_dimension=input_dimension, hidden_dim=50, output_dim=1, num_layers=1, criterion=criterion, 
                        train_loader=train_loader, test_loader=test_loader, scaler=scaler_analyte, num_epochs=20
                    )
                elif model_choice == 'nODEBiLSTM':
                    model, _, _, smape_loss, rmse_loss, mape_loss, mse_loss = train_and_evaluate_model(
                        model_class=nODEBiLSTM, model_name="nODE BiLSTM", input_dimension=input_dimension,
                        hidden_dim=50, output_dim=1, num_layers=1, criterion=criterion,
                        train_loader=train_loader, test_loader=test_loader, scaler=scaler_analyte, num_epochs=20
                    )
                elif model_choice == 'nODELSTM':
                    model, _, _, smape_loss, rmse_loss, mape_loss, mse_loss = train_and_evaluate_model(
                        model_class=nODEBiLSTM, model_name="nODE LSTM", input_dimension=input_dimension,
                        hidden_dim=50, output_dim=1, num_layers=1, criterion=criterion,
                        train_loader=train_loader, test_loader=test_loader, scaler=scaler_analyte, num_epochs=20
                    )
                else:
                    print(f"Model {model_choice} is not implemented. Skipping...")
                    continue

                with open(results_file, 'a') as f:
                    f.write(f"{station_id},{main_analyte},{'|'.join(associated_analytes)},{model_choice},{smape_loss},{rmse_loss},{mape_loss},{mse_loss}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test models on selected dataset and models")
    parser.add_argument('--dataset', type=str, choices=['USGS', 'DbHydro'], required=True, help="Choose dataset: 'USGS' or 'DbHydro'")
    parser.add_argument('--models', nargs='+', choices=['LSTM', 'nLSTM', 'BiLSTM', 'NeuralODE', 'nODEBiLSTM', 'nODELSTM', 'TCN', 'nBiLSTM', 'aLSTM', 'CNN+LSTM', 'LSTM+Transformer', 'TCN+LSTM'], required=True, help="Choose models to train: 'LSTM', 'nLSTM', 'BiLSTM', 'nODEBiLSTM', 'nODELSTM', 'NeuralODE', 'TCN', 'nBiLSTM', 'aLSTM', 'CNN+LSTM', 'LSTM+Transformer', 'TCN+LSTM'")

    args = parser.parse_args()
    main(args)

