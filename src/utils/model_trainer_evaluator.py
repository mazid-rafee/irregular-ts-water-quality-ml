import torch
from train import train_model
from test import evaluate_model
from utils.plot import plot_model_results

def train_and_evaluate_model(model_class, model_name, input_dimension, hidden_dim, output_dim, num_layers, criterion, train_loader, test_loader, scaler, num_epochs=20, lr=0.001):
        model = model_class(input_dim=input_dimension, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)     
        print(f"Training {model_name}...")
        train_model(model, model_name, optimizer, criterion, train_loader, scaler, num_epochs=num_epochs)
        
        test_preds, test_actuals, test_smape_loss, test_rmse_loss, test_mape_loss, test_mse_loss = evaluate_model(
            model, model_name, criterion, test_loader, scaler
        )
        
        # plot_model_results(test_preds, test_actuals, scaler, model_name)
        
        return model, test_preds, test_actuals, test_smape_loss, test_rmse_loss, test_mape_loss, test_mse_loss
