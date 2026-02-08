import torch
from utils.loss_functions import sMAPELoss, RMSELoss, MAPELoss
from torch.nn import MSELoss

def evaluate_model(model, model_name, criterion, test_loader, scaler, device=torch.device("cpu")):
    model.to(device)
    model.eval()

    test_smape_loss = 0.0
    test_rmse_loss = 0.0
    test_mape_loss = 0.0
    test_mse_loss = 0.0

    test_preds = []
    test_actuals = []

    smape_criterion = sMAPELoss()
    rmse_criterion = RMSELoss()
    mape_criterion = MAPELoss()
    mse_criterion = MSELoss()

    with torch.no_grad():
        for X_batch, timestamp_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            timestamp_batch = timestamp_batch.to(device)
            y_batch = y_batch.to(device)

            if model_name.lower() == "neural ode":
                outputs = model(X_batch)
            else:
                outputs = model(X_batch, timestamp_batch)

            outputs_rescaled = scaler.inverse_transform(outputs.cpu().numpy().reshape(-1, 1))
            y_batch_rescaled = scaler.inverse_transform(y_batch.cpu().numpy().reshape(-1, 1))

            outputs_rescaled_tensor = torch.tensor(outputs_rescaled, dtype=torch.float32).view(-1, 1)
            y_batch_rescaled_tensor = torch.tensor(y_batch_rescaled, dtype=torch.float32).view(-1, 1)

            test_preds.append(outputs_rescaled_tensor)
            test_actuals.append(y_batch_rescaled_tensor)

            test_smape_loss += smape_criterion(outputs_rescaled_tensor, y_batch_rescaled_tensor).item()
            test_rmse_loss += rmse_criterion(outputs_rescaled_tensor, y_batch_rescaled_tensor).item()
            test_mape_loss += mape_criterion(outputs_rescaled_tensor, y_batch_rescaled_tensor).item()
            test_mse_loss += mse_criterion(outputs_rescaled_tensor, y_batch_rescaled_tensor).item()

    test_smape_loss /= len(test_loader)
    test_rmse_loss /= len(test_loader)
    test_mape_loss /= len(test_loader)
    test_mse_loss /= len(test_loader)

    print(f'Test sMAPE Loss: {test_smape_loss}')
    print(f'Test RMSE Loss: {test_rmse_loss}')
    print(f'Test MAPE Loss: {test_mape_loss}')
    print(f'Test MSE Loss: {test_mse_loss}')

    return test_preds, test_actuals, test_smape_loss, test_rmse_loss, test_mape_loss, test_mse_loss
