import torch

def evaluate_model(model, criterion, test_loader, scaler):
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_actuals = []

    with torch.no_grad():
        for X_batch, timestamp_batch, y_batch in test_loader:
            sequence_input = X_batch[:, :, :]
            timestamp_input = timestamp_batch

            # Forward pass
            outputs = model(sequence_input, timestamp_input)

            # Rescale the outputs and actuals (Inverse transform)
            outputs_rescaled = scaler.inverse_transform(outputs.cpu().numpy().reshape(-1, 1))
            y_batch_rescaled = scaler.inverse_transform(y_batch.cpu().numpy().reshape(-1, 1))

            # Convert rescaled values back to PyTorch tensors
            outputs_rescaled_tensor = torch.tensor(outputs_rescaled, dtype=torch.float32).view(-1, 1)
            y_batch_rescaled_tensor = torch.tensor(y_batch_rescaled, dtype=torch.float32).view(-1, 1)

            # Append predictions and actuals for later analysis
            test_preds.append(outputs_rescaled_tensor)
            test_actuals.append(y_batch_rescaled_tensor)

            # Calculate loss on rescaled values (no broadcasting issue now)
            loss = criterion(outputs_rescaled_tensor, y_batch_rescaled_tensor)
            test_loss += loss.item()

    test_loss = test_loss / len(test_loader)
    print(f'Test Loss (MSE) on Original Scale: {test_loss}')

    return test_preds, test_actuals
