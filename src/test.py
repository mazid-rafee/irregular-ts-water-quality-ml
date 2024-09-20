import torch

# Model evaluation function
def evaluate_model(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            test_preds.append(outputs)
            test_actuals.append(y_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            test_loss += loss.item()

    test_loss = test_loss / len(test_loader)
    print(f'Test Loss (MSE): {test_loss}')
    return test_preds, test_actuals
