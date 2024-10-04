import matplotlib.pyplot as plt
import torch

# Function to plot actual vs predicted results
def plot_results(actuals, predictions, title):
    plt.figure(figsize=(12, 6))
    plt.plot(actuals.flatten(), label='Actual', color='blue', linewidth=2)
    plt.plot(predictions.flatten(), label='Predicted', color='red', linewidth=2)
    plt.title(title, fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Dissolved Oxygen (mg/L)', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True)
    plt.show()

# Function to rescale and plot results for models
def plot_model_results(test_preds, test_actuals, scaler, model_name):
    test_preds = torch.cat(test_preds).numpy()
    test_actuals = torch.cat(test_actuals).numpy()

    # Rescale predictions and actuals back to original scale
    # test_actuals_rescaled = scaler.inverse_transform(test_actuals.reshape(-1, 1))
    # test_preds_rescaled = scaler.inverse_transform(test_preds.reshape(-1, 1))

    # Plot results
    plot_results(test_actuals, test_preds, f'{model_name}: Actual vs Predicted')
