import matplotlib.pyplot as plt
import torch

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

def plot_model_results(test_preds, test_actuals, scaler, model_name):
    test_preds = torch.cat(test_preds).numpy()
    test_actuals = torch.cat(test_actuals).numpy()
    plot_results(test_actuals, test_preds, f'{model_name}: Actual vs Predicted')
