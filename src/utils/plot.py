import matplotlib.pyplot as plt
import torch
import matplotlib.dates as mdates
from matplotlib.widgets import Button


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

def plot_data_spread_real_time(df_station, main_analyte, associated_analytes):
    def plot_data(year):
        # Clear previous plot
        ax.clear()

        # Filter data for the current year
        df_year = df_station[df_station['Datetime'].dt.year == year]
        timestamps = df_year['Datetime'].values
        time_diff_normalized = df_year['TimeDiff_normalized'].values
        analyte_scaled = df_year[main_analyte + '_scaled'].values
        associated_scaled = df_year[[analyte + '_scaled' for analyte in associated_analytes]].values

        # Plot data for the current year
        ax.scatter(timestamps, analyte_scaled, label=main_analyte + ' (scaled)', color='blue', alpha=0.7)

        for i, analyte in enumerate(associated_analytes):
            ax.scatter(timestamps, associated_scaled[:, i], label=analyte + ' (scaled)', alpha=0.7)

        ax.scatter(timestamps, time_diff_normalized, label='TimeDiff (normalized)', color='red', alpha=0.5)

        # Set ticks and format for months
        ax.xaxis.set_major_locator(mdates.MonthLocator())  # Major ticks every month
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Display month names
        fig.autofmt_xdate()

        ax.set_xlabel('Months', fontsize=14)
        ax.set_ylabel('Feature Values (scaled/normalized)', fontsize=14)
        ax.set_title(f'Data Spread for {year}', fontsize=16)
        ax.legend(loc='upper right')
        ax.grid(True)

        # Draw the updated plot
        fig.canvas.draw()

    def next_plot(event):
        # Update the index and plot the next year
        nonlocal current_index
        current_index = (current_index + 1) % len(unique_years)
        year = unique_years[current_index]
        plot_data(year)

    # Extract unique years from the Datetime column
    unique_years = sorted(df_station['Datetime'].dt.year.unique())
    current_index = 0

    # Initialize figure and button
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)  # Adjust for button placement
    ax_button_next = plt.axes([0.8, 0.05, 0.1, 0.075])
    button_next = Button(ax_button_next, 'Next')
    button_next.on_clicked(next_plot)

    # Plot the first year's data
    plot_data(unique_years[current_index])

    plt.show()