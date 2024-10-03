import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(file_path, station_id, analyte):
    # Load the data
    df = pd.read_csv(file_path, sep='\t', header=0, low_memory=False)
    
    # Filter the dataframe by station ID and create a copy to avoid SettingWithCopyWarning
    df_station = df[df['Station Identifier'] == station_id].copy()

    # Parse datetime from separate Date and Time columns
    def parse_datetime(row):
        if pd.isnull(row['Date']) or pd.isnull(row['Time']):
            return pd.NaT  # Return NaT (Not a Time) for missing values
        
        date_str = str(row['Date'])
        time_str = str(row['Time']).split('.')[0].zfill(4)  # Ensure time is in HHMM format
        datetime_str = date_str + ' ' + time_str
        return pd.to_datetime(datetime_str, format='%Y%m%d %H%M')

    # Create Datetime column and sort by datetime
    df_station['Datetime'] = df_station.apply(parse_datetime, axis=1)
    
    # Remove rows with missing Datetime or analyte
    df_station = df_station.dropna(subset=[analyte, 'Datetime'])

    # Ensure the data is sorted by 'Datetime' before calculating time differences
    df_station = df_station.sort_values('Datetime')

    # Calculate time differences in seconds from the start
    df_station['TimeDiff'] = df_station['Datetime'].diff().dt.total_seconds().fillna(0)

    # Scale the analyte values
    scaler_analyte = MinMaxScaler()
    df_station['Analyte_scaled'] = scaler_analyte.fit_transform(df_station[[analyte]])

    return df_station, scaler_analyte

# Plotting function to visualize the time deltas with serial index on x-axis and TimeDiff in days on y-axis
def plot_time_deltas(df_station):
    plt.figure(figsize=(10, 6))
    
    # Ensure the data is sorted by 'Datetime' in ascending order
    df_station = df_station.sort_values(by='Datetime')

    # Convert TimeDiff from seconds to days (no filtering of negative values)
    df_station['TimeDiff_days'] = df_station['TimeDiff'] / 86400

    # Create a serial index for the data points (1, 2, 3, ...)
    df_station['SerialIndex'] = range(1, len(df_station) + 1)
    
    # Use scatter plot to show points with serial index on x-axis and TimeDiff in days on y-axis
    plt.scatter(df_station['SerialIndex'], df_station['TimeDiff_days'], label="Time Difference (days)", color='blue', marker='o')
    
    # Set plot labels and title
    plt.xlabel('Data Points (Serially)')
    plt.ylabel('Time Difference (days)')
    plt.title(f"Time Deltas in Days for Station {df_station['Station Identifier'].iloc[0]}")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
file_path_usgs = os.path.join(base_dir, 'data', 'USGS PhysChem Top 100 - Time Series Research.txt')
file_path_dbhydro = os.path.join(base_dir, 'data', 'DbHydro PhysChem Top 100 - Time Series Research.txt')
top_station_id_usgs = '21FLSJWM_WQX'
top_station_id_dbhydro = 'SE 03'
analyte_usgs = 'Organic Carbon (mg/L)'
analyte_dbhydro = 'Turbidity (NTU)'

df_station, _ = load_data(file_path_dbhydro, top_station_id_dbhydro, analyte_dbhydro)

if not df_station.empty:
    plot_time_deltas(df_station)
