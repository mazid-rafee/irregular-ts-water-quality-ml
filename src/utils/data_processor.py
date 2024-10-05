import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import MinMaxScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_data(file_path, station_id, main_analyte, associated_analytes):

    df = pd.read_csv(file_path, sep='\t', header=0, low_memory=False) 
    print("Total Datapoints in the data: " + str(df.shape))
    df_station = df[df['Station Identifier'] == station_id]
    print("Total Datapoints in the station: " + str(df_station.shape))
    
    df_station.loc[:, 'Datetime'] = df_station.apply(parse_datetime, axis=1)
    df_station = df_station.sort_values('Datetime')
    all_analytes = [main_analyte] + associated_analytes
    required_columns = all_analytes + ['Datetime']
    missing_columns = [col for col in required_columns if col not in df_station.columns]

    if missing_columns:
        raise ValueError(f"The following required columns are missing from the dataframe: {missing_columns}")

    nan_columns = df_station[required_columns].isna().sum()
    empty_columns = nan_columns[nan_columns > 0]
    if not empty_columns.empty:
        print(f"The following columns have missing values (NaNs):\n{empty_columns}")  
    print("Total Datapoints in the station again: " + str(df_station.shape))

    df_station = df_station.dropna(subset=all_analytes + ['Datetime'])
    df_station = df_station[(df_station[all_analytes + ['Datetime']] != 0).all(axis=1)]
    df_station = df_station[
        df_station[all_analytes].apply(lambda col: pd.to_numeric(col, errors='coerce')).notnull().all(axis=1)
    ]

    print("Total Datapoints in the station with the analytes values: " + str(df_station.shape))
    if df_station.empty:
        raise ValueError("No valid data after dropping rows with NaN values in analytes or Datetime columns.")
  
    start_time = df_station['Datetime'].iloc[0]
    df_station['Time_in_seconds'] = (df_station['Datetime'] - start_time).dt.total_seconds()
    df_station['TimeDiff_in_seconds'] = df_station['Time_in_seconds'].diff().fillna(0)  # Time difference
    
    time_scaler = MinMaxScaler()
    time_diff_scaler = MinMaxScaler()
    df_station['Time_normalized'] = time_scaler.fit_transform(df_station[['Time_in_seconds']])
    df_station['TimeDiff_normalized'] = time_diff_scaler.fit_transform(df_station[['TimeDiff_in_seconds']])


    scaler_analyte_main = MinMaxScaler()
    scaler_analyte_associated = MinMaxScaler()
    print('df_station[[main_analyte]] Shape: ' + str(df_station[[main_analyte]].shape))
    # Scale the main analyte and associated analytes
    df_station[main_analyte + '_scaled'] = scaler_analyte_main.fit_transform(df_station[[main_analyte]])
    print('df_station[main_analyte_scaled] Shape: ' + str(df_station[main_analyte + '_scaled'].shape))
    for analyte in associated_analytes:
        df_station[analyte + '_scaled'] = scaler_analyte_associated.fit_transform(df_station[[analyte]])
    
    return df_station, scaler_analyte_main


def create_sequences(data, seq_length):
    xs, ys, timestamps_101st, time_diffs_101st = [], [], [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:i + seq_length, :]  # 100 data points with analytes + timestamps and time difference
        xs.append(x)
        
        # Extract the 101st timestamp and time difference
        timestamp_101st = data[i + seq_length, -2]  # Time_normalized (second-to-last column)
        time_diff_101st = data[i + seq_length, -1]  # TimeDiff_normalized (last column)
        
        timestamps_101st.append(timestamp_101st)
        time_diffs_101st.append(time_diff_101st)
        
        # The main analyte value of the 101st data point (this will be the target/output)
        y = data[i + seq_length, 0]
        ys.append(y)

    return np.array(xs), np.array(timestamps_101st), np.array(time_diffs_101st), np.array(ys)


def parse_datetime(row):
    date_str = str(row['Date'])
    time_str = str(row['Time']).split('.')[0].zfill(4)
    return pd.to_datetime(date_str + ' ' + time_str, format='%Y%m%d %H%M')

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def prepare_data_loaders(df_station, main_analyte, associated_analytes, seq_length=100, batch_size=32, shuffle=True):
    # Combine scaled analyte data and time data
    df_combined_scaled = np.hstack((
        df_station[[main_analyte + '_scaled'] + [analyte + '_scaled' for analyte in associated_analytes]].values,
        df_station[['Time_normalized']].values,  # Adding time as a feature
        df_station[['TimeDiff_normalized']].values  # Adding time difference as a feature
    ))

    print("Total Datapoints after creating sequence: " + str(df_combined_scaled.shape))

    # Create sequences with new input-output structure
    X, input_timestamps, input_time_diffs, y = create_sequences(df_combined_scaled, seq_length)
    print("Input Data Shape       : " + str(X.shape))
    print("Timestamp Data Shape   : " + str(input_timestamps.shape))
    print("TimeDiff Data Shape    : " + str(input_time_diffs.shape))
    print("Output Data Shape      : " + str(y.shape))

    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    input_timestamps_train, input_timestamps_test = input_timestamps[:train_size], input_timestamps[train_size:]
    input_time_diffs_train, input_time_diffs_test = input_time_diffs[:train_size], input_time_diffs[train_size:]

    # Convert numpy arrays to PyTorch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32)

    input_timestamps_train_torch = torch.tensor(input_timestamps_train, dtype=torch.float32)
    input_timestamps_test_torch = torch.tensor(input_timestamps_test, dtype=torch.float32)
    input_time_diffs_train_torch = torch.tensor(input_time_diffs_train, dtype=torch.float32)
    input_time_diffs_test_torch = torch.tensor(input_time_diffs_test, dtype=torch.float32)

    # Combine timestamps and time differences as input 2
    input_2_train_torch = torch.stack([input_timestamps_train_torch, input_time_diffs_train_torch], dim=1)
    input_2_test_torch = torch.stack([input_timestamps_test_torch, input_time_diffs_test_torch], dim=1)

    # Create data loaders
    train_dataset = TensorDataset(X_train_torch, input_2_train_torch, y_train_torch)
    test_dataset = TensorDataset(X_test_torch, input_2_test_torch, y_test_torch)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_train.shape[2]
