import os
import pandas as pd

def parse_datetime_utc_to_est(row):
    date_str = str(row['Date'])
    time_str = str(row['Time']).split('.')[0].zfill(4)
    dt = pd.to_datetime(date_str + ' ' + time_str, format='%Y%m%d %H%M')
    dt = dt.tz_localize('UTC').tz_convert('US/Eastern')
    return dt

usgs_stations = ['21FLSFWM_WQX', '21FLHILL_WQX', '21FLSJWM_WQX', '21FLGW_WQX', '21FLSWFD_WQX']

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
file_path_usgs = os.path.join(base_dir, 'data', 'USGS PhysChem Top 100 - Time Series Research.txt')

# Load USGS dataset
data = pd.read_csv(file_path_usgs, sep='\t')

# List of analytes
analytes = [
    'Dissolved Oxygen (mg/L)', 
    'pH', 
    'Turbidity (NTU)', 
    'Specific Conductance (uS/cm)', 
    'Salinity (PSU)', 
    'Secchi Disk Depth (m)'
]

# List of lower and upper bounds for time gaps
time_gap_thresholds = [(0.5, 480)]  # Example range for lower and upper bounds

# Output folder for the clipped data files
output_folder = os.path.join(base_dir, 'clipped_data')
os.makedirs(output_folder, exist_ok=True)  # Create the directory if it doesn't exist

# Loop through all USGS stations and analytes
for selected_station in usgs_stations:
    print(f"\nProcessing station: {selected_station}")
    
    # Filter data for the current station
    station_data = data[data['Station Identifier'] == selected_station]
    
    for selected_analyte in analytes:
        print(f"Analyzing analyte: {selected_analyte}")

        filtered_data = station_data[['Station Identifier', 'Date', 'Time', selected_analyte]]
        filtered_data = filtered_data.dropna(subset=[selected_analyte])

        if filtered_data.empty:
            print(f"No data available for {selected_station} and {selected_analyte} after removing missing values.")
            continue

        # Perform datetime conversion in the loop
        print("Converting Date and Time to EST...")
        filtered_data['Datetime'] = filtered_data.apply(parse_datetime_utc_to_est, axis=1)
        filtered_data = filtered_data.sort_values('Datetime')

        # Calculate time differences
        filtered_data['Time_Difference_Hours'] = filtered_data['Datetime'].diff().dt.total_seconds() / 3600

        # Initial data count before clipping
        initial_data_count = len(filtered_data)

        for lower_bound, upper_bound in time_gap_thresholds:
            # Clipping based on both lower and upper bounds
            clipped_data = filtered_data[
                (filtered_data['Time_Difference_Hours'] >= lower_bound) & 
                (filtered_data['Time_Difference_Hours'] <= upper_bound)
            ]

            if clipped_data.empty:
                continue

            # Save the clipped data to a separate text file
            output_file = os.path.join(output_folder, f"{selected_station}_{selected_analyte.replace('/', '_')}.txt")
            clipped_data_to_save = clipped_data[['Datetime', selected_analyte]]  # Select only Datetime and Analyte columns
            clipped_data_to_save.to_csv(output_file, sep='\t', index=False, header=True)
            print(f"Clipped data for {selected_station} - {selected_analyte} saved to {output_file}")
