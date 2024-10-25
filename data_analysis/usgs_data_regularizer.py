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

# List of lower and upper bounds for time gaps
time_gap_thresholds = [(0.5, 480)]  # Example range for lower and upper bounds

# Output file for the combined clipped data
output_file = os.path.join(base_dir, 'combined_clipped_data.txt')

# Initialize an empty DataFrame to store combined data
combined_clipped_data = pd.DataFrame()

# Loop through all USGS stations
for selected_station in usgs_stations:
    print(f"\nProcessing station: {selected_station}")
    
    # Filter data for the current station
    station_data = data[data['Station Identifier'] == selected_station]
    
    if station_data.empty:
        print(f"No data available for {selected_station}")
        continue

    # Perform datetime conversion for each station
    print("Converting Date and Time to EST...")
    station_data['Datetime'] = station_data.apply(parse_datetime_utc_to_est, axis=1)
    station_data = station_data.sort_values('Datetime')

    # Calculate time differences
    station_data['Time_Difference_Hours'] = station_data['Datetime'].diff().dt.total_seconds() / 3600

    # Apply time difference threshold
    for lower_bound, upper_bound in time_gap_thresholds:
        # Clipping based on both lower and upper bounds
        clipped_data = station_data[
            (station_data['Time_Difference_Hours'] >= lower_bound) & 
            (station_data['Time_Difference_Hours'] <= upper_bound)
        ]

        if clipped_data.empty:
            print(f"No data available for {selected_station} within time difference threshold {lower_bound}-{upper_bound} hours")
            continue

        # Remove the extra columns 'Datetime' and 'Time_Difference_Hours' to match the input file
        clipped_data = clipped_data.drop(columns=['Datetime', 'Time_Difference_Hours'])

        # Append clipped data to the combined DataFrame
        combined_clipped_data = pd.concat([combined_clipped_data, clipped_data])

# Check if there is any data to save
if not combined_clipped_data.empty:
    # Save the combined clipped data to a single text file, maintaining the original columns
    combined_clipped_data.to_csv(output_file, sep='\t', index=False, header=True)
    print(f"Combined clipped data saved to {output_file}")
else:
    print("No data available within the specified time difference thresholds.")
