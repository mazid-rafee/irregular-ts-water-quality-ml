import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

def parse_datetime_utc_to_est(row):
    date_str = str(row['Date'])
    time_str = str(row['Time']).split('.')[0].zfill(4)
    dt = pd.to_datetime(date_str + ' ' + time_str, format='%Y%m%d %H%M')
    dt = dt.tz_localize('UTC').tz_convert('US/Eastern')
    return dt

dbhydro_stations = ['SE 03', 'SE 01', 'SE 02', 'SE 06', 'SE 09', 'SE 10', 'SE 04', 'SE 11', 'HR1', 'SE 07']
usgs_stations = ['21FLSFWM_WQX', '21FLHILL_WQX', '21FLSJWM_WQX', '21FLGW_WQX', '21FLSWFD_WQX']

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
file_path_dbhydro = os.path.join(base_dir, 'data', 'DbHydro PhysChem Top 100 - Time Series Research.txt')
file_path_usgs = os.path.join(base_dir, 'data', 'USGS PhysChem Top 100 - Time Series Research.txt')

data_source = input("Please select the data source (1 for DBHydro, 2 for USGS): ").strip()

if data_source == '1':
    file_path = file_path_dbhydro
    stations = dbhydro_stations
    print("Loading DBHydro dataset...")
elif data_source == '2':
    file_path = file_path_usgs
    stations = usgs_stations
    print("Loading USGS dataset...")
else:
    print("Invalid selection. Please choose either '1' (DBHydro) or '2' (USGS).")
    exit()

data = pd.read_csv(file_path, sep='\t')

print("\nAvailable Stations:")
for i, station in enumerate(stations):
    print(f"{i + 1}. {station}")

selected_station_index = int(input("\nPlease select a station by number: ")) - 1

if selected_station_index not in range(len(stations)):
    print("Invalid station selection.")
    exit()

selected_station = stations[selected_station_index]
print(f"Selected station: {selected_station}")

data = data[data['Station Identifier'] == selected_station]

analytes = [
    'Dissolved Oxygen (mg/L)', 
    'pH', 
    'Turbidity (NTU)', 
    'Specific Conductance (uS/cm)', 
    'Salinity (PSU)', 
    'Secchi Disk Depth (m)'
]

print("\nAvailable Analytes:")
for i, analyte in enumerate(analytes):
    print(f"{i + 1}. {analyte}")

selected_analyte_index = int(input("\nPlease select one analyte by number: ")) - 1

if selected_analyte_index not in range(len(analytes)):
    print("Invalid analyte selection.")
    exit()

selected_analyte = analytes[selected_analyte_index]
print(f"Selected analyte: {selected_analyte}")

filtered_data = data[['Station Identifier', 'Date', 'Time', selected_analyte]]
filtered_data = filtered_data.dropna(subset=[selected_analyte])

if filtered_data.empty:
    print("No data available for the selected station and analyte after removing missing values.")
    exit()

print("Converting Date and Time to EST...")
filtered_data['Datetime'] = filtered_data.apply(parse_datetime_utc_to_est, axis=1)
filtered_data = filtered_data.sort_values('Datetime')
filtered_data['Time_Difference_Hours'] = filtered_data['Datetime'].diff().dt.total_seconds() / 3600

# List of lower and upper bounds for time gaps
time_gap_thresholds = [(0.5, 480)]  # Example ranges for lower and upper bounds

for lower_bound, upper_bound in time_gap_thresholds:
    # Clipping based on both lower and upper bounds
    clipped_data = filtered_data[
        (filtered_data['Time_Difference_Hours'] >= lower_bound) & 
        (filtered_data['Time_Difference_Hours'] <= upper_bound)
    ]
    
    print(f"\nAnalyzing with Time Gap Bounds: {lower_bound} to {upper_bound} hours")
    print(f"Number of data points after clipping: {len(clipped_data)}")
    
    time_gap_summary = clipped_data['Time_Difference_Hours'].describe()
    print("Time Gap Analysis Summary (in Hours):")
    print(time_gap_summary)

    # Plotting the timeline of time differences for clipped data
    plt.figure(figsize=(10, 6))
    plt.plot(clipped_data['Datetime'], clipped_data['Time_Difference_Hours'], marker='o', linestyle='-', label=f'Time Gap {lower_bound}-{upper_bound} hours')
    
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Show major ticks every year
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the tick labels as years
    plt.gcf().autofmt_xdate()  # Rotate the date labels to prevent overlap

    plt.title(f'Time Differences Between Observations for {selected_station} ({selected_analyte})\nBounds: {lower_bound} to {upper_bound} hours')
    plt.xlabel('Year')
    plt.ylabel('Time Difference (hours)')
    plt.legend()
    plt.show()

    # Plot distribution of time gaps after clipping
    plt.figure(figsize=(10, 6))
    sns.histplot(clipped_data['Time_Difference_Hours'], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of Time Differences (in Hours) for {selected_station} ({selected_analyte})\nBounds: {lower_bound} to {upper_bound} hours')
    plt.xlabel('Time Difference (hours)')
    plt.ylabel('Frequency')
    plt.show()
