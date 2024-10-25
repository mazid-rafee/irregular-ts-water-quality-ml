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

# To store the report, summary stats, and data for combined charts
report_data = []
combined_clipped_data = pd.DataFrame()
summary_stats_data = []

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

            # Data count after clipping
            clipped_data_count = len(clipped_data)
            
            if clipped_data.empty:
                continue

            # Add 'Analyte' column to clipped data
            clipped_data['Analyte'] = selected_analyte

            # Add the result to the report
            report_data.append({
                'Station': selected_station,
                'Analyte': selected_analyte,
                'Initial Count': initial_data_count,
                'Clipped Count': clipped_data_count
            })

            # Combine clipped data for final plots
            combined_clipped_data = pd.concat([combined_clipped_data, clipped_data])

            # Calculate and store summary statistics for each station-analyte pair
            summary_stats = clipped_data['Time_Difference_Hours'].describe()
            summary_stats_data.append({
                'Station': selected_station,
                'Analyte': selected_analyte,
                'Count': summary_stats['count'],
                'Mean': summary_stats['mean'],
                'Std': summary_stats['std'],
                'Min': summary_stats['min'],
                '25%': summary_stats['25%'],
                '50%': summary_stats['50%'],
                '75%': summary_stats['75%'],
                'Max': summary_stats['max']
            })

# Generate the combined report as a DataFrame
report_df = pd.DataFrame(report_data)
summary_stats_df = pd.DataFrame(summary_stats_data)

# Save the combined report and summary statistics to a single formatted text file
output_file_path = os.path.join(base_dir, 'combined_report_summary.txt')

with open(output_file_path, 'w') as f:
    f.write("Combined Report:\n")
    f.write(report_df.to_string(index=False))
    
    f.write("\n\nSummary Statistics for Each Station and Analyte:\n")
    f.write(summary_stats_df.to_string(index=False))

print(f"\nCombined report and summary statistics saved to {output_file_path}")

# Generate combined plots for all stations and analytes

# Plot combined timeline of time differences for clipped data
plt.figure(figsize=(12, 8))
for station_analyte, group_data in combined_clipped_data.groupby(['Station Identifier', 'Analyte']):
    plt.plot(group_data['Datetime'], group_data['Time_Difference_Hours'], marker='o', linestyle='-', label=f'{station_analyte}')

plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Show major ticks every year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the tick labels as years
plt.gcf().autofmt_xdate()  # Rotate the date labels to prevent overlap

plt.title(f'Combined Time Differences for USGS Stations and Analytes\nBounds: {time_gap_thresholds[0][0]} to {time_gap_thresholds[0][1]} hours')
plt.xlabel('Year')
plt.ylabel('Time Difference (hours)')
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
plt.tight_layout()
plt.show()

# Plot combined distribution of time gaps after clipping
plt.figure(figsize=(12, 8))
sns.histplot(combined_clipped_data['Time_Difference_Hours'], kde=True, bins=30, color='blue')
plt.title(f'Combined Distribution of Time Differences (in Hours)\nBounds: {time_gap_thresholds[0][0]} to {time_gap_thresholds[0][1]} hours')
plt.xlabel('Time Difference (hours)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
