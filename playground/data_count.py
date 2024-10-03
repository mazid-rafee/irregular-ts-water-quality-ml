import pandas as pd
import os

# Define the base directory and file paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
file_path_usgs = os.path.join(base_dir, 'data', 'USGS PhysChem Top 100 - Time Series Research.txt')

# Read the USGS file (assuming it's tab-separated)
df_usgs = pd.read_csv(file_path_usgs, sep='\t')

# Rename columns based on the provided header
df_usgs.columns = ['Station_Identifier', 'Organization_Formal_Name', 'Activity_Identifier', 'Ammonia_mg_L',
                   'Ammonia_Nitrogen_mg_L', 'Organic_Carbon_mg_L', 'Chlorophyll_A_Corrected_For_Pheophytin_ug_L', 
                   'Dissolved_Oxygen_mg_L', 'Orthophosphate_mg_L', 'pH', 'Phosphate_Phosphorus_mg_L', 
                   'Phosphorus_mg_L', 'Salinity_PSU', 'Secchi_Disk_Depth_m', 'Specific_Conductance_uS_cm', 
                   'Temperature_Water_deg_C', 'Total_Suspended_Solids_mg_L', 'Turbidity_NTU', 'Date', 
                   'Time', 'Latitude', 'Longitude']

# Group by 'Station_Identifier' and count non-null entries for each analyte
station_wise_count_usgs = df_usgs.groupby('Station_Identifier').count()

# Select only the relevant analyte columns
station_wise_count_usgs = station_wise_count_usgs[['Ammonia_mg_L', 'Ammonia_Nitrogen_mg_L', 
                                                   'Organic_Carbon_mg_L', 'Chlorophyll_A_Corrected_For_Pheophytin_ug_L', 
                                                   'Dissolved_Oxygen_mg_L', 'Orthophosphate_mg_L', 'pH', 
                                                   'Phosphate_Phosphorus_mg_L', 'Phosphorus_mg_L', 'Salinity_PSU', 
                                                   'Secchi_Disk_Depth_m', 'Specific_Conductance_uS_cm', 
                                                   'Temperature_Water_deg_C', 'Total_Suspended_Solids_mg_L', 
                                                   'Turbidity_NTU']]

# Calculate the total count of non-null values for each station
station_wise_count_usgs['Total_Count'] = station_wise_count_usgs.sum(axis=1)

# Sort by the total count in descending order
station_wise_count_usgs_sorted = station_wise_count_usgs.sort_values(by='Total_Count', ascending=False)

# Define the output file path
output_file_path_usgs = os.path.join(base_dir, 'data', 'Station_wise_Analyte_Counts_USGS.txt')

# Save the sorted DataFrame to a text file (tab-separated format)
station_wise_count_usgs_sorted.to_csv(output_file_path_usgs, sep='\t', index=True)

print(f"Data successfully saved to {output_file_path_usgs}")
