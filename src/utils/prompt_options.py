from stations_and_analytes import usgs_stations, dbhydro_stations, usgs_analytes_with_associated, dbhydro_analytes_with_associated

def choose_station(dataset):
    
    if dataset == 'USGS':
        stations = usgs_stations
    elif dataset == 'DbHydro':
        stations = dbhydro_stations
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Choose between 'USGS' or 'DbHydro'.")
    
    print(f"Available stations for {dataset}:")
    for i, station in enumerate(stations):
        print(f"{i + 1}: {station}")
    
    station_idx = int(input("Enter the number corresponding to the station: ")) - 1
    return stations[station_idx]


def choose_analytes(dataset):

    if dataset == 'USGS':
        analytes_with_associated = usgs_analytes_with_associated
    elif dataset == 'DbHydro':
        analytes_with_associated = dbhydro_analytes_with_associated
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Choose between 'USGS' or 'DbHydro'.")
    
    print(f"Available main analytes for {dataset}:")
    for i, analyte_set in enumerate(analytes_with_associated):
        print(f"{i + 1}: {analyte_set[0]}")
    
    analyte_idx = int(input("Enter the number corresponding to the main analyte: ")) - 1
    selected_set = analytes_with_associated[analyte_idx]  
    main_analyte = selected_set[0].replace('Main Analyte: ', '')
    available_associated_analytes = [a.replace('Associated: ', '') for a in selected_set[1:]]
    
    print(f"Available associated analytes for {main_analyte}:")
    for i, analyte in enumerate(available_associated_analytes):
        print(f"{i + 1}: {analyte}")
    
    selected_indices = input(
        "Enter the numbers corresponding to the associated analytes you want to select (comma-separated), or press Enter to select none: "
    )
    
    if selected_indices:
        selected_indices = [int(idx) - 1 for idx in selected_indices.split(',')]
        associated_analytes = [available_associated_analytes[i] for i in selected_indices]
    else:
        associated_analytes = []

    return main_analyte, associated_analytes