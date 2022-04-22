import pandas as pd

# Load the dataset
data_file = '../data/airline_delay_dataframe.csv'
df = pd.read_csv(data_file)

# Dataset from http://www.partow.net/miscellaneous/airportdatabase/index.html#Downloads
df_loc = pd.read_csv('../data/GlobalAirportDatabase.txt', delimiter=":", header=None)
columns = ['ICAO Code','IATA Code', 'Airport Name', 'City/Town', 'Country', 'Latitude Degrees', 'Latitude Minutes', 'Latitude Seconds', '	Latitude Direction', '	Longitude Degrees', 'Longitude Minutes', 'Longitude Seconds', 'Longitude Direction', 'Altitude', 'lat', 'long']
df_loc = df_loc.set_axis(columns, axis=1)
df_loc.drop(['ICAO Code', 'Airport Name', 'City/Town', 'Country', 'Latitude Degrees', 'Latitude Minutes', 'Latitude Seconds', '	Latitude Direction', '	Longitude Degrees', 'Longitude Minutes', 'Longitude Seconds', 'Longitude Direction', 'Altitude', ], axis=1, inplace=True)

# Adding airport longtitude and latitude to dataframe
for index, row in df.iterrows():
    origin_airport = row['Origin']
    dest_airport = row['Dest']
    try:
        origin_lat = df_loc.loc[df_loc['IATA Code'] == origin_airport]['lat']
        origin_long = df_loc.loc[df_loc['IATA Code'] == origin_airport]['long']
        dest_lat = df_loc.loc[df_loc['IATA Code'] == dest_airport]['lat']
        dest_long = df_loc.loc[df_loc['IATA Code'] == dest_airport]['long']
        if (origin_lat.values[0] or origin_long.values[0] or dest_lat.values[0] or dest_long.values[0]) == 0.0:
            pass
        else:
            df.loc[index, 'origin_long'] = origin_long.values[0]
            df.loc[index, 'origin_lat'] = origin_lat.values[0]
            df.loc[index, 'dest_lat'] = dest_lat.values[0]
            df.loc[index, 'dest_long'] = dest_long.values[0]
    except IndexError:
        pass

# Removing unfind locations
df = df[df.origin_long != 0.0]

# Exporting to csv
df = df[['Month', 'origin_long', 'origin_lat', 'dest_lat', 'dest_long', 'ArrDelay']]
df.to_csv('flight_routes.csv', index=False)