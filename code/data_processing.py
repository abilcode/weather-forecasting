
import os
import numpy as np
import pandas as pd

def data_processing(a,b,file):
    ### Columns to join from hourly data 
    cols_hourly = ['pressure_msl (hPa)',
            'surface_pressure (hPa)', 'snowfall (cm)', 'cloudcover (%)',
            'cloudcover_low (%)', 'cloudcover_mid (%)', 'cloudcover_high (%)',
            'shortwave_radiation (W/m²)', 'direct_radiation (W/m²)',
            'diffuse_radiation (W/m²)', 'direct_normal_irradiance (W/m²)',
            'vapor_pressure_deficit (kPa)', 'soil_temperature_0_to_7cm (°C)',
            'soil_temperature_7_to_28cm (°C)', 'soil_temperature_28_to_100cm (°C)',
            'soil_temperature_100_to_255cm (°C)', 'soil_moisture_0_to_7cm (m³/m³)',
            'soil_moisture_7_to_28cm (m³/m³)', 'soil_moisture_28_to_100cm (m³/m³)',
            'soil_moisture_100_to_255cm (m³/m³)']

    data1 = a
    data2 = b
    
    data1['time'] = pd.to_datetime(data1['time'])
    data1 = data1.set_index('time')
    
    data2['time'] = pd.to_datetime(data2['time'])
    data2 = data2.set_index('time')

    ### Dropiing Missing Values
    data1=data1.dropna(thresh=11,axis=0)
    data1 = data1.drop('winddirection_10m_dominant (°)',axis=1)
    merged=pd.merge(data1,data2, how='inner', left_index=True, right_index=True)
    merged['month'] = merged.index.month
    merged['month'] = merged.index.year
    merged.to_csv(f'{file}/merged_{data1.city.unique()}.csv')

if __name__ == "__main__":
    
    print("Data Processing Started!")
    ### get to know the data :
    dir_ = '../data'
    for dirpath, dirname, filename in os.walk(dir_+'/per_city'):
        print(f"There are {len(dirname)} directories and {len(filename)} file in '{dirpath}'.")
    entries = os.listdir(dir_+'/per_city')
    entries.sort()
    entries_hourly = os.listdir(dir_+'/train_per_city')
    entries_hourly.sort()
    #print(entries)
    #print(entries_hourly)
    for i in range(len(entries)):
        print('===+++==='*10)
        print(f"{dir_}/per_city/{entries[i]}")
        print(f"{dir_}/train_per_city/{entries_hourly[i]}")
        df = pd.read_csv(f'{dir_}/per_city/{entries[i]}')
        df_ = pd.read_csv(f'{dir_}/train_per_city/{entries_hourly[i]}')
        data_processing(df,df_,file='../data/merged')
        print(f'{entries[i]} df has been merged!')
        print('===+++==='*10)
    print("Data have been proceeded!")
   



