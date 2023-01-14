import pandas as pd
import numpy as np


def seperate_city(data,file='file.csv'):
    city = data.city.unique()
    city_1,city_2,city_3,city_4,city_5,city_6,city_7,city_8,city_9,city_10 = np.zeros(10)
    city_arr = [city_1,city_2,city_3,city_4,city_5,city_6,city_7,city_8,city_9,city_10]
    for i in range(len(city)) :
        city_arr[i] = data[data['city']==city[i]]
        city_arr[i].to_csv(f'{file}/train_hourly_{city[i]}.csv')
        print(f'{city[i]} has been extracted to csv format!')

if __name__ == "__main__":

    train = pd.read_csv('/Users/abilfad/Downloads/weather-forecasting-datavidia/data/train.csv')
    train['time'] = pd.to_datetime(train['time'])
    train_hourly = pd.read_csv('/Users/abilfad/Downloads/weather-forecasting-datavidia/data/train_hourly.csv')
    train_hourly['time'] = pd.to_datetime(train_hourly['time'])

    train = train.set_index('time')
    train_hourly = train_hourly.set_index('time')

    print('Seperating city started!')
    seperate_city(train_hourly,'/Users/abilfad/Downloads/weather-forecasting-datavidia/data/train_per_city')
    print('Finished!')
    #seperate_city(train_hourly)