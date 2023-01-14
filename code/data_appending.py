import os
import pandas as pd
import numpy as np

if __name__ == "__main__":
    print('Start!')
    dir_ = '../weather-forecasting-datavidia/data'
    for dirpath, dirname, filename in os.walk(dir_+'/merged'):
        print(f"There are {len(dirname)} directories and {len(filename)} file in '{dirpath}'.")
    a,b,c,d,e,f,g,h,i,j = np.zeros(10)
    arr_con = [a,b,c,d,e,f,g,h,i,j]
    entries = os.listdir(dir_+'/merged')
    entries.sort()
    for i in range(10):
        arr_con[i] = pd.read_csv(f'{dir_}/merged/{entries[i]}')
    appended_data = pd.concat(arr_con, axis=0)
    appended_data.to_csv('../weather-forecasting-datavidia/appended_data.csv')


