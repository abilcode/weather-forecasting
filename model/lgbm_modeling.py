'''



'''
import warnings
warnings.simplefilter("ignore", UserWarning)

import re
# data related library
import numpy as np
import pandas as pd
import sklearn
import lightgbm

# progres bar
from tqdm import tqdm
# silent warning

def model_search(data,n,city):
    mean = data.rain_summm.mean()
    c = city
    feature = [i for i in data.columns if i not in ['rain_summm']]
    train = data.loc[data.index < '2020-12-31']
    x_train = train[feature]
    scaler = sklearn.preprocessing.RobustScaler()
    y_train =train['rain_summm']
    scaler.fit(y_train.values.reshape(-1, 1))
    y_train = scaler.transform(y_train.values.reshape(-1, 1))

    test = data.loc[data.index >= '2020-12-31']
    x_test = test[feature]
    y_test = test['rain_summm']
    y_test = scaler.transform(y_test.values.reshape(-1, 1))
    
    x_train.city = x_train.city.astype('category')
    x_test.city = x_test.city.astype('category')
    lgb_list_tuned = {'params':[],
                        'score':[]}

    lgb =  lightgbm.LGBMRegressor(random_state=0)
    grid_lgb = {'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'n_estimators' : [900, 1100, 1300, 1500, 2000, 2500, 3000,3500],
                'learning_rate' : [0.01, 0.025, 0.05, 0.1, 0.2, 0.25, 0.3],
                'max_depth':[7,9,11,15],
                'min_child_weight': [1,3,5,7,9],
                'reg_lambda': [0.01, 0.03, 0.05, 0.07, 0.1, 0.5, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                'reg_lambda':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]} 
    print('mean',mean)
    tscv = sklearn.model_selection.TimeSeriesSplit(n_splits=5)
    lgb_cv = sklearn.model_selection.RandomizedSearchCV(lgb, grid_lgb, scoring = 'neg_mean_squared_error',cv=tscv, n_jobs=-1,n_iter=50)
    for n in range(n):
        lgb_cv.fit(x_train, y_train.ravel(),verbose=-1)
        print(f"Experiment number {n+1}")     
        print(lgb_cv.best_params_)
        lgb_list_tuned['params'].append(lgb_cv.best_params_)
        
        with tqdm(total=100) as pbar:
            for i in range(100):
                y_pred = lgb_cv.predict(x_test)
                pbar.update(1)
        if c != 'q':
            y_pred = np.array([0 if x < 0 else x for x in y_pred])
        else :
            y_pred = np.array([mean if x < 0 else x for x in y_pred])
        print(sklearn.metrics.mean_squared_error(scaler.inverse_transform(y_test.reshape(1, -1)), scaler.inverse_transform(y_pred.reshape(1, -1))))
        print("------------------------------------------------------------")
    

if __name__ == "__main__":
    print('Running...')
    data = pd.read_csv("/Users/abilfad/Documents/CODE/weather-forecasting-datavidia/appended_data.csv",index_col='time')
    data.index = pd.to_datetime(data.index)
    data_ = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    #data_ = pd.read_csv("/Users/abilfad/Documents/CODE/weather-forecasting-datavidia/data/merged_test/8merged_test['t'].csv")
    #data_.index = data_.time
    #data_.index = pd.to_datetime(data_.index)
    data_.drop(['sunriseiso8601', 'sunsetiso8601', 'city_max', 'city_min'],axis=1,inplace=True)
    #data_.drop(['time','sunrise (iso8601)', 'sunset (iso8601)', 'city_max', 'city_min'],axis=1,inplace=True)
    city_obj = ['su','si','u','le','p','lh','b','t','sa','q']
    for c in city_obj :
        print('=='*14,c,'=='*14)
        model_search(data_[data_['city']==c],30,c)
    
    