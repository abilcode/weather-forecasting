'''



'''
import warnings
warnings.simplefilter("ignore", UserWarning)

import re
# data related library
import numpy as np
import pandas as pd
import sklearn
import xgboost

# progres bar
from tqdm import tqdm
# silent warning

def model_search(data,n,city):
    mean = data.rain_summm.mean()
    c = city
    feature = [i for i in data.columns if i not in ['rain_summm']]
    num_feature = [i for i in feature if data[i].dtypes!='O']
    train = data.loc[data.index < '2020-12-31']
    x_train = train[num_feature]
    scaler = sklearn.preprocessing.RobustScaler()
    y_train =train['rain_summm']
    scaler.fit(y_train.values.reshape(-1, 1))
    y_train = scaler.transform(y_train.values.reshape(-1, 1))

    test = data.loc[data.index >= '2020-12-31']
    x_test = test[num_feature]
    y_test = test['rain_summm']
    y_test = scaler.transform(y_test.values.reshape(-1, 1))
    
    #x_train.city = x_train.city.astype('category')
    #x_test.city = x_test.city.astype('category')
    xgb_list_tuned = {'params':[],
                        'score':[]}

    xgb =  xgboost.XGBRegressor(random_state=0)
    grid_xgb = {
                'max_depth': [3, 4, 5, 6, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300, 400, 500],
                'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9]
            }
    print('mean',mean)
    tscv = sklearn.model_selection.TimeSeriesSplit(n_splits=5)
    xgb_cv = sklearn.model_selection.RandomizedSearchCV(xgb, grid_xgb, scoring = 'neg_mean_squared_error',cv=5, n_jobs=-1,n_iter=50)
    for n in range(n):
        xgb_cv.fit(x_train, y_train.ravel())
        print(f"Experiment number {n+1}")     
        print(xgb_cv.best_params_)
        xgb_list_tuned['params'].append(xgb_cv.best_params_)
        
        with tqdm(total=100) as pbar:
            for i in range(100):
                y_pred = xgb_cv.predict(x_test)
                pbar.update(1)
       
        y_pred = np.array([0 if x < 0 else x for x in y_pred])
        
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
    
    