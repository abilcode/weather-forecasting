import pandas as pd
import numpy as np
import sklearn
import optuna
import lightgbm


def objective(trial):
    
    # Invoke suggest methoxds of a Trial object to generate hyperparameters.

    lgbm_n_estimators = trial.suggest_int('n_estimators',3000,5000,step=100,log=False)
    lgbm_num_leaves = trial.suggest_int('num_leaves',1000,5000,step=100,log=False)
    lgbm_max_bin = trial.suggest_int('max_bin',1000,5000,step=100,log=False)
    lgbm_min_child_samples = trial.suggest_int('min_child_samples',1000,5000,step=100,log=False)
    lgbm_max_depth = trial.suggest_int('max_depth',1000,5000,step=100,log=False)
    lgbm_learning_rate = trial.suggest_float('learning_rate', 1e-10, 1, log=True)
    
    lgbm_boosting_type = trial.suggest_categorical('boosting_type',['gbdt'])
    regressor_obj = lightgbm.LGBMRegressor(learning_rate = lgbm_learning_rate,
                                            boosting_type=lgbm_boosting_type,
                                            min_child_samples=lgbm_min_child_samples,
                                            num_leaves=lgbm_num_leaves,
                                            max_bin=lgbm_max_bin,

                                            )

    data = pd.read_csv("../weather-forecasting-datavidia/appended_data.csv",index_col='time')
    data.city = data["city"].astype("category")
    
    feature = [x for x in data.columns if x not in ['rain_sum (mm)','sunrise (iso8601)','sunset (iso8601)','Unnamed: 0','city_max','city_min']]

    error = sklearn.model_selection.cross_val_score(regressor_obj,
                                                    data[feature],
                                                    data['rain_sum (mm)'],
                                                    scoring = 'neg_mean_squared_error',
                                                    cv=5)
    accuracy = error.mean()

    return accuracy


if __name__ == "__main__":
  data = pd.read_csv("../weather-forecasting-datavidia/appended_data.csv",index_col='time')
  print(data['rain_sum (mm)'].mean())
  print(data.columns)
  study = optuna.create_study(direction='maximize')  # Create a new study.
  study.optimize(objective, n_trials=4000)
 