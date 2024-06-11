import pandas as pd
import pandas as pd
import numpy as np

train_df = pd.read_csv('train.csv')
train_df

b1 = train_df['b1']
b11 = train_df['b11']
b12 = train_df['b12']
b2 = train_df['b2']
b3 = train_df['b3']
b4 = train_df['b4']
b5 = train_df['b5']
b6 = train_df['b6']
b7 = train_df['b7']
b8 = train_df['b8']
b8a = train_df['b8_a']
b9 = train_df['b9']

train_df['NDVI'] = (train_df['b8'] - train_df['b4']) / (train_df['b8'] + train_df['b4'])
train_df['EVI'] = 2.5 * ((train_df['b8'] - train_df['b4']) / (train_df['b8'] + 6 * train_df['b4'] - 7.5 * train_df['b2'] + 1.01))
train_df['NDWI'] = (train_df['b3'] - train_df['b8']) / (train_df['b3'] + train_df['b8'])
train_df['SAVI'] = (train_df['b8'] - train_df['b4']) * (1 + 0.5) / (train_df['b8'] + train_df['b4'] + 0.5)
train_df['MSAVI'] = (2 * train_df['b8'] + 1 - ( (2 * train_df['b8'] + 1) ** 2 - 8 * (train_df['b8'] - train_df['b4'])) ** (1 / 2)) / 2
train_df['GNDVI'] = (train_df['b8'] - train_df['b3']) / (train_df['b8'] + train_df['b3'])
train_df['RENDVI'] = (train_df['b8'] - train_df['b5']) / (train_df['b8'] + train_df['b5'])
train_df['NDMI'] = (train_df['b8'] - train_df['b11']) / (train_df['b8'] + train_df['b11'])
train_df['GRVI'] = (train_df['b3'] - train_df['b4']) / (train_df['b3'] + train_df['b4'])
train_df['TVI'] = ( (train_df['b8'] - train_df['b4']) / (train_df['b8'] + train_df['b4'] + 0.5) ) ** (1 / 2)
train_df['MCARI'] = ((train_df['b5'] - train_df['b4']) - 0.2 * (train_df['b5'] - train_df['b3'])) / (train_df['b5'] / train_df['b4'])
train_df['BSI'] =  ((train_df['b11'] + train_df['b4']) - (train_df['b8'] + train_df['b2'])) / ((train_df['b11'] + train_df['b4']) + (train_df['b8'] + train_df['b2']))
train_df['NBR'] = (train_df['b8'] - train_df['b12']) / (train_df['b8'] + train_df['b12'])
train_df['MSI'] = train_df['b11'] / train_df['b8']


feature = [
    'b1', 'b11', 'b12', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b8_a', 'b9', 'NDVI', 'EVI', 'NDWI', 'SAVI', 'MSAVI',
    'GNDVI', 'RENDVI', 'NDMI', 'GRVI', 'TVI', 'MCARI', 'BSI', 'NBR', 'MSI'
]

target = ['nforest_type']

X = train_df[feature]
y = train_df[target]

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)

#----------------------------------------------------------------------------------------

y_train_2 = y_train.copy()

num_list = []
for i in y_train_2['nforest_type']:
  if i == "MDF":
    num_list.append(0)
  elif i == "DDF":
    num_list.append(1)
  else:
    num_list.append(2)
y_train_2['nforest_type'] = num_list

y_val_2 = y_val.copy()

num_list = []
for i in y_val_2['nforest_type']:
  if i == "MDF":
    num_list.append(0)
  elif i == "DDF":
    num_list.append(1)
  else:
    num_list.append(2)
y_val_2['nforest_type'] = num_list

#----------------------------------------------------------------------------------------

import catboost as cb
from sklearn.metrics import mean_squared_error
import optuna

def objective(trial):
    params = {
        "iterations": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 10),
        # "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-2, 0.1, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    model = cb.CatBoostRegressor(**params, silent=True)
    model.fit(X_train, y_train_2)
    predictions = model.predict(X_val)
    rmse = mean_squared_error(y_val_2, predictions, squared=False)
    return rmse



study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print('Best hyperparameters:', study.best_params)
print('Best RMSE:', study.best_value)