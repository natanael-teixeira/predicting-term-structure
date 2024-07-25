import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from typing import Callable
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats
from dm_test import dm_test

def rolling_window(data, window_rate, horizon):
    """ Gera Janela Móvel

    Args:
        data (_type_): df
        window_rate (_type_): qual % dos dados será usado pra treino 
        horizon (_type_): horizonte de previsão (ex. t+1)

    Returns:
        _type_: lista de tuplas, 2 itens, primeiro e janela movel e o segundo horizonte de previsao
    """
    window_size = int(len(data) * window_rate)
    windows = []
    # for i in range(len(data) - window_size + 1):
    i = 0
    while i <= (len(data) - window_size - horizon - 1):
        windows.append((
            data[i:i+window_size],
            data.loc[[i + window_size + horizon]]
        ))
        i += 1
    return windows

def create_model(model_function:Callable, scaler_function:Callable, data:pd.DataFrame, y:str, horizon:pd.DataFrame, **kwargs):
    """_summary_

    Args:
        model_function (Callable): modelo a ser rodado
        scaler_function (Callable): normalização
        data (pd.DataFrame): df
        y (str): nome da coluna da variavel target
        horizon (pd.DataFrame): df do horizonte de previsao

    Returns:
        _type_: tupla valor previsto e o valor real
    """
    X_train = data.drop(y, axis=1)
    y_train = data[y]

    
    random_walk = y_train.values[-1]
    


    scaler = scaler_function()

    X_train = scaler.fit_transform(X_train)

    model = model_function(**kwargs)
    model.fit(X_train, y_train)

    X_horizon = horizon.drop(y, axis=1)
    y_horizon = horizon.reset_index().at[0, y]
    

    predict = model.predict(scaler.transform(X_horizon))

    return (predict, y_horizon, random_walk)

file_path = 'data/1m.xlsx'
data = pd.read_excel(file_path)

data.drop(columns='Data', inplace=True)

df_windows = rolling_window(data, 0.8, 30)
random = []
predicted = []
real = []
for k in df_windows:
    p, r, rw = create_model(LinearRegression, RobustScaler, k[0], "ETTJ - 1m", k[1])
    random.append(rw)
    predicted.append(p[0])
    real.append(r)


rmse_linear = root_mean_squared_error(real, predicted)
rmse_rw = root_mean_squared_error(real, random)

print(rmse_linear)
print(rmse_rw)

dm_rw = dm_test(real, predicted, random, h = 30, crit="MSE")
print (dm_rw)


