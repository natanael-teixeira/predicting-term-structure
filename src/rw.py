import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'data/1m.xlsx'
df = pd.read_excel(file_path)

date_col = 'Data'
rate_col = 'ETTJ - 1m'

df[date_col] = pd.to_datetime(df[date_col])

df_train = df[df[date_col] < '2022-01-01']

df_train['Retorno'] = df_train[rate_col].diff()
df_train.dropna(inplace=True)

mu = df_train['Retorno'].mean()
sigma = df_train['Retorno'].std()

np.random.seed(42)  
n_months = 12  
last_rate = df_train[rate_col].iloc[-1]

simulated_rates = [last_rate]
for _ in range(n_months):
    simulated_rate = simulated_rates[-1] + mu + sigma * np.random.normal()
    simulated_rates.append(simulated_rate)

dates_2021 = pd.date_range(start='2022-01-01', periods=n_months, freq='D')
df_2021_pred = pd.DataFrame({'Data': dates_2021, 'ETTJ - 1m': simulated_rates[1:]})

df_2022_real = df[df[date_col] >= '2022-01-01']

print("Taxas Reais de 2022:")
print(df_2022_real[[date_col, rate_col]])
print("\nTaxas Previstas para 2022:")
print(df_2021_pred)

