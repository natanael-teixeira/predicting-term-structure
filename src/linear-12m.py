import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler

file_path = 'data/12m.xlsx'
data = pd.read_excel(file_path)

data['Data'] = pd.to_datetime(data['Data'])
train_data = data[(data['Data'] >= '2010-01-01') & (data['Data'] <= '2021-12-31')]
test_data = data[data['Data'].dt.year == 2022]

columns_to_scale = ['Dólar', 'Selic', 'IPCA', 'CDS']

scaler = RobustScaler()

X_train = train_data[columns_to_scale]
y_train = train_data['ETTJ - 12m']

X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

X_test = test_data[columns_to_scale]

X_test_scaled = scaler.transform(X_test)

predictions_2022 = model.predict(X_test_scaled)

test_data['ETTJ - 12m_Predicted'] = predictions_2022

print("Previsões para ETTJ - 12m em 2022:")
print(test_data[['Data', 'ETTJ - 12m', 'ETTJ - 12m_Predicted']])



