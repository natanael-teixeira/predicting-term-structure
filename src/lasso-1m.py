import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler

file_path = 'data/1m.xlsx'
data = pd.read_excel(file_path)

data['Data'] = pd.to_datetime(data['Data'])
train_data = data[(data['Data'] >= '2010-01-01') & (data['Data'] <= '2020-12-31')]
test_data = data[data['Data'].dt.year == 2021]

columns_to_scale = ['Dollar', 'Selic', 'IPCA', 'CDS']

scaler = RobustScaler()

X_train = train_data[columns_to_scale]
y_train = train_data['ETTJ - 1m']

X_train_scaled = scaler.fit_transform(X_train)

model = Lasso(alpha=0.1)
model.fit(X_train_scaled, y_train)

X_test = test_data[columns_to_scale]

X_test_scaled = scaler.transform(X_test)

predictions_2021 = model.predict(X_test_scaled)

test_data['ETTJ - 1m_Predicted'] = predictions_2021

print("Previsões para ETTJ - 1m em 2021:")
print(test_data[['Data', 'ETTJ - 1m', 'ETTJ - 1m_Predicted']])
