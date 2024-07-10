import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler

# Carregar os dados
file_path = 'data/3m.xlsx'
data = pd.read_excel(file_path)

# Filtrar os dados de 2010 a 2020 para treino e 2021 para teste
data['Data'] = pd.to_datetime(data['Data'])
train_data = data[(data['Data'] >= '2010-01-01') & (data['Data'] <= '2020-12-31')]
test_data = data[data['Data'].dt.year == 2021]

# Selecionar as colunas para normalização
columns_to_scale = ['Dollar', 'Selic', 'IPCA', 'CDS']

# Inicializar o RobustScaler
scaler = RobustScaler()

# Separar as variáveis independentes e a variável dependente para treino
X_train = train_data[columns_to_scale]
y_train = train_data['ETTJ - 1m']

# Ajustar o scaler nos dados de treinamento e transformar os dados
X_train_scaled = scaler.fit_transform(X_train)

# Inicializar e treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Preparar os dados de 2021 para previsão
X_test = test_data[columns_to_scale]

# Normalizar os dados de 2021 usando o scaler ajustado anteriormente
X_test_scaled = scaler.transform(X_test)

# Fazer previsões para 2021
predictions_2021 = model.predict(X_test_scaled)

# Adicionar as previsões ao dataframe de teste
test_data['ETTJ - 1m_Predicted'] = predictions_2021

# Exibir as previsões
print("Previsões para ETTJ - 1m em 2021:")
print(test_data[['Data', 'ETTJ - 1m', 'ETTJ - 1m_Predicted']])



