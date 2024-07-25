import pandas as pd

df = pd.read_excel('data/12m.xlsx')

desc = df.describe().T

print(desc)

