import pandas as pd
import numpy as np

data_path = r'./econometrics_data/'
df = pd.read_csv(f'{data_path}/FRED.csv', index_col=0, skiprows=[1], header=0).dropna(axis=1)
selected_columns = [
    'RPI',  # Real Person Income
    'INDPRO',  # Internal Product
    'UNRATE',  # Civilian Unemployment rate
    # 'HWIURATIO', #Ratio of Help-wanted / N. Unemployed
    'UEMPLT5',  # Civilian unemployed for less then 5 weeks
    'UEMPMEAN',  # Average Duration of unemployment (weeks)
    'DPCERA3M086SBEA',  # Real personal consumption
    'M2REAL',  # Real M2 Money stock
    # 'TB3MS', #3-months Treasury Rate
    'AAA',  # AAA-bonds rate
    'CPIAUCSL',  # Consumer Price Index: all goods
    'S&P div yield',  # Dividend Yield of S&P 500
    # 'S&P PE ratio', #PE ratio of S&P 500
]

###Start processing
df = df[selected_columns]
df.to_csv(f'{data_path}/FRED_clean.csv')
assert not (df.isnull().any().any())
df_deltas = {}
for col in df.columns:
    if (df[col] <= 0).sum() == 0:
        print(f'{col} -> log')
        df_deltas[col] = np.log(df[col].values[1:] / df[col].values[:-1])
    else:
        print(f'{col} -> var')
        df_deltas[col] = df[col].values[1:] - df[col].values[:-1]

df_deltas = pd.DataFrame(df_deltas, index=df.index[1:])
assert (df_deltas.isnull().sum().sum() == 0)
assert (np.isinf(df_deltas.values).sum().sum() == 0)
df_deltas.to_csv(f'{data_path}/FRED.csv')
