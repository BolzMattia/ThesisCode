import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

path = r'./experiments/econometrics/results.csv'
df = pd.read_csv(path)

sns.lineplot(x='time', y='value', data=df, hue='statistics', style='method')
plt.show()
