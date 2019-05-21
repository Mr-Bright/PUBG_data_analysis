import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
match = pd.read_csv('match.csv')

data = match.copy()
data = data[data['boosts'] < data['boosts'].quantile(0.98)]
plt.figure(figsize=(15,10))
plt.title("Boosts Item",fontsize=15)
sns.countplot(data['boosts'])
plt.show()


data = match.copy()
data = data[data['heals'] < data['heals'].quantile(0.98)]
plt.figure(figsize=(15,10))
plt.title("Heals Item",fontsize=15)
sns.countplot(data['heals'])
plt.show()