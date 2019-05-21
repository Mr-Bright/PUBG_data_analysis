import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
match = pd.read_csv('match.csv')

data = match.copy()
data.loc[data['kills'] > data['kills'].quantile(0.99)] = '8+'
plt.figure(figsize=(15,10))
sns.countplot(data['kills'].astype('str').sort_values())
plt.title("Kill Count",fontsize=15)
plt.show()

print("max kills="+str(match.kills.max()))
persent = 0.98
print(str(persent*100)+'% players\' kill number are under '+str(match['kills'].quantile(persent)))
print(str(len(match[match.kills==0])/len(match)*100)+'% players are 0 kill')