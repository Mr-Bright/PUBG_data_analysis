import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
match = pd.read_csv('match.csv')

data = match.copy()
data.loc[data['weaponsAcquired'] > 9] = '9+'
plt.figure(figsize=(15,10))
sns.countplot(data['weaponsAcquired'].astype('str').sort_values())
plt.title("weapon Count",fontsize=15)
plt.show()


persent = 0.98
print(str(persent*100)+'% players\'s weapon count are under '+str(match['kills'].quantile(persent)))
print(str(len(match[(match.kills==0)&(match.damageDealt==0)&(match.weaponsAcquired==0)])/len(match)*100)+'% players have a pretty bad experience')