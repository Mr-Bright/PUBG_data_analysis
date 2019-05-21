import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
match = pd.read_csv('match.csv')

plt.figure(figsize=(15,10))
plt.title("Damage Dealt",fontsize=15)
sns.distplot(match['damageDealt'])
plt.show()

print(str(len(match[(match.kills==0)&(match.damageDealt==0)])/len(match)*100)+'% players are both 0 kill and 0 damage')