import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
match = pd.read_csv('match.csv')


data = match.copy()
data = data[data['walkDistance'] < data['walkDistance'].quantile(0.98)]
plt.figure(figsize=(15,10))
plt.title("Walking Distance Distribution",fontsize=15)
sns.distplot(data['walkDistance'])
plt.show()

print(str(len(match[match.walkDistance<500])/len(match)*100)+'% players move under 500 meters')