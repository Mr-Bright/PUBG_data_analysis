import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
match = pd.read_csv('match.csv')


f,ax = plt.subplots(figsize=(len(match.columns), len(match.columns)))
sns.heatmap(match.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


k = 6
f,ax = plt.subplots(figsize=(11, 11))
cols = match.corr().nlargest(k, 'winPlacePerc')['winPlacePerc'].index
cm = np.corrcoef(match[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


kills = match.copy()
kills['killsCategories'] = pd.cut(kills['kills'], [-1, 0, 2, 5, 10, 60], labels=['0_kills','1-2_kills', '3-5_kills', '6-10_kills', '10+_kills'])
plt.figure(figsize=(15,8))
sns.boxplot(x="killsCategories", y="winPlacePerc", data=kills)
plt.show()