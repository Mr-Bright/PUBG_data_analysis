import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.cluster as KMeans
import warnings
warnings.filterwarnings("ignore")
match = pd.read_csv('match.csv')

km = np.array(match[['kills','walkDistance','boosts','winPlacePerc']])
clf = KMeans.KMeans(n_clusters=3).fit(km)
print(clf.cluster_centers_)
match['label'] = clf.labels_

m0 = match[match.label==0]
m1 = match[match.label==1]
m2 = match[match.label==2]

ax = plt.subplot(111, projection='3d')
ax.scatter(clf.cluster_centers_[0][0],clf.cluster_centers_[0][1],clf.cluster_centers_[0][2], c='r',marker='o',s = len(m0)/len(match)*50*len(m0)/len(match)*100)
ax.scatter(clf.cluster_centers_[1][0],clf.cluster_centers_[1][1],clf.cluster_centers_[1][2], c='g',marker='o',s = len(m1)/len(match)*50*len(m1)/len(match)*100)
ax.scatter(clf.cluster_centers_[2][0],clf.cluster_centers_[2][1],clf.cluster_centers_[2][2], c='b',marker='o',s = len(m2)/len(match)*50*len(m2)/len(match)*100)
ax.set_zlabel('boosts')
ax.set_ylabel('walkDistance')
ax.set_xlabel('kills')
plt.show()
