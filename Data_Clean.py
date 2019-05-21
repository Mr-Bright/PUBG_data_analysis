import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('train_V2.csv')


print(train[train.isnull().any(axis=1)])
train = train.fillna(0)



match = train[(train.matchType=='solo')|(train.matchType=='duo')|(train.matchType=='squad')|(train.matchType=='solo-fpp')|(train.matchType=='duo-fpp')|(train.matchType=='squad-fpp')]
custom = train[~((train.matchType=='solo')|(train.matchType=='duo')|(train.matchType=='squad')|(train.matchType=='solo-fpp')|(train.matchType=='duo-fpp')|(train.matchType=='squad-fpp'))]


match.to_csv('match.csv',index=False)