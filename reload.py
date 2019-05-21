import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('test_V2.csv')


print(train[train.isnull().any(axis=1)])



