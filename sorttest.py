import tensorflow as tf
import numpy as np
import pandas as pd
a = [1,2,3]
data = pd.DataFrame(a)
data.to_csv('some.csv',index=False,header=False)