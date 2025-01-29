import numpy as np
import pandas as pd


a = np.array(list(range(100)))
print(a)

dt = dict(zip(a, a))

df = pd.DataFrame(index=list(range(100)), data=dt)
print(df)