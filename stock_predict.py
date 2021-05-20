import pandas as pd
import numpy as np
import mplcursors
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
%matplotlib inline
# use %matplotlib qt instead of inline to visualize and hover the graph, but it is too slow

rcParams['figure.figsize'] = 20,10

df = pd.read_csv('C:/Users/dchandra/Downloads/Mine/Study/Projects/stock_predict/tataglobal.csv')
df.index = df['Date']
print(df.head())

plt.plot(df['Close'], label='Close Price history')
plt.gca().invert_xaxis()
mplcursors.cursor(hover=True)
plt.show()
