import pandas as pd
import numpy as np
import mplcursors
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from pandas import DataFrame
#%matplotlib inline
# use %matplotlib qt instead of inline to visualize and hover the graph, but it is too slow
#instead of using this command here go to tools-> preferences -> ipython console -> graphics -> backend(dropdown) to inline

rcParams['figure.figsize'] = 20,10

df = pd.read_csv('C:/Users/dchandra/Downloads/Mine/Study/Projects/stock_prediction/tataglobal.csv')
df = df.sort_values('Date')
df.index = df['Date']
mplcursors.cursor(hover=True)

class RollingAverage(object):
    
    def __init__(self, n):
        self.avg = 0
        self.n = n
        self.sum = 0
        self.predictVals = list()
        self.actualVals = df['Close'].tolist()
        
    def calculateInitialNValues(self):
        for i in range(self.n):
            self.sum += df['Close'][i]
    
    def predict(self):
        self.calculateInitialNValues()
        av = 0
        for i in range(self.n, len(df)):
            self.sum += av
            av = self.sum / self.n
            self.predictVals.append(av)
            self.sum = self.sum - df['Close'][i - self.n]
        
    def plotGraph(self):
        actDf = DataFrame(self.actualVals)
        preDf = DataFrame(self.actualVals[:self.n-1] + self.predictVals)
        plt.plot(actDf)
        plt.plot(preDf)
        mplcursors.cursor(hover=True)
        plt.show()
        

ravg = RollingAverage(500)
ravg.predict()
ravg.plotGraph()