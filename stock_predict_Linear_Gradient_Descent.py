import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from pandas import DataFrame
from fastai.tabular.all import add_datepart
import math
#%matplotlib inline
# use %matplotlib qt instead of inline to visualize and hover the graph, but it is too slow
#instead of using this command here go to tools-> preferences -> ipython console -> graphics -> backend(dropdown) to inline
rcParams['figure.figsize'] = 20,10
pd.set_option('display.max_columns', None)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

df = pd.read_csv('C:/Users/dchandra/Downloads/Mine/Study/Projects/stock_prediction/ITC.BO.csv')
df = df.sort_values('Date')
df.index = df['Date']
df = add_datepart(df, 'Date')
df.reset_index(drop = True, inplace = True)
df = df.drop(['Open', 'High', 'Low', 'Elapsed', 'Adj Close', 'Volume',
'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start',
'Is_year_end', 'Is_year_start'], axis = 1)

class Linear(object):
    def __init__(self, learningRate, numberOfIteration, trainDataLen):
        self.weights = np.zeros(len(df.T))
        self.costs = np.zeros(len(df.T))
        self.bias = 1.0
        self.lr = learningRate
        self.n = numberOfIteration
        self.predictVals = list()
        self.costList = list()
        self.trainDataLen = trainDataLen
        
    def updatedWeightAndBias(self):
        c = 0
        for f in df:
            weight = 0.0
            for i in range(len(df)):
                    weight += (-2 * df[f][i] * (df['Close'][i] - self.weights[c] * df[f][i] - self.bias))
                    
            self.weights[c] -= (weight / len(df)) * self.lr
            c += 1

        
    def normalize(self):
        for i in df:
            df[i] -= np.min(df[i])
            minMax = np.max(df[i]) - np.min(df[i])
            if(minMax != 0):
                df[i] /= minMax
    
    def calculateCost(self):
        c = 0
        for f in df:
            cost = 0.0
            for i in range(len(df)):
                    cost += (df['Close'][i] - (self.weights[c] * df[f][i] + self.bias)) * (df['Close'][i] - (self.weights[c] * df[f][i] + self.bias))
                    
            self.costs[c] = math.sqrt(cost / len(df))
            c += 1
    
    def train(self):
        for i in range(self.n):
            self.updatedWeightAndBias()
            self.calculateCost()
            print(self.costs)
        
    def predict(self):
        predlist = list()
        for i in range(trainDataLen, len(actualVals)):
            c = 0
            predval = 0.0
            for f in df:
                predval += self.weights[c] * df2[f][i]
                c += 1
            predlist.append(predval + self.bias)
                
        predlist -= np.min(predlist)
        minMax = np.max(predlist) - np.min(predlist)
        if(minMax != 0):
            predlist /= minMax
        for i in range(0, trainDataLen):
            self.predictVals.append(0)
        for i in range(0, len(predlist)):
            self.predictVals.append(predlist[i])
            
    def plotGraph(self):
        plt.show()
        plt.plot(actualVals)
        plt.plot(self.predictVals)
        plt.show()

trainDataLen = 6094
ravg = Linear(0.9, 10, trainDataLen)
ravg.normalize()
actualVals = df['Close'].tolist()
df2 = DataFrame(df)
df = df[:trainDataLen]
print(df.head)
ravg.train()
ravg.predict()
ravg.plotGraph()