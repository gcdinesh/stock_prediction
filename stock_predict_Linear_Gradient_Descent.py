import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from pandas import DataFrame
from fastai.tabular.all import add_datepart
#%matplotlib inline
# use %matplotlib qt instead of inline to visualize and hover the graph, but it is too slow
#instead of using this command here go to tools-> preferences -> ipython console -> graphics -> backend(dropdown) to inline

rcParams['figure.figsize'] = 20,10
pd.set_option('display.max_columns', None)
df1 = pd.read_csv('C:/Users/dchandra/Downloads/Mine/Study/Projects/stock_prediction/tataglobal.csv')
df = DataFrame(df1)
df = df.sort_values('Date')
df.index = df['Date']
df = add_datepart(df, 'Date')
actualVals = df['Close'].tolist()
actualDayOfWeekVals = df['Dayofweek']
df.reset_index(drop = True, inplace=True)
df = df.drop(['Open', 'High', 'Low', 'Last', 'Total Trade Quantity', 'Turnover (Lacs)', 'Elapsed', 'Is_month_end',
'Is_month_start',
'Is_quarter_end',
'Is_quarter_start',
'Is_year_end',
'Is_year_start'], axis = 1)


class Linear(object):
    def __init__(self, weight, bias, learningRate, numberOfIteration, trainDataLen):
        self.weight = weight
        self.bias = bias
        self.lr = learningRate
        self.n = numberOfIteration
        self.predictVals = list()
        self.costList = list()
        self.trainDataLen = trainDataLen
        
    def updatedWeightAndBias(self):
        weightDerivative = 0.0
        biasDerivative = 0.0
        for i in range(len(df)):
            weightDerivative += (-2 * df['Dayofweek'][i] * (df['Close'][i] - self.weight * df['Dayofweek'][i] - self.bias))
            biasDerivative += -2 * (df['Close'][i] - self.weight * df['Dayofweek'][i] - self.bias)
                   
        self.weight -= (weightDerivative / len(df)) * self.lr
        self.bias -= (biasDerivative / len(df)) * self.lr
        
    def normalize(self):
        for i in df:
            mean = np.mean(df[i])
            df[i] -= mean
            minMax = np.max(df[i]) - np.min(df[i])
            df[i] /= minMax
            
    def calculateCost(self):
        cost = 0.0
        for i in range(len(df)):
            cost += (df['Close'][i] - (self.weight * df['Dayofweek'][i] + self.bias)) * (df['Close'][i] - (self.weight * df['Dayofweek'][i] + self.bias))
            
        return cost / len(df)
    
    def train(self):
        for i in range(self.n):
            #self.normalize()
            self.updatedWeightAndBias()
            cost = self.calculateCost()
            self.costList.append(np.sqrt(cost))
            print("iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2f}".format(i, self.weight, self.bias, np.sqrt(cost)))
        
    def predict(self):
        for i in range(trainDataLen, len(actualVals)):
            predval = self.weight * actualDayOfWeekVals[i] + self.bias
            self.predictVals.append(predval)
            
    def plotGraph(self):
        plt.plot(self.costList)
        plt.show()
        plt.plot(actualVals)
        plt.plot(actualVals[:self.trainDataLen] + self.predictVals)
        plt.show()

trainDataLen = 1000
df = df[:trainDataLen]

ravg = Linear(0, 0, 0.05, 170, trainDataLen)
ravg.train()
ravg.predict()
ravg.plotGraph()