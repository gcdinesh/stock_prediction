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
df1 = pd.read_csv('C:/Users/dchandra/Downloads/Mine/Study/Projects/stock_prediction/ADANITRANS.BO.csv')
df = DataFrame(df1)
df = df.sort_values('Date')
df.index = df['Date']
df = add_datepart(df, 'Date')
df.reset_index(drop = True, inplace=True)

#df = df.drop(['Open', 'High', 'Low', 'Last', 'Total Trade Quantity', 'Turnover (Lacs)', 'Elapsed', 'Is_month_end',
#'Is_month_start',
#'Is_quarter_end',
#'Is_quarter_start',
#'Is_year_end',
#'Is_year_start'], axis = 1)

df = df.drop(['Open', 'High', 'Low', 'Elapsed', 'Adj Close', 'Volume',
'Is_month_end',
'Is_month_start',
'Is_quarter_end',
'Is_quarter_start',
'Is_year_end',
'Is_year_start'], axis = 1)

class Linear(object):
    def __init__(self, learningRate, numberOfIteration, trainDataLen):
        self.weightDOWDerivative = 0.0
        self.weightYearDerivative = 0.0
        self.weightMonthDerivative = 0.0
        self.weightWeekDerivative = 0.0
        self.weightDayDerivative = 0.0
        self.weightDayofyearDerivative = 0.0
        self.bias = 1.0
        self.lr = learningRate
        self.n = numberOfIteration
        self.predictVals = list()
        self.costList = list()
        self.trainDataLen = trainDataLen
        
    def updatedWeightAndBias(self):
        weightDOWDerivative = 0.0
        weightYearDerivative = 0.0
        weightMonthDerivative = 0.0
        weightWeekDerivative = 0.0
        weightDayDerivative = 0.0
        weightDayofyearDerivative = 0.0
        #biasDerivative = 0.0
        
        for i in range(len(df)):
            if(math.isnan(df['Close'][i]) == False):
                weightDOWDerivative += (-2 * df['Dayofweek'][i] * (df['Close'][i] - self.weightDOWDerivative * df['Dayofweek'][i] - self.bias))
                weightYearDerivative += (-2 * df['Year'][i] * (df['Close'][i] - self.weightYearDerivative * df['Year'][i] - self.bias))
                weightMonthDerivative += (-2 * df['Month'][i] * (df['Close'][i] - self.weightMonthDerivative * df['Month'][i] - self.bias))
                weightWeekDerivative+= (-2 * df['Week'][i] * (df['Close'][i] - self.weightWeekDerivative * df['Week'][i] - self.bias))
                weightDayDerivative += (-2 * df['Day'][i] * (df['Close'][i] - self.weightDayDerivative * df['Day'][i] - self.bias))
                weightDayofyearDerivative += (-2 * df['Dayofyear'][i] * (df['Close'][i] - self.weightDayofyearDerivative * df['Dayofyear'][i] - self.bias))
                #biasDerivative += -2 * (df['Close'][i] - self.weight * df['Dayofweek'][i] - self.bias)
                   
        self.weightDOWDerivative -= (weightDOWDerivative / len(df)) * self.lr
        self.weightYearDerivative -= (weightYearDerivative / len(df)) * self.lr
        self.weightMonthDerivative -= (weightMonthDerivative / len(df)) * self.lr
        self.weightWeekDerivative -= (weightWeekDerivative / len(df)) * self.lr
        self.weightDayDerivative -= (weightDayDerivative / len(df)) * self.lr
        self.weightDayofyearDerivative -= (weightDayofyearDerivative / len(df)) * self.lr
        #self.bias -= (biasDerivative / len(df)) * self.lr
        
    def normalize(self):
        for i in df:
            df[i] -= np.min(df[i])
            minMax = np.max(df[i]) - np.min(df[i])
            if(minMax != 0):
                df[i] /= minMax
    
    def calculateCost(self):
        cost1,cost2,cost3,cost4,cost5,cost6 = 0.0,0.0,0.0,0.0,0.0,0.0
        for i in range(len(df)):
            if(math.isnan(df['Close'][i]) == False):
                cost1 += (df['Close'][i] - (self.weightDOWDerivative * df['Dayofweek'][i] + self.bias)) * (df['Close'][i] - (self.weightDOWDerivative * df['Dayofweek'][i] + self.bias))
                cost2 += (df['Close'][i] - (self.weightYearDerivative * df['Year'][i] + self.bias)) * (df['Close'][i] - (self.weightYearDerivative * df['Year'][i] + self.bias))
                cost3 += (df['Close'][i] - (self.weightMonthDerivative * df['Month'][i] + self.bias)) * (df['Close'][i] - (self.weightMonthDerivative * df['Month'][i] + self.bias))
                cost4 += (df['Close'][i] - (self.weightWeekDerivative * df['Week'][i] + self.bias)) * (df['Close'][i] - (self.weightWeekDerivative * df['Week'][i] + self.bias))
                cost5 += (df['Close'][i] - (self.weightDayDerivative * df['Day'][i] + self.bias)) * (df['Close'][i] - (self.weightDayDerivative * df['Day'][i] + self.bias))
                cost6 += (df['Close'][i] - (self.weightDayofyearDerivative * df['Dayofyear'][i] + self.bias)) * (df['Close'][i] - (self.weightDayofyearDerivative * df['Dayofyear'][i] + self.bias))
            
        return (cost1 / len(df), cost2 / len(df), cost3 / len(df), cost4 / len(df), cost5 / len(df), cost6 / len(df))
    
    def train(self):
        for i in range(self.n):
            self.updatedWeightAndBias()
            cost = self.calculateCost()
            #self.costList.append(np.sqrt(cost))
            print("DOW={:.2f} y={:.2f} m={:.2f} w={:.2f} day={:.2f} DOY={:.2f} bias={:.2f} cost={}"
                  .format(self.weightDOWDerivative, self.weightYearDerivative, self.weightMonthDerivative, self.weightWeekDerivative,
                          self.weightDayDerivative, self.weightDayofyearDerivative, self.bias, np.sqrt(cost)))
        
    def predict(self):
        predlist = list()
        for i in range(trainDataLen, len(actualVals)):
            predval = self.weightDOWDerivative * df2['Dayofweek'][i] + \
                        self.weightYearDerivative * df2['Year'][i] +  \
                        self.weightMonthDerivative * df2['Month'][i] +  \
                        self.weightWeekDerivative * df2['Week'][i] +  \
                        self.weightDayDerivative * df2['Day'][i] + \
                        self.weightDayofyearDerivative * df2['Dayofyear'][i] +\
                        self.bias
            predlist.append(predval)
        predlist -= np.min(predlist)
        minMax = np.max(predlist) - np.min(predlist)
        predlist /= minMax
        for i in range(0, trainDataLen):
            self.predictVals.append(0)
        for i in range(0, len(predlist)):
            self.predictVals.append(predlist[i])
            
    def plotGraph(self):
        #plt.plot(self.costList)
        plt.show()
        plt.plot(actualVals)
        plt.plot(self.predictVals)
        plt.show()

trainDataLen = 1000
ravg = Linear(0.9, 10, trainDataLen)
ravg.normalize()
actualVals = df['Close'].tolist()
df2 = DataFrame(df)
df = df[:trainDataLen]
ravg.train()
ravg.predict()
ravg.plotGraph()