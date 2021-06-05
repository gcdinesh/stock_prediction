import pandas as pd
import numpy as np
import mplcursors
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from pandas import DataFrame
from fastai.tabular.all import add_datepart
#%matplotlib inline
# use %matplotlib qt instead of inline to visualize and hover the graph, but it is too slow
#instead of using this command here go to tools-> preferences -> ipython console -> graphics -> backend(dropdown) to inline

rcParams['figure.figsize'] = 20,10

df = pd.read_csv('C:/Users/dchandra/Downloads/Mine/Study/Projects/stock_prediction/ITC.BO.csv')
df = df.sort_values('Date')
        
class LinearGeneral(object):
    def __init__(self, new_data):
        self.new_data = new_data
        
    def normalize(self):
        for i in self.new_data:
            self.new_data[i] -= np.min(self.new_data[i])
            minMax = np.max(self.new_data[i]) - np.min(self.new_data[i])
            if(minMax != 0):
                self.new_data[i] /= minMax
        
    def calculate(self):
        length = 5000
        self.new_data = add_datepart(self.new_data, 'Date')
        self.new_data.drop(['Open', 'High', 'Low', 'Elapsed', 'Adj Close', 'Volume', 'Is_month_end',
        'Is_month_start',              
        'Is_quarter_end',
        'Is_quarter_start',
        'Is_year_end',
        'Is_year_start'], axis=1, inplace=True)
        #self.normalize()
        train = self.new_data[:length]
        valid = DataFrame(self.new_data[length:])
        
        x_train = train.drop('Close', axis=1)
        y_train = train['Close']
        x_valid = valid.drop('Close', axis=1)
        y_valid = DataFrame(valid['Close'])
    
        
        #implement linear regression
        weight = np.zeros(len(df.T))
        bias = np.zeros(len(df.T))
        c = 0
        for f in x_train:
            x_data = x_train[f]
            x_mean = np.mean(x_data)
            x_data = x_data - x_mean
            y_data = y_train
            y_mean = np.mean(y_data)
            y_data = y_data - y_mean
            num = 0.0
            den = 0.0
            for i in range(len(x_data)):
                num += (x_data[i]) * (y_data[i])
                den += (x_data[i])**2
                weight[c] = num / den
                bias[c] = y_mean - (weight[c] * x_mean)
            c += 1
            
        print("weight={}".format(weight))
        print("bias={}".format(bias))
    
        #make predictions and find the rmse
        predlist = list()
        for i in range(length, length + len(x_valid)):
            c = 0
            predval = 0.0
            for f in x_valid:
                x_valid[f] = x_valid[f] - np.mean(x_valid[f])
                predval += weight[c] * x_valid[f][i]
                c += 1
            predlist.append(predval)
                
        #predlist -= np.min(predlist)
        #minMax = np.max(predlist) - np.min(predlist)
        #if(minMax != 0):
        #    predlist /= minMax
        predictVals = list()
        for i in range(0, length):
            predictVals.append(0)
        for i in range(0, len(predlist)):
            predictVals.append(predlist[i])
            
        y_valid -= y_mean
        rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(predlist)),2)))
        print(rms)
        
        #plot
        plt.plot(self.new_data['Close'])
        plt.plot(predictVals)
        plt.show()

ravg = LinearGeneral(df)
ravg.calculate()