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

df1 = pd.read_csv('C:/Users/dchandra/Downloads/Mine/Study/Projects/stock_prediction/tataglobal.csv')
df = DataFrame(df1)
df = df.sort_values('Date')
df.index = df['Date']
df = add_datepart(df, 'Date')
df = df[['Dayofweek', 'Close']]
actualVals = df['Close'].tolist()
actualDayOfWeekVals = df['Dayofweek']


class LinearFromSklearn(object):
    def __init__(self, new_data):
        self.new_data = new_data
        
    def calculateFromSklearn(self):
        self.new_data = add_datepart(self.new_data, 'Date')
        self.new_data.drop('Elapsed', axis=1, inplace=True)
        train = self.new_data[:987]
        valid = DataFrame(self.new_data[987:])
        
        x_train = train.drop('Close', axis=1)
        y_train = train['Close']
        x_valid = valid.drop('Close', axis=1)
        y_valid = valid['Close']
        
        #implement linear regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(x_train,y_train)
    
        #make predictions and find the rmse
        preds = model.predict(x_valid)
        rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
        print(rms)
        
        #plot
        valid['Predictions'] = 0
        valid['Predictions'] = preds
        
        valid.index = self.new_data[987:].index
        train.index = self.new_data[:987].index
        
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])


ravg = LinearFromSklearn(df1)
ravg.calculateFromSklearn()