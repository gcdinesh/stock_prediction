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

class LinearUsingSVD(object):
    def __init__(self, new_data):
        self.new_data = new_data
        
    def calculate(self):
        length = 5000
        self.new_data = add_datepart(self.new_data, 'Date')
        self.new_data.drop(['Open', 'High', 'Low', 'Elapsed', 'Adj Close', 'Volume', 'Is_month_end',
        'Is_month_start',              
        'Is_quarter_end',
        'Is_quarter_start',
        'Is_year_end',
        'Is_year_start'], axis=1, inplace=True)
        
        close_mean = np.mean(self.new_data['Close'])
        
        for f in self.new_data:
            self.new_data[f] = self.new_data[f] - np.mean(self.new_data[f])
        
        train = self.new_data[:length]
        valid = DataFrame(self.new_data[length:])
        
        x_train = train.drop('Close', axis=1)
        y_train = train['Close']
        x_valid = valid.drop('Close', axis=1)
        y_valid = valid['Close']
        
        #implement linear regression
        # y = wx => w = (x^-1)(y)
        # (x^-1) = VD(Ut) --> we need V, D = 1/S, U
        # we know that X = US(Vt) => this we can get it from SVD of numpy
        U, S, Vt = np.linalg.svd(x_train, full_matrices = False)
        Ut = U.T
        V = Vt.T
        #The number of nonzero singular values of A equals the rank of A.
        #The rank of any square matrix equals the number of nonzero eigenvalues (with repetitions)
        n = x_train.shape[1]
        r = np.linalg.matrix_rank(x_train)
        D = np.diag(np.hstack([1/S[:r], np.zeros(n-r)]))
        X_INV = V.dot(D).dot(Ut)
        weight = X_INV.dot(y_train)
        
        #make predictions and find the rmse
        predlist = list()
        for i in range(length, length + len(x_valid)):
            c = 0
            predval = 0.0
            for f in x_valid:
                predval += weight[c] * x_valid[f][i] + 16 #this bias was adjusted need to check how actually the data was preprocessed in sklearn
                c += 1
            predlist.append(predval)
        
        predictVals = list()
        for i in range(0, length):
            predictVals.append(self.new_data['Close'][i] + close_mean)
        for i in range(0, len(predlist)):
            predictVals.append(predlist[i])
            
        rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(predlist)),2)))
        print(rms)
        
        #plot
        plt.plot(self.new_data['Close'] + close_mean)
        plt.plot(predictVals)
        plt.show()
        

ravg = LinearUsingSVD(df)
ravg.calculate()