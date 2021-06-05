import pandas as pd
import numpy as np
import mplcursors
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from pandas import DataFrame
from fastai.tabular.all import add_datepart
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
#%matplotlib inline
# use %matplotlib qt instead of inline to visualize and hover the graph, but it is too slow
#instead of using this command here go to tools-> preferences -> ipython console -> graphics -> backend(dropdown) to inline

rcParams['figure.figsize'] = 20,10

df = pd.read_csv('C:/Users/dchandra/Downloads/Mine/Study/Projects/stock_prediction/ADANITRANS.BO.csv')
df = df.sort_values('Date')
        
class KNN(object):
    def __init__(self, new_data):
        self.new_data = new_data
        
    def normalize(self, data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        return data
        
    def calculate(self):
        length = 1000
        self.new_data = add_datepart(self.new_data, 'Date')
        self.new_data.drop(['Open', 'High', 'Low', 'Elapsed', 'Adj Close', 'Volume', 'Is_month_end',
        'Is_month_start',              
        'Is_quarter_end',
        'Is_quarter_start',
        'Is_year_end',
        'Is_year_start'], axis=1, inplace=True)
        
        train = self.new_data[:length]
        valid = DataFrame(self.new_data[length:])
        
        x_train = self.normalize(train.drop('Close', axis=1))
        y_train = train['Close']
        x_valid = self.normalize(valid.drop('Close', axis=1))
        y_valid = valid['Close']
    
        
        #implement KNN
        params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
        knn = neighbors.KNeighborsRegressor()
        model = GridSearchCV(knn, params, cv=5)
        model.fit(x_train, y_train)
        
        #make predictions and find the rmse
        preds = model.predict(x_valid)
        rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
        print(rms)
        
        #plot
        valid['Predictions'] = 0
        valid['Predictions'] = preds
        
        valid.index = self.new_data[length:].index
        train.index = self.new_data[:length].index
        
        plt.plot(train['Close'].append(valid['Close']))
        plt.plot(valid[['Predictions']])

ravg = KNN(df)
ravg.calculate()