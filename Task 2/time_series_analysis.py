import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from itertools import product

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
#from statsmodels.iolib.table import SimpleTable

#import pylab
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

def ADF(series):
    dftest = adfuller(series.dropna())
    print('adf: {:.3f}'. format(dftest[0]))
    print('p-value: {:.3f}'.format(dftest[1]))
    print('critical values: ', dftest[4])
    if dftest[0] > dftest[4]['5%']: 
        print('есть единичные корни, ряд не стационарен')
    else:
        print('единичных корней нет, ряд стационарен')

def additive(data):
    result = sm.tsa.seasonal_decompose(data.Value, model='additive')
    result.plot()
    plt.show()

    from random import randrange
    print("\nCheck the components:")
    print('Trend:')
    ADF(result.trend)

    print('')
    print('Seasonal:')
    ADF(result.seasonal)

    print('')
    print('Resid:')
    ADF(result.resid)

def multiplicative(data):
    result = sm.tsa.seasonal_decompose(data.Value, model='multiplicative')
    result.plot()
    plt.show()

    from random import randrange
    print("\nCheck the components:")
    print('Trend:')
    ADF(result.trend)

    print('')
    print('Seasonal:')
    ADF(result.seasonal)

    print('')
    print('Resid:')
    ADF(result.resid)

data_training = pd.read_excel('~/prac/training.xlsx', index_col='Date', parse_dates=True)
data_testing = pd.read_excel('~/prac/testing.xlsx', index_col='Date', parse_dates=True)
sns.set()

ADF(data_training)
data_training.Value.plot(label='Training series')
plt.legend(loc='upper left')
plt.show()

additive(data_training)
multiplicative(data_training)

data_training_diff = data_training.Value.diff(periods=1).dropna()
ADF(data_training_diff)
data_training_diff.plot(label='Training series diff', figsize=(18,5))
plt.legend(loc='upper left')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data_training_diff.squeeze(), lags=50, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data_training_diff.squeeze(), lags=50, ax=ax2)

model_0 = sm.tsa.ARIMA(data_training, order=(0,1,0), freq='MS').fit()
model_1 = sm.tsa.ARIMA(data_training, order=(1,1,1), freq='MS').fit()
model_2 = sm.tsa.ARIMA(data_training, order=(12,1,4), freq='MS').fit()

pred_0 = model_0.predict('1989-01-01','1993-12-01', typ='levels')
pred_1 = model_1.predict('1989-01-01','1993-12-01', typ='levels')
pred_2 = model_2.predict('1989-01-01','1993-12-01', typ='levels')

data_testing.Value.plot(label='Testing series')
pred_0.plot(color='red', label='mod0')
pred_1.plot(color='green', label='mod1')
pred_2.plot(color='yellow', label='mod2')
plt.legend(loc='upper left')
plt.show()

print('AIC model0: %1.3f' % model_0.aic)
print('AIC model1: %1.3f' % model_1.aic)
print('AIC model2: %1.3f' % model_2.aic)

r2_0 = r2_score(data_testing, pred_0)
r2_1 = r2_score(data_testing, pred_1)
r2_2 = r2_score(data_testing, pred_2)
print('r2_0 R^2: %1.3f' % r2_0)
print('r2_1 R^2: %1.3f' % r2_1)
print('r2_2 R^2: %1.3f' % r2_2)

ps = range(0, 2)
d=1
qs = range(0, 4)

parameters = product(ps, qs)
parameters_list = list(parameters)

results = []
best_aic = float("inf")

for param in parameters_list:
    #try except нужен, потому что на некоторых наборах параметров модель не обучается
    try:
        model=sm.tsa.ARIMA(data_training.Value, order=(param[0], d, param[1]),  freq='MS').fit()
    #выводим параметры, на которых модель не обучается и переходим к следующему набору
    except ValueError:
        print('wrong parameters:', param)
        continue
    aic = model.aic
    #сохраняем лучшую модель, aic, параметры
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])

result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True))

print(best_model.summary())

data_training.Value.plot(label='Training series')
data_testing.Value.plot(label='Testing series')

pred = best_model.predict('1989-01-01','1993-12-01', typ='levels')
pred.plot(label='Time Series Prediction', style='r--')

plt.legend(loc='upper left')
plt.show()

r2 = r2_score(data_testing.Value, pred)
print('R^2: %1.3f' % r2)
