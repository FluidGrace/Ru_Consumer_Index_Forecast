import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error as MSE
from statsmodels.graphics.tsaplots import plot_predict
import matplotlib.pyplot as plt

data = pd.read_csv('month-aggregate.csv')
data = data.unstack()
data = data[:data.last_valid_index()]
month = [x[1] for x in data.index]
y = data.values
data = pd.DataFrame(y, columns=['Index'])
data['Month']=month
data = data[100:]
y = y[100:]
start = int(0.9*len(data)) - 10
end = len(data)
train = y[:start]
test = y[start:]

# глянем поближе на нашу лучшую SARIMA модель
model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,0,1,12), enforce_stationarity=0).fit(disp=0)

# распределена очень нормально
model.plot_diagnostics(figsize=(15, 12))
plt.show()

forecast = model.get_forecast(steps=1)
pred = list(forecast.predicted_mean)
pred_low = [forecast.conf_int()[0][0]]
pred_high = [forecast.conf_int()[0][1]]
for i in range(end-start-1):
    model = model.append([test[i]], refit=1, fit_kwargs={'disp':0})
    forecast = model.get_forecast(steps=1)
    pred += list(forecast.predicted_mean)
    pred_low.append(forecast.conf_int()[0][0])
    pred_high.append(forecast.conf_int()[0][1])
print(pred)
plt.plot(test)
plt.plot(pred, color='red')
x = list(range(len(pred)))
plt.fill_between(x, pred_low, pred_high, color='k', alpha=.25)
plt.show()
# из 40 точек только две вышли за дов. интервал 95%, как и полагается

print(f"Test Error = {MSE(pred, test, squared=False)}")
print(f'AIC = {model.aic}')
# модель не самая точная, надо пробовать GBDT
