import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_predict
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
import warnings
import pickle
import math

warnings.filterwarnings("ignore")

#cutting NaN tail
data = pd.read_csv('month-aggregate.csv')
data = data.unstack()
data=data[:data.last_valid_index()]
month = [x[1] for x in data.index]
y = data.values
data = pd.DataFrame(y, columns=['Index'])
data = data[100:]
y = y[100:]

#для временных рядов годится только кросс-валидация со сдвигом, подберём с её помощью параметры
minmean = 10
best = (1,0,0)
best_season = (0,0,0,12)
bestaic = (0,0,0)
bestaic_s = (0,0,0,12)
minaic = 4000
models = []
models_s = []
models_aic = []
models_mse = []
modelweights = []
n=1
#после просмотра автокорреляций можно обойтись прогоном по этим параметрам
for d_s in [0,1]:
    for p_s in [0,1,2]:
        for q_s in [1,2,3]:
            for d in [0,1,2]:
                for p in [1,6,8,11]:
                    for q in [0,1,2,6]:
                        errors = []
                        for split in [0.4, 0.5, 0.6, 0.7, 0.8]:
                            start = int(split*(len(data)+100))-100
                            end = int((split+0.1)*(len(data)+100))-100
                            model = SARIMAX(y[:start], order=(p,d,q), 
                                seasonal_order=(p_s,d_s,q_s,12), enforce_stationarity=0).fit(disp=0)
                            pred = [model.forecast()]
                            for k in range(start,end-1):
                                if math.isnan(pred[-1]):
                                    if len(pred)>1:
                                        pred[-1] = pred[-2]
                                    else:
                                        pred[-1] = y[k-1:k]
                                try:
                                    model = model.append(y[k:k+1], refit=0, enforce_stationarity=0)
                                    pred.append(model.forecast(disp=0))
                                except KeyError:
                                    pred.append(pred[-1])
                            error = MSE(pred, y[start:end], squared=0)
                            errors.append(error)
                        models.append((p,d,q))
                        models_s.append((p_s,d_s,q_s,12))
                        models_aic.append(model.aic)
                        models_mse.append(np.mean(errors))
                        if np.mean(errors)<minmean:
                            minmean = np.mean(errors)
                            best = (p,d,q)
                            best_season = (p_s,d_s,q_s,12)
                            print(f'Mean MSE for {p,d,q}, {p_s,d_s,q_s,12} = {np.mean(errors)}')
                            print(f'StD MSE for {p,d,q}, {p_s,d_s,q_s,12} = {np.std(errors)}')
                        if model.aic<minaic:
                            bestaic = (p,d,q)
                            bestaic_s = (p_s,d_s,q_s,12)
                            minaic = model.aic
                            print(f'AIC for {p,d,q}, {p_s,d_s,q_s,12} = {model.aic}')
weightsum = 0
for i in range(len(models)):
    modelweight = 2.718**(-1000*(models_mse[i] - minmean))
    if modelweight>0.01:
        print(models[i],models_s[i],modelweight)
        modelweights.append(modelweight)
        weightsum+=modelweight
    else: 
        modelweights.append(0)
start = int(0.9*len(data))-10
end = len(data)

model = SARIMAX(y[:start], order=bestaic, enforce_stationarity=0,
                seasonal_order=bestaic_s).fit(disp=0)
pred = [model.forecast()]
if math.isnan(pred[-1]):
    pred[-1] = pred[-2]
for i in range(start, end-1):
    model = model.append(y[i:i+1], refit=1, fit_kwargs={'disp':0})
    pred.append(model.forecast())
print(f"Test Error = {MSE(pred, data['Index'][start:], squared=False)}")
print(f'AIC = {model.aic}')
#plt.plot(y[start:])
#plt.plot(pred, color='red')
#plt.show()
print(len(modelweights))
print(len(pred))

#лучшую на валидации модель дотренировываем и сохраняем
model = SARIMAX(y[:start], order=best, seasonal_order=best_season, enforce_stationarity=0).fit(disp=0)
pred = [model.forecast()]
for i in range(start, end-1):
    model = model.append(y[i:i+1], refit=1, fit_kwargs={'disp':0})
    pred.append(model.forecast())
print(f"Test Error = {MSE(pred, data['Index'][start:], squared=False)}")
print(f'AIC = {model.aic}')
plt.plot(y[start:])
plt.plot(pred, color='red')
plt.show()
pickle.dump(model, open(f'sarima-{best}-{best_sea}-1999.pkl', 'wb'))
#у нашей модели MSE=1.332, лучше чем 1.439 базового, но не намного, значит надо что-то получше sarima
#часто при короткой выборке для времянных рядов из машинных методов наиболее перспективны бусты

for i in range(len(pred)):
    pred[i]/=weightsum
for i in range(len(models)):
    if modelweights[i]>0.01 and modelweights[i]<1:
        model = SARIMAX(y[:start], order=models[i],
                        seasonal_order=models_s[i]).fit(disp=0)
        pred[0]+=model.forecast()*modelweights[i]/weightsum
        for j in range(start, end-1):
            model = model.append(y[j:j+1], refit=1, fit_kwargs={'disp':0})
            pred[j-start+1] += model.forecast(disp=0)*modelweights[i]/weightsum
plt.plot(y[start:])
plt.plot(pred, color='red')
plt.show()
print(f"Test Error = {MSE(pred, data['Index'][start:], squared=False)}")
