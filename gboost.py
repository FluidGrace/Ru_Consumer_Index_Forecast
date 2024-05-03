import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('month-aggregate.csv')
data=data.unstack()
data=data[:data.last_valid_index()]
y=data.values
data = pd.DataFrame(y, columns=['Index'])
for i in range(1,14):
    data[f'Index_Lag_{i}'] = data['Index'].shift(i)
for i in range(2,7):
    data[f'SMA{i}-12'] = data['Index'].shift(i).rolling(13-i).mean()
data=data[100:]
y=y[100:]

minmean = 10
bestset = []
best = [128,1,1]
#опять же, кросс-валидация через сдвиг
for sets in [['Index_Lag_1','Index_Lag_6','Index_Lag_12','SMA2-12']]:
    print(f'Features added = {sets}')
    for m in (64,128,256):
        for depth in (2,3,4,6,8,16):
            for leaf in (1,2,3,4,6,8,12,16,24,32):
                    errors = []
                    scores = []
                    for split in [0.4, 0.5, 0.6, 0.7, 0.8]:
                        start = int(split * (len(data)+100)) - 100
                        end = int((split+0.1) * (len(data)+100)) - 100
                        X_train = data[[] + sets][12:start]
                        X_test = data[[] + sets][start:end]
                        y_train = y[12:start]
                        y_test = y[start:end]
                        model = GradientBoostingRegressor(n_estimators=m, max_depth=depth,
                           min_samples_leaf = leaf, random_state=0).fit(X_train, y_train)
                        pred = model.predict(X_test)
                        error = MSE(pred, y_test, squared=False)
                        errors.append(error)
                        scores.append(model.score(X_test, y_test))
                    if np.mean(errors) < minmean:
                        minmean = np.mean(errors)
                        bestset = sets
                        best = [m, depth, leaf]
                        print(f'Number of trees = {m}')
                        print(f'Depth = {depth}')
                        print(f'MinLeaf = {leaf}')
                        print(f'Mean MSE = {np.mean(errors)}')
                        print(f'Mean R2-scores = {np.mean(scores)}')
                        print([]+sets, model.feature_importances_)

start = int(0.9*len(data)) - 10
X_train = data[['Index_Lag_1', 'Index_Lag_6', 'Index_Lag_12', 'SMA2-12']][12:start]
X_test = data[['Index_Lag_1', 'Index_Lag_6', 'Index_Lag_12', 'SMA2-12']][start:]
y_train = y[12:start]
y_test = y[start:]

params = dict(n_estimators=best[0], max_depth=best[1],
              min_samples_leaf=best[2], random_state=0)
model = GradientBoostingRegressor(**params).fit(X_train, y_train)
print(MSE(model.predict(X_test), y_test, squared=False))
print(model.score(X_test, y_test))
#модель заметно лучше чем SARIMA в плане MSE, 1.04MSE уже заметно лучше базовых 1.44
#pickle.dump(model, open('gboost-lag1_6_12-sma12.pkl', 'wb'))

#напоследок график доверительного интервала
model_low = GradientBoostingRegressor(loss="quantile", alpha=0.025,
                                      **params).fit(X_train, y_train)
model_high = GradientBoostingRegressor(loss="quantile", alpha=0.975,
                                      **params).fit(X_train, y_train)
x = list(range(len(y_test)))
y_pred = model.predict(X_test)
y_low = model_low.predict(X_test)
y_high = model_high.predict(X_test)
pred_plot = plt.plot(x, y_test)
plt.plot(y_pred, color='red', label='GradBoostDT Forecast')
plt.fill_between(x, y_low, y_high, color='k', alpha=.25)
plt.legend()
plt.show()
#7 точек из 40 не влезли в дов. интервал, так что у SARIMA есть свои плюсы
