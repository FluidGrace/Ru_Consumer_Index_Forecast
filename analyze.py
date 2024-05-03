import pandas as pd
import numpy as np
#pip install matplotlib
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn.metrics import mean_squared_error as MSE

#убираем пустые ячейки, убираем тестовую часть
data=pd.read_csv('month-aggregate.csv')
data=data.unstack()
data=data[:data.last_valid_index()]
data = pd.DataFrame(data.values, columns=['Index'])
for i in range(1,14):
    data[f'Index_Lag_{i}']=data['Index'].shift(i)
data['SMA2-12']=data['Index'].shift(2).rolling(11).mean()
bound=int(0.9*len(data))
train=data[100:bound]
test=data['Index'][bound:].values
pred=data['Index'][bound-1:-1].values
# MSE для наивного предсказания около 1.439/1.647/1.745/1.764 для 1/2/3/4 шагов
print(f"MSE для наивного предсказания = {MSE(pred, test, squared=False)}")

data['Index'].plot()
plt.show()
print(train.corr())
#смотрим автокорреляцию, проверяем на стационарность
plot_acf(train['Index']) # на ACF графе видно сезонность 12 месяцев
plot_pacf(train['Index']) # PACF граф не спешит затухать
plt.show()
print(adfuller(train['Index'])) 
print(kpss(train['Index']))  # похоже, что стационарности нет
#в общем стоит начать с SARIMA с сезоном 12
