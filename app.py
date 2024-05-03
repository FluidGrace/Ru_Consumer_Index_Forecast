from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import download
import xltocsv
import pickle

app = Flask(__name__)
months = ['Янв','Фев','Мар','Апр','Май','Июн','Июл','Авг','Сен','Окт','Ноя','Дек']

def download():
    download # скрипт скачивания
    xltocsv # xlsx -> csv
    data = pd.read_csv('month-aggregate.csv')
    data = data.unstack()
    data = data[:data.last_valid_index()]
    data = pd.DataFrame(data.values, columns=['Index'])
    for i in [1,2,3,4,6,7,11,12]:
        data[f'Index_Lag_{i}'] = data['Index'].shift(i)
    for i in [2,3,4,6]:
        data[f'SMA{i}-12'] = data['Index'].shift(i).rolling(13-i).mean()
    year = 1991 + int(len(data)/12)
    month = len(data)%12
    x = [months[month]]
    for i in range(6):
        x.append(months[(month+i+1)%12])
    for i in range(7):
        x[i]+=' '+str(year+int((month+i)/12))
    return data, x

def arima():
    plt.close()
    data, x = download()
    with open('sarima-(1,1,1)-(1,0,1,12).pkl','rb') as z:
        sarima = pickle.load(z)
    sarima = sarima.apply(data['Index'], refit=1, fit_kwargs={'disp':0}) # запускаем модель на свежих данных
    predictions_sarima = sarima.get_forecast(steps=6)
    y_sarima, y_sarima_low, y_sarima_high = data['Index'].values[-1:], data['Index'].values[-1:], data['Index'].values[-1:]
    y_sarima = np.append(y_sarima, predictions_sarima.predicted_mean)
    y_sarima_low = np.append(y_sarima_low, predictions_sarima.conf_int().iloc[:,0])
    y_sarima_high = np.append(y_sarima_high, predictions_sarima.conf_int().iloc[:,1])
    plt.plot(x, y_sarima, label='SARIMA-(1,1,1)-(1,0,1,12)')
    plt.legend()
    plt.fill_between(x, y_sarima_low, y_sarima_high, color='k', alpha=.25)
    plt.ylabel('Прогнозируемое значение')
    plt.title('Прогноз ИПЦ России')
    plt.savefig('static/arima.png') # строим график прогноза

def gbdt(): 
    plt.close()   
    data, x = download()
    data_low = data.copy()
    data_high = data.copy()
    with open('gboost-lag1_6_12-sma12.pkl','rb') as z:
        gboost = pickle.load(z)
    with open('gboost-lag2_7_12-sma12.pkl','rb') as z:
        gboost2 = pickle.load(z)
    with open('gboost-lag3_11_12-sma12.pkl','rb') as z:
        gboost3 = pickle.load(z)
    with open('gboost-lag4_12-sma12.pkl','rb') as z:
        gboost4 = pickle.load(z)
    with open('gboost-lag12-sma12.pkl','rb') as z:
        gboost5 = pickle.load(z)
    X_train = data[['Index_Lag_1','Index_Lag_6','Index_Lag_12','SMA2-12']][12:]
    X_train_2 = data[['Index_Lag_2','Index_Lag_7','Index_Lag_12','SMA2-12']][12:]
    X_train_3 = data[['Index_Lag_3','Index_Lag_11','Index_Lag_12','SMA3-12']][12:]
    X_train_4 = data[['Index_Lag_4','Index_Lag_12','SMA4-12']][12:]
    X_train_5 = data[['Index_Lag_12','SMA6-12']][12:]
    y_train = data['Index'][12:]
    residuals = [[] for i in range(6)]
    residuals[0] = y_train - gboost.predict(X_train)
    residuals[1] = y_train - gboost2.predict(X_train_2)
    residuals[2] = y_train - gboost3.predict(X_train_3)
    residuals[3] = y_train - gboost4.predict(X_train_4)
    residuals[4] = residuals[5] = y_train - gboost5.predict(X_train_5)
    bootstrapped_residuals = np.zeros((6, 10000))
    for i in range(6):
        bootstrapped_residuals[i] = np.random.choice(residuals[i], 10000)
    bootstrapped_predictions = np.zeros((6, 10000))
    y_gboost = data['Index'].values[-1:]
    y_gboost_low = data['Index'].values[-1:]
    y_gboost_high = data['Index'].values[-1:]
    prediction = np.zeros(6)
    prediction[0] = gboost.predict(data[['Index_Lag_1','Index_Lag_6','Index_Lag_12','SMA2-12']][-1:])
    prediction[1] = gboost2.predict(data[['Index_Lag_2','Index_Lag_7','Index_Lag_12','SMA2-12']][-1:])
    prediction[2] = gboost3.predict(data[['Index_Lag_3','Index_Lag_11','Index_Lag_12','SMA3-12']][-1:])
    prediction[3] = gboost4.predict(data[['Index_Lag_4','Index_Lag_12','SMA4-12']][-1:])
    prediction[4] = gboost5.predict(data[['Index_Lag_12','SMA6-12']][-1:])
    prediction[5] = gboost5.predict(data[['Index_Lag_12','SMA6-12']][-1:])
    for i in range(6):
        for j in range(10000):
            bootstrapped_predictions[i][j] = prediction[i] + bootstrapped_residuals[i][j]
    for i in range(5):
        for j in range(6-i,6):
            bootstrapped_residuals[j] = np.random.choice(residuals[j], 10000)
            for k in range(10000):
                bootstrapped_predictions[j][k] += bootstrapped_residuals[j][k]
    for i in range(6):
        low_bound = np.percentile(bootstrapped_predictions[i], 2.5)
        high_bound = np.percentile(bootstrapped_predictions[i], 97.5)
        y_gboost = np.append(y_gboost, prediction[i])
        y_gboost_low = np.append(y_gboost_low, low_bound)
        y_gboost_high = np.append(y_gboost_high, high_bound)
    plt.plot(x, y_gboost, label='GradBoostDT')
    plt.legend()
    plt.fill_between(x, y_gboost_low, y_gboost_high, color='k', alpha=.25)
    plt.ylabel('Прогнозируемое значение')
    plt.title('Прогноз ИПЦ России')
    plt.savefig('static/gbdt.png') # строим график прогноза

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'button1' in request.form:
            arima()
            return render_template('index.html', graph1=True)
        elif 'button2' in request.form:
            gbdt()
            return render_template('index.html', graph2=True)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
