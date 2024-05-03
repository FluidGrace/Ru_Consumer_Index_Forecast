# Прогнозирование Индексов Потребительских Цен методами машинного обучения
Проект состоит из нескольких частей:
1) Загрузка/первичная обработка данных.
Скрипт download.py автоматически ищет на сайте РосКомНадзора информацию о месячных индексах потребительских цен и скачивает.
Далее скрипт xltocsv.py преобразует .xlsx в .csv, попутно убирая лишние строки/столбцы.
Оба этих базовых скрипта встроены в веб-приложение.
2) Краткий разведочный анализ данных. С помощью инструментов statsmodels изучаем наши данные, чтобы догадываться, какие модели стоит тренировать. Основные пункты сохранены в analyze.py
3) Обучение и анализ SARIMA моделей.
4) Обучение и анализ Gradient Boosting Decision Trees.
5) Создание веб-приложения (app.py), выводящего график прогнозов моделей на 6 месяцев вперёд. Бонусом идёт 95%-доверительный интервал поверх графика.

## Установка
### Вариант 1. 
Устанавливаем библиотеки из файла requirements.txt с помощью pip.

```bash
pip install scikit-learn==0.24.1
pip install statsmodels==0.13.1
pip install pandas==1.3.5
pip install numpy==1.21.5
pip install flask==2.0.3
pip install openpyxl==3.1.2
pip install matplotlib==3.8.4
pip install Werkzeug==2.2.2
```
app.py - Flask-приложение с веб-интерфейсом
download.py - скрипт-загрузчик данных
xltocsv.py - .xlsx -> .csv
analyze.py - графики/тесты на исходных данных
arima.py - модуль тренировки SARIMA моделей
sarima_analyze.py - разбор избранной SARIMA модели (доверительные интервалы, plot_diagnostics)

### Вариант 2. 
Позволяему Docker'у установить их.

## Приложение
Начать ознакомление предпочтительно с приложения. Реализована возможность мимолётным движением руки получить график прогноза на 6 месяцев вперёд.
На выбор - классическая модель SARIMA и модель градиентного спуска.
Модель GBDT получилась намного точнее в плане MSE, однако у SARIMA более предсказуемое поведение с небольшим доверительным интервалом.
![webapp](https://github.com/FluidGrace/Ru_Consumer_Index_Forecast/assets/168632884/8bce9666-72bb-417b-ab73-eaa14a146691)

## Предварительный анализ данных
С самого начала было решено оставить последние 40 точек данных под тест.
Имеем дело с временным рядом, поэтому первым делом была проверена "наивная" модель y_pred[t]=y[t-1], - в каждый месяц ожидать, что будет как в прошлом. У неё около 1.439 MSE на тесте.
```bash
prediction=data['Index'].shift(1)
```
Изначальный массив данных вёл число от 1991 года. Сразу возник вопрос оставлять ли данные 1990-х. Безусловно, 400 точек обучения, лучше чем 300, но данные того периода с инфляцией по 2600% в год вносили огромный шум в общую картину.
Доверительные интервалы даже ARIMA моделей составляют по 10-20% в каждую сторону, если оставить как есть. Решено отбросить первые 100 месяцев было, когда опытным путём обнаружилось, что модели с вырезанными 90ми имеют ощутимое превосходство в MSE. Даже лучшие модели с 90ми годами имели не менее 1.1 MSE.

Итак, отбросив первые 100 месяцев и запустив 
```bash
plot_acf(train['Index'])
```
Видим периодичность данных в 12 месяцев невооруженным взглядом.
![acf](https://github.com/FluidGrace/Ru_Consumer_Index_Forecast/assets/168632884/507e3f68-b768-46c3-9381-4c54a9fcaa1a)

График же pacf намекает, что при переборе параметров SARIMA значения p от 2 до 5 можно оставить на потом:
![pacf](https://github.com/FluidGrace/Ru_Consumer_Index_Forecast/assets/168632884/480cea20-0317-425e-9a83-92debe1aec27)

Ну и напоследок.
'''
print(adfuller(train['Index'])) 
print(kpss(train['Index']))
'''

Нулевую гипотезу ADF отвергаем, а KPSS нет.

## SARIMA
После предварительного анализа решено было строить SARIMA с периодом 12.
Так как имеем дело с временным рядом, то валидацию пришлось делать сдвигом - постепенно увеличивая тренировочную часть, тренируя на первых 40%, потом 50%, 60%, 70% и 80%. В итоге из всех параметров выбрана была модель с (p,d,q)(p_s,d_s,q_s) = (1,1,1)(1,0,1)
Её MSE оказался не впечатляющим - около 1.33 на тесте. Зато почти все точки попали в доверительный интервал (40 точек):
![sarima](https://github.com/FluidGrace/Ru_Consumer_Index_Forecast/assets/168632884/8fbda021-e001-4f0a-97ed-31bfeb9b9a3f)

