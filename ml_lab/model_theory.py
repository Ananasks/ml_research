"""Model explanations, conceptual diagrams and inspected fitted tree structures."""
import inspect
import json
from importlib.metadata import version
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from sklearn.tree import export_graphviz
from ml_lab.online_docs import render_online_document
from ml_lab.model_registry import MODEL_REGISTRY,MODELS,documentation,online_catalog_status

GUIDES={
 'Линейные модели':'linear_model.html','Деревья и ансамбли':'ensemble.html','Бустинг':'ensemble.html#gradient-boosting',
 'Нейронные сети':'neural_networks_supervised.html','Методы опорных векторов':'svm.html','Соседние объекты':'neighbors.html',
 'Наивный Байес':'naive_bayes.html','Дискриминантный анализ':'lda_qda.html','Ядерные методы':'kernel_ridge.html',
 'Гауссовские процессы':'gaussian_process.html','Базовые модели':'model_evaluation.html#dummy-estimators'}
PROFILES={
 'Линейные модели':('Линейная комбинация признаков. Обучение подбирает коэффициенты, регуляризация ограничивает сложность. Нелинейные зависимости требуют дополнительных признаков.',r'\hat y=b+w^Tx', ['Признаки X','Взвешенная сумма: wᵀX + b','Значение / решение класса']),
 'Деревья и ансамбли':('Дерево последовательно проверяет пороги признаков и возвращает ответ листа. Ансамбль объединяет ответы нескольких деревьев, уменьшая нестабильность отдельных решений.',r'\hat y(x)=\sum_{l=1}^{L}c_l\,\mathbf{1}(x\in R_l)', ['Признаки X','Проверки порогов','Листья деревьев','Ответ / объединение ответов']),
 'Бустинг':('Деревья строятся последовательно: очередная модель улучшает текущую композицию. Learning rate ограничивает вклад нового дерева. Слишком сложная композиция может переобучаться.',r'F_m(x)=F_{m-1}(x)+\eta h_m(x)', ['Признаки X','Начальное приближение F₀','Дерево h₁: коррекция','Следующие коррекции h₂…hₘ','Сумма → выход']),
 'Нейронные сети':('Нейронная сеть сочетает обучаемые линейные преобразования и нелинейные активации. Скрытые слои строят представления данных; веса подбираются градиентными методами.',r'h^{(l)}=\phi(W^{(l)}h^{(l-1)}+b^{(l)})', ['Входные признаки','Скрытые слои','Выходной слой']),
 'Методы опорных векторов':('SVM строит разделяющую поверхность с зазором; SVR допускает область малых ошибок вокруг регрессионной функции. Ядро задаёт форму зависимости, C — компромисс между регуляризацией и ошибками.',r'f(x)=b+\sum_i\alpha_i y_i K(x_i,x)', ['Признаки X','Ядро / линейное пространство','Опорные объекты','Решение / значение']),
 'Соседние объекты':('Ответ строится по близким объектам обучающей выборки. Масштаб признаков определяет расстояния; слишком малое число соседей делает прогноз чувствительным к шуму.',r'\hat y(x)=\frac{\sum_{i\in N_k(x)}w_i y_i}{\sum_{i\in N_k(x)}w_i}', ['Новый объект','Расстояния до train','Ближайшие объекты','Среднее / голосование']),
 'Наивный Байес':('Вероятность класса вычисляется по теореме Байеса при условной независимости признаков внутри класса. Предположение упрощает модель; коррелирующие признаки могут делать вероятности излишне уверенными.',r'P(c\mid x)\propto P(c)\prod_j P(x_j\mid c)', ['Признаки X','Вероятности признаков внутри класса','Априорные вероятности классов','Постериорные оценки → класс']),
 'Дискриминантный анализ':('Классы описываются гауссовскими распределениями. LDA использует общую ковариацию классов, QDA — отдельную для каждого класса. Поэтому границы решения могут быть линейными или квадратичными.',r'\hat c=\arg\max_c\{\log P(c)+\log\mathcal N(x;\mu_c,\Sigma_c)\}', ['Признаки X','Средние и ковариации классов','Гауссовские оценки + априорные вероятности','Класс с максимальной оценкой']),
 'Ядерные методы':('Kernel Ridge сочетает гребневую регуляризацию и ядро сходства объектов. Нелинейность определяется ядром; матрица сходства требует памяти, растущей квадратично с размером train.',r'\hat y(x)=k(x,X)^T(K+\alpha I)^{-1}y', ['Признаки X','Ядро сходства с train','Регуляризованная система','Прогноз']),
 'Гауссовские процессы':('Гауссовский процесс задаёт распределение над функциями через среднюю функцию и ядро ковариации. Регрессия обновляет его по наблюдениям; классификация использует вероятностную связь и приближение.',r'f\sim\mathcal{GP}(m(x),k(x,x^\prime))', ['Признаки X','Ядро ковариации','Условное распределение по train','Средний прогноз / вероятность']),
 'Базовые модели':('Простая отправная точка, не использующая зависимость от X: среднее значение train для регрессии или наиболее частый класс train. Помогает понять, даёт ли сложная модель полезное улучшение.',r'\hat y=\bar y_{train}\quad\text{или}\quad\hat c=\operatorname{mode}(y_{train})', ['Цели train','Среднее / частоты классов','Постоянный ответ']),
}
SPECIAL={
 'Ridge':('L2-регуляризация уменьшает большие коэффициенты и стабилизирует решение при коррелирующих признаках.',r'\min_{w,b}\|y-Xw-b\|_2^2+\alpha\|w\|_2^2'),
 'Lasso':('L1-регуляризация может занулять коэффициенты и выполнять отбор признаков.',r'\min_{w,b}\frac{1}{2n}\|y-Xw-b\|_2^2+\alpha\|w\|_1'),
 'Lasso Lars':('Решает Lasso по пути наименьших углов; это другой алгоритм оптимизации той же L1-задачи.',r'\min_{w,b}\frac{1}{2n}\|y-Xw-b\|_2^2+\alpha\|w\|_1'),
 'Elastic Net':('Сочетает L1 и L2: разреженность коэффициентов и устойчивость к коррелирующим признакам.',r'\min_{w,b}\frac{\|y-Xw-b\|_2^2}{2n}+\alpha\rho\|w\|_1+\frac{\alpha(1-\rho)}{2}\|w\|_2^2'),
 'Logistic Regression':('Логистическая регрессия решает классификацию. Для двух классов логистическая функция преобразует линейную оценку в вероятность; для нескольких используется многоклассовая постановка.',r'P(y=1\mid x)=\sigma(w^Tx+b),\qquad\sigma(z)=\frac{1}{1+e^{-z}}'),
 'Huber Regression':('Устойчивая регрессия: малые остатки штрафуются квадратично, крупные — линейно; дополнительно применяется L2-регуляризация.',r'L_\epsilon(r)=\begin{cases}r^2,&|r|\le\epsilon\\2\epsilon|r|-\epsilon^2,&|r|>\epsilon\end{cases}'),
 'Bayesian Ridge':('Байесовская линейная регрессия с гауссовским априорным распределением весов; точность шума и регуляризации оценивается по данным.',r'y\mid X,w\sim\mathcal N(Xw,\alpha^{-1}I),\quad w\sim\mathcal N(0,\lambda^{-1}I)'),
 'ARD Regression':('Байесовская регрессия с отдельной точностью для каждого веса; модель может подавлять несущественные признаки.',r'w_j\sim\mathcal N(0,\lambda_j^{-1})'),
 'Lars':('Наименьшие углы: коэффициенты увеличиваются вдоль направлений, согласованных с наиболее коррелирующими с остатком признаками.',r'\hat y=Xw+b'),
 'SGD':('Стохастический градиентный спуск обновляет веса по отдельным наблюдениям. В лаборатории классификатор использует log loss, регрессор — стандартную квадратичную потерю.',r'w_{t+1}=w_t-\eta_t\nabla_w[\ell(y_t,f_w(x_t))+\alpha R(w)]'),
 'Passive Aggressive':('Обновляет веса, когда нарушается допустимая ошибка или зазор, стараясь минимально изменить прежние веса.',r'w_{t+1}=\arg\min_w\frac12\|w-w_t\|^2+C\xi\quad\text{при ограничении на потерю}'),
 'Perceptron':('Линейный классификатор обновляет веса при ошибочном распознавании. Его линейные оценки не являются калиброванными вероятностями.',r'w\leftarrow w+\eta y_i x_i\quad\text{при}\quad y_i(w^Tx_i+b)\le0'),
 'Ridge Classifier':('Классы кодируются в числовые цели и применяется гребневая регрессия; класс выбирается по линейным оценкам.',r'\min_W\|Y-XW\|_F^2+\alpha\|W\|_F^2'),
 'SVR':('Регрессия опорных векторов использует ε-нечувствительную потерю: малые ошибки не штрафуются.',r'\min_{w,b}\frac12\|w\|^2+C\sum_i\max(0,|y_i-f(x_i)|-\epsilon)'),
 'Nearest Centroid':('Выбирает класс ближайшего центра train. При евклидовом расстоянии центр — среднее, при манхэттенском — покоординатная медиана.',r'\hat c=\arg\min_c d(x,\mu_c)'),
 'Random Forest':('Множество деревьев обучается на bootstrap-подвыборках со случайным выбором признаков. Регрессия усредняет ответы; классификация усредняет вероятности классов.',r'\hat f(x)=\frac1B\sum_{b=1}^B T_b(x)'),
 'Extra Trees':('Ансамбль деревьев со случайными порогами разбиения. Случайность сильнее, чем у Random Forest; bootstrap по умолчанию отключён.',r'\hat f(x)=\frac1B\sum_{b=1}^B T_b(x)'),
 'Bagging':('Базовые модели обучаются независимо на подвыборках train, затем ответы объединяются. В текущей конфигурации используются деревья.',r'\hat f(x)=\operatorname{aggregate}(f_1(x),\ldots,f_B(x))'),
 'Extra Tree':('Одно дерево со случайным выбором порогов; в отличие от Extra Trees, усреднения ансамбля здесь нет.',r'\hat y(x)=c_{\operatorname{leaf}(x)}'),
 'AdaBoost':('Последовательный адаптивный ансамбль: вклад наблюдений или моделей меняется с учётом ошибок. Для классификации и регрессии используются разные алгоритмы AdaBoost.',r'F(x)=\operatorname{aggregate}(\alpha_1h_1(x),\ldots,\alpha_Mh_M(x))'),
 'Hist Gradient Boosting':('Перед поиском разбиений значения группируются в гистограммные интервалы, что ускоряет обучение на табличных данных.',r'F_m=F_{m-1}+\eta h_m'),
 'XGBoost':('Бустинг деревьев с регуляризованной целевой функцией и использованием градиентов и вторых производных потерь. В лаборатории используется histogram tree method.',r'\mathcal L=\sum_i\ell(y_i,\hat y_i)+\sum_m\Omega(h_m)'),
 'CatBoost':('Бустинг деревьев со специальными алгоритмами работы с категориальными признаками и упорядоченным обучением. В этом приложении X предварительно проходят общий one-hot pipeline; нативные категории не передаются.',r'F_M(x)=\sum_{m=1}^M\eta h_m(x)'),
 'LightGBM':('Гистограммный бустинг, который по умолчанию развивает наиболее выгодные листья. num_leaves и ограничения листьев контролируют сложность.',r'F_m=F_{m-1}+\eta h_m'),
 'Gaussian Naive Bayes':('Числовые признаки внутри каждого класса моделируются независимыми гауссовскими распределениями.',r'P(x_j\mid c)=\mathcal N(x_j;\mu_{cj},\sigma_{cj}^2)'),
 'Bernoulli Naive Bayes':('Признаки описываются независимыми бернуллиевскими событиями после бинаризации по выбранному порогу.',r'P(x\mid c)=\prod_j p_{cj}^{x_j}(1-p_{cj})^{1-x_j}'),
}


def theory(name,task):
    group=MODEL_REGISTRY[name]['group']
    text,formula,stages=PROFILES[group]
    if name in SPECIAL: text,formula=SPECIAL[name]
    if name=='Linear SVM' and task=='Регрессия':
        text=SPECIAL['SVR'][0]+' Используется линейная функция, без ядрового преобразования.';formula=SPECIAL['SVR'][1]
    if name=='LSTM':
        text='LSTM хранит скрытое состояние h и память c. Входные, забывающие и выходные ворота регулируют поток информации по временному окну.'
        formula=r'c_t=f_t\odot c_{t-1}+i_t\odot\widetilde c_t,\qquad h_t=o_t\odot\tanh(c_t)'
    elif name=='GRU':
        text='GRU использует ворота обновления и сброса; отдельного состояния памяти c нет. Обучение выполняется на окнах последовательных строк.'
        formula=r'h_t=(1-z_t)\odot n_t+z_t\odot h_{t-1}'
    elif name=='RNN':
        text='Обычная рекуррентная сеть переносит скрытое состояние между шагами окна. В отличие от LSTM/GRU, не имеет специальных ворот памяти.'
        formula=r'h_t=\tanh(W_{ih}x_t+b_{ih}+W_{hh}h_{t-1}+b_{hh})'
    if name=='Linear SVM' and task=='Классификация': formula=r'f(x)=w^Tx+b'
    if task=='Классификация' and group=='Соседние объекты' and name!='Nearest Centroid':
        formula=r'\hat c=\arg\max_c\sum_{i\in N_k(x)}w_i\mathbf 1(y_i=c)'
    if task=='Классификация' and group=='Гауссовские процессы':
        formula=r'f\sim\mathcal{GP}(0,k),\qquad P(y=1\mid f)=\sigma(f)'
    guidance = {
        'Линейные модели': {
            'intuition':'Модель назначает каждому признаку вклад с отдельным весом, затем суммирует вклады. Коэффициенты можно читать как направление и силу связи при фиксированных остальных признаках.',
            'when':'Хорошая отправная точка для числовых данных, интерпретируемых зависимостей и сравнения с более сложными алгоритмами.',
            'assumptions':'Зависимость приближённо линейна в используемых признаках. Для сравнения коэффициентов важен масштаб признаков; выбросы могут заметно влиять на обычную квадратичную ошибку.',
            'strengths':'Быстро обучается, даёт компактную формулу и понятные коэффициенты; регуляризация помогает при большом числе признаков.',
            'limits':'Без преобразования признаков не описывает сложные нелинейные взаимодействия. Коррелирующие признаки затрудняют отдельную интерпретацию весов.'},
        'Деревья и ансамбли': {
            'intuition':'Алгоритм делит пространство признаков на области. Для одного дерева ответом служит значение листа; ансамбль усредняет ответы многих деревьев.',
            'when':'Подходит для табличных данных, пороговых эффектов и нелинейностей; одиночное дерево удобно для наглядного разбора решений.',
            'assumptions':'Линейность и масштабирование признаков не требуются. Разбиения зависят от обучающей выборки и могут быть нестабильны при малых изменениях данных.',
            'strengths':'Улавливает взаимодействия и пороги, обычно не требует масштабирования; лес снижает разброс одиночных деревьев.',
            'limits':'Глубокое дерево переобучается и может быть нестабильным. Большой ансамбль сложнее объяснять и занимает больше ресурсов.'},
        'Бустинг': {
            'intuition':'Композиция строится по шагам: каждое новое небольшое дерево корректирует ошибки текущего прогноза.',
            'when':'Часто эффективен на структурированных таблицах, когда важны точность и нелинейные зависимости.',
            'assumptions':'Качество зависит от корректного разделения данных и настройки сложности композиции. Случайный split временных данных может дать утечку будущего.',
            'strengths':'Моделирует сложные взаимодействия и постепенно улучшает слабый базовый прогноз.',
            'limits':'Чувствителен к настройкам числа деревьев, глубины и learning rate; обучение последовательное и может быть медленнее леса.'},
        'Нейронные сети': {
            'intuition':'Слои последовательно преобразуют признаки; нелинейные активации позволяют строить сложные представления. Рекуррентные варианты дополнительно используют порядок наблюдений.',
            'when':'Полезны для сложных нелинейных зависимостей и последовательностей при достаточном объёме данных.',
            'assumptions':'Нужны корректные масштабирование и разбиение. Результат зависит от инициализации, архитектуры и оптимизации; для временных рядов порядок и временные окна должны быть осмысленны.',
            'strengths':'Гибкая архитектура, способная описывать нелинейные зависимости и последовательные эффекты.',
            'limits':'Сложнее объяснять и настраивать; небольшие таблицы часто не дают преимуществ перед деревьями.'},
        'Методы опорных векторов': {
            'intuition':'SVM ищет границу с большим зазором между классами; ядро позволяет задать нелинейную границу через сходство с обучающими объектами.',
            'when':'Подходит для задач с чёткой границей и небольшим или средним числом наблюдений; SVR применяет аналогичный принцип к регрессии.',
            'assumptions':'Масштаб признаков влияет на расстояния и зазор, поэтому подготовка признаков существенна. Параметр C задаёт компромисс между шириной зазора и ошибками.',
            'strengths':'Эффективен в пространствах с большим числом признаков; ядра позволяют моделировать нелинейность.',
            'limits':'Настройка ядра и C важна; обучение ядровых моделей плохо масштабируется на очень большие выборки.'},
        'Соседние объекты': {
            'intuition':'Для нового объекта находятся ближайшие строки train, затем их ответы усредняются или голосуют за класс.',
            'when':'Полезен как понятная локальная модель и для данных, где близкие объекты имеют близкие ответы.',
            'assumptions':'Расстояние должно отражать смысл близости. Признаки нужно масштабировать; нерелевантные измерения и высокая размерность ухудшают соседство.',
            'strengths':'Простая интерпретация на уровне похожих наблюдений; почти нет отдельного этапа подгонки параметров.',
            'limits':'Хранит train и медленно прогнозирует на больших наборах; результат чувствителен к k, метрике расстояния и масштабу.'},
        'Наивный Байес': {
            'intuition':'Сравнивает вероятности наблюдаемых признаков при каждом классе с априорной частотой класса.',
            'when':'Быстрый базовый классификатор, особенно для разреженных данных и высокой размерности.',
            'assumptions':'Ключевое упрощение — условная независимость признаков внутри класса; Gaussian вариант предполагает нормальные распределения числовых признаков.',
            'strengths':'Быстро обучается, экономно использует память и выдаёт вероятностные оценки.',
            'limits':'Корреляции могут приводить к чрезмерно уверенным вероятностям; форма распределения зависит от варианта алгоритма.'},
        'Дискриминантный анализ': {
            'intuition':'Оценивает распределение признаков в каждом классе и выбирает класс с наибольшей апостериорной оценкой.',
            'when':'Подходит для компактных классификационных задач с приблизительно эллиптическими распределениями признаков.',
            'assumptions':'LDA предполагает общую ковариацию классов, QDA — отдельную ковариацию. Оценка ковариаций требует достаточно наблюдений.',
            'strengths':'Даёт компактную вероятностную модель и объяснимую границу; часто служит хорошим базовым методом.',
            'limits':'Чувствителен к выбросам и плохо обусловленным ковариациям; предположения о распределениях могут не выполняться.'},
        'Ядерные методы': {
            'intuition':'Ядро измеряет сходство объектов. Регуляризованная комбинация сходств с объектами train задаёт нелинейный прогноз.',
            'when':'Для гладких нелинейных зависимостей, когда размер выборки позволяет хранить матрицу попарных сходств.',
            'assumptions':'Выбор ядра и его масштаба задаёт понятие похожести; регуляризация балансирует точность и гладкость.',
            'strengths':'Описывает нелинейность без явного ручного построения новых признаков.',
            'limits':'Память и вычисления быстро растут с числом строк; интерпретация отдельных влияний сложнее.'},
        'Гауссовские процессы': {
            'intuition':'Вместо одной функции задаётся распределение возможных функций. После наблюдений модель получает средний прогноз и неопределённость.',
            'when':'Для небольших выборок, когда важны гладкость прогноза и оценка неопределённости.',
            'assumptions':'Структуру функций задаёт ядро; его выбор выражает предположение о гладкости и масштабе изменений.',
            'strengths':'Вероятностный прогноз с оценкой неопределённости и гибкое описание гладких зависимостей.',
            'limits':'Кубический рост вычислений и квадратичная память ограничивают размер обучающей выборки.'},
        'Базовые модели': {
            'intuition':'Игнорирует признаки и выдаёт постоянное значение: среднее для регрессии или наиболее частый класс.',
            'when':'Используйте как обязательную точку отсчёта, чтобы проверить, превосходит ли модель простую стратегию.',
            'assumptions':'Не использует X и не моделирует закономерности между признаками и целью.',
            'strengths':'Почти мгновенно обучается и даёт честный ориентир для метрик.',
            'limits':'Не предназначена для содержательных предсказаний; одинаковый прогноз для всех строк.'},
    }
    insight=guidance[group]
    stage_explanations=[]
    for index,stage in enumerate(stages):
        if index == 0:
            explanation=f'Здесь формируется вход алгоритма: {stage.lower()}. {insight["assumptions"]}'
        elif index == len(stages)-1:
            explanation=f'На выходе этапа «{stage}» получается итоговый ответ модели. {insight["intuition"]}'
        elif group == 'Линейные модели':
            explanation='Для каждого подготовленного признака берётся обученный коэффициент; вклады складываются, а свободный член задаёт базовый уровень прогноза.'
        elif group in ('Деревья и ансамбли','Бустинг'):
            explanation='Модель проверяет условия по признакам и направляет наблюдение в соответствующую ветвь. В ансамбле ответы деревьев объединяются.'
        elif group == 'Нейронные сети':
            explanation='Скрытые слои преобразуют представление признаков с обученными весами и нелинейной активацией; в рекуррентных сетях сохраняется состояние окна.'
        elif group == 'Методы опорных векторов':
            explanation='Опорные наблюдения и функция сходства определяют положение границы или регрессионной функции; регуляризация ограничивает сложность.'
        elif group == 'Соседние объекты':
            explanation='Сравниваются расстояния до обучающих наблюдений. Число соседей и метрика определяют, чьи ответы повлияют на прогноз.'
        elif group == 'Наивный Байес':
            explanation='Оцениваются правдоподобия признаков внутри классов и их априорные вероятности; условные вклады комбинируются по формуле Байеса.'
        elif group == 'Дискриминантный анализ':
            explanation='Параметры распределений классов используются для расчёта оценки принадлежности наблюдения каждому классу.'
        elif group == 'Ядерные методы':
            explanation='Ядро рассчитывает сходство с обучающими строками, а регуляризованная система использует эти сходства для прогноза.'
        elif group == 'Гауссовские процессы':
            explanation='Ядро связывает наблюдения по ковариации и обновляет распределение функций с учётом обучающих данных.'
        else:
            explanation='На этапе используется статистика обучающей выборки; признаки не влияют на постоянный базовый прогноз.'
        stage_explanations.append(explanation)
    return dict(text=text,formula=formula,stages=stages,group=group,
                intuition=insight['intuition'],whenToUse=insight['when'],
                assumptions=insight['assumptions'],strengths=insight['strengths'],
                limitations=insight['limits'],stageExplanations=stage_explanations)


def architecture_dot(name,task,params=None):
    profile=theory(name,task);params=params or {}
    stages=list(profile['stages'])
    if name=='MLP Neural Network':
        sizes=params.get('hidden_layer_sizes',params.get('hidden_layers',[64,32]))
        stages=['Признаки X']+[f'Dense: {units} нейронов / {params.get("activation","relu")}' for units in sizes]+['Выход: значение' if task=='Регрессия' else 'Выход: вероятности классов']
    elif name in ('LSTM','GRU','RNN'):
        stages=[f'Окно: {params.get("window",12)} строк × признаки']+[f'{name}: {params.get("units",32)} нейронов / слой {i+1}' for i in range(params.get('layers',1))]+[
            'Последнее скрытое состояние',f'Dropout {params.get("dropout",.1)}',f'Dense {params.get("dense_units",16)} + ReLU','Dense 1 → масштаб цели']
    elif name in ('Random Forest','Extra Trees','Bagging'):
        stages=['Признаки X',f'{params.get("n_estimators",params.get("trees",100 if name!="Bagging" else 10))} независимых деревьев','Усреднение ответов / вероятностей','Прогноз']
    escape=lambda value:json.dumps(str(value),ensure_ascii=False)
    lines=['digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#e6f4f1", color="#0f766e"];']
    for i,label in enumerate(stages):
        lines.append(f'n{i} [label={escape(label)}];')
        if i: lines.append(f'n{i-1} -> n{i};')
    if name in ('LSTM','GRU','RNN'): lines.append('n1 -> n1 [label="состояние внутри окна"];')
    lines.append('}')
    return '\n'.join(lines)


def fitted_tree_dot(result,max_depth=3):
    model=result['pipeline'].named_steps['model']
    if not hasattr(model,'tree_'):
        estimators=getattr(model,'estimators_',None)
        if estimators is None: return None
        model=np.asarray(estimators,dtype=object).ravel()[0]
    if not hasattr(model,'tree_'): return None
    names=result['pipeline'].named_steps['preprocess'].get_feature_names_out().tolist()
    # Bagging may train its first estimator on a subset of transformed features.
    outer=result['pipeline'].named_steps['model']
    subsets=getattr(outer,'estimators_features_',None)
    if subsets is not None: names=[names[i] for i in subsets[0]]
    return export_graphviz(model,out_file=None,feature_names=names,filled=True,rounded=True,max_depth=max_depth)


def render_model_descriptions():
    st.subheader('Описание моделей')
    task=st.session_state.get('cfg_task','Регрессия')
    options=MODELS[task]
    current=st.session_state.get('chosen_model',options[0])
    if current not in options: current=options[0]
    st.caption('Выберите одну или несколько моделей. Теория и концептуальные схемы доступны до обучения; фактическое дерево — после обучения поддерживаемой модели.')
    key='description_models_'+task
    st.session_state.setdefault(key,[current])
    a,b=st.columns(2)
    if a.button('Добавить текущую модель',key='description_add_current'):
        st.session_state[key]=list(dict.fromkeys(st.session_state[key]+[current]))
    result=st.session_state.get('result')
    if b.button('Добавить модели последнего эксперимента',key='description_add_runs',disabled=not result or 'error' in result):
        selected=[row['Модель'] for row in result['leaderboard'] if row['Модель'] in options]
        st.session_state[key]=list(dict.fromkeys(st.session_state[key]+selected))
    selected=st.multiselect('Модели для изучения',options,key=key)
    for name in selected:
        with st.expander(name+' · '+MODEL_REGISTRY[name]['group'],expanded=True):
            profile=theory(name,task)
            st.write(profile['text']);st.latex(profile['formula'])
            st.caption('x — признаки, y — цель, веса и параметры оцениваются на train. Формула показывает основную идею; детали зависят от задачи и реализации.')
            params={k:spec['default'] for k,spec in MODEL_REGISTRY[name]['parameters'].items()}
            origin='Значения параметров по умолчанию'
            if result and result.get('name')==name and result.get('task')==task:
                params.update(result['params'])
                fitted=result['pipeline'].named_steps['model']
                fitted=getattr(fitted,'regressor_',fitted)
                fitted=getattr(fitted,'estimator_',fitted)
                effective=fitted.get_params(deep=False)
                params.update({field:effective[field] for field in params if field in effective})
                origin='Параметры последней обученной модели'
            elif name==current:
                for field in params:
                    widget=f'model_{task}_{name}_Ручная настройка_{field}'
                    if widget in st.session_state: params[field]=st.session_state[widget]
                if name=='MLP Neural Network':
                    base=f'model_{task}_{name}_Ручная настройка_hidden_layer_sizes'
                    count=st.session_state.get(base+'_count',2)
                    params['hidden_layer_sizes']=[st.session_state.get(base+'_'+str(i),64 if i==0 else 32) for i in range(count)]
                origin='Текущие ручные настройки или значения по умолчанию'
            st.markdown('**Концептуальная схема**');st.graphviz_chart(architecture_dot(name,task,params))
            st.caption(origin+'. Схема показывает состав вычислений; число связей и обученные веса в ней не воспроизводятся.')
            if result and result.get('name')==name and 'pipeline' in result:
                dot=fitted_tree_dot(result,3)
                if dot:
                    depth=st.slider('Глубина отображения дерева',1,5,3,key='theory_depth_'+name)
                    if depth!=3: dot=fitted_tree_dot(result,depth)
                    st.markdown('**Фактически обученное дерево**');st.graphviz_chart(dot)
                    st.caption('Для ансамбля показано первое дерево. X и пороги относятся к признакам после preprocessing; глубокие ветви свёрнуты. samples — объём узла, value — средняя цель или распределение классов. Путь от корня до листа показывает последовательность условий для ответа.')
            st.markdown('**Параметры реализации**');st.json(params)
            cls=MODEL_REGISTRY[name]['classes'][task]
            st.code(str(inspect.signature(cls)),language='text')
            summary=(inspect.getdoc(cls) or '').split('Parameters')[0].strip()
            if summary:
                st.markdown('**Справка установленной библиотеки**')
                st.text(' '.join(summary.split()[:80]))
                st.caption('Извлечено из документации установленного класса; доступно без интернета.')
            api=documentation(name,task)
            guide='https://scikit-learn.org/stable/modules/'+GUIDES[profile['group']]
            if name in ('LSTM','GRU','RNN'): guide=api
            elif name=='XGBoost': guide='https://xgboost.readthedocs.io/en/stable/tutorials/model.html'
            elif name=='CatBoost': guide='https://catboost.ai/docs/en/concepts/algorithm-main-stages'
            elif name=='LightGBM': guide='https://lightgbm.readthedocs.io/en/stable/Features.html'
            st.markdown('**Материалы официальной документации · загружаются автоматически**')
            anchors={'Linear Regression':'ordinary-least-squares','Ridge':'ridge-regression-and-classification',
                'Lasso':'lasso','Elastic Net':'elastic-net','Bayesian Ridge':'bayesian-ridge-regression',
                'ARD Regression':'automatic-relevance-determination-ard','Logistic Regression':'logistic-regression',
                'Lars':'least-angle-regression','Lasso Lars':'lars-lasso','Huber Regression':'huber-regression',
                'Random Forest':'forests-of-randomized-trees','Extra Trees':'forests-of-randomized-trees',
                'Bagging':'bagging-meta-estimator','AdaBoost':'adaboost','Gradient Boosting':'gradient-boosting',
                'Hist Gradient Boosting':'histogram-based-gradient-boosting'}
            if name in ('Decision Tree','Extra Tree'): guide='https://scikit-learn.org/stable/modules/tree.html'
            if name in anchors: guide=guide.split('#')[0]+'#'+anchors[name]
            render_online_document(guide,task+'_'+name)
            st.markdown(f'[Официальный API модели]({api}) · [Теория и алгоритм]({guide})')
            if st.checkbox('Показать официальный API внутри вкладки',key='model_inline_docs_'+task+'_'+name):
                components.iframe(api,height=600,scrolling=True)
                st.caption('Если сайт запрещает встроенный просмотр, откройте официальный API по ссылке выше.')
            st.caption('Установленная библиотека: '+MODEL_REGISTRY[name]['package']+' '+version(MODEL_REGISTRY[name]['package'])+'. Ссылки открывают актуальную документацию разработчиков; её версия может отличаться от установленной.')
    if st.button('Обновить онлайн-сведения библиотек',key='description_online'):
        st.session_state.description_online=online_catalog_status(include_extensions=True)
    if st.session_state.get('description_online'): st.dataframe(st.session_state.description_online,width='stretch',hide_index=True)
