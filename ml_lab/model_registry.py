"""Reviewed model catalogue and UI schemas; only installed, trusted library classes."""
from importlib.metadata import version, PackageNotFoundError
from importlib import import_module
import math
import requests
from sklearn import linear_model as lm, ensemble as en, tree, svm, neighbors, naive_bayes, discriminant_analysis as da
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from ml_lab.adapters import EncodedClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from ml_lab.neural import RecurrentRegressor

REG, CLS = 'Регрессия', 'Классификация'


def number(label, low, high, default, step=1, log=False):
    return dict(label=label, type='int' if isinstance(default,int) else 'float', min=low, max=high, default=default, step=step, log=log)


def choice(label, values, default=None):
    return dict(label=label,type='choice',choices=values,default=values[0] if default is None else default)


ALPHA = number('Регуляризация · alpha', .000001, 10., .1, .000001, True)
ITER = number('Максимум итераций', 10, 2000, 200, 10)
DEPTH = number('Глубина деревьев · 0 = без ограничения', 0, 40, 6)
TREES = number('Количество деревьев', 10, 500, 100, 10)
RATE = number('Learning rate', .0001, 1., .05, .0001, True)
C = number('C · обратная регуляризация', .001, 1000., 1., .001, True)
LEAF = number('Минимум строк в листе', 1, 50, 1)
FIT = choice('Свободный член', [True,False])
DENSE = dict(hidden_layer_sizes=dict(label='Архитектура скрытых слоёв',type='layers',default=[64,32]),
             activation=choice('Функция активации',['relu','tanh','logistic','identity']),
             learning_rate_init=number('Learning rate',.0001,.1,.001,.0001,True), alpha=ALPHA,max_iter=ITER)
RECURRENT = dict(layers=number('Рекуррентных слоёв',1,3,1), units=number('Нейронов в слое',8,256,32,8),
    window=number('Окно последовательности · строк',2,96,12),dense_units=number('Нейронов в Dense-слое',8,128,16,8),
    dropout=number('Dropout',0.,.5,.1,.05),learning_rate=number('Learning rate',.0001,.1,.001,.0001,True),
    epochs=number('Максимум эпох',2,200,30),batch_size=choice('Batch size',[16,32,64,128],64),optimizer=choice('Оптимизатор',['Adam','RMSprop','SGD']))
MODEL_REGISTRY = {}


def register(name, reg=None, cls=None, group='Классическое ML', parameters=None):
    MODEL_REGISTRY[name] = dict(classes={task: c for task,c in [(REG,reg),(CLS,cls)] if c},group=group,parameters=parameters or {})


register('Linear Regression',lm.LinearRegression,parameters={'fit_intercept':FIT})
register('Ridge',lm.Ridge,parameters={'alpha':ALPHA,'fit_intercept':FIT,'max_iter':ITER})
register('Lasso',lm.Lasso,parameters={'alpha':ALPHA,'fit_intercept':FIT,'max_iter':ITER})
register('Elastic Net',lm.ElasticNet,parameters={'alpha':ALPHA,'l1_ratio':number('Доля L1-регуляризации',0.,1.,.5,.05),'max_iter':ITER})
register('Bayesian Ridge',lm.BayesianRidge,parameters={'tol':number('Допуск сходимости',.000001,.1,.001,.000001,True)})
register('Huber Regression',lm.HuberRegressor,parameters={'epsilon':number('Порог устойчивости',1.01,3.,1.35,.01),'alpha':ALPHA,'max_iter':ITER})
register('SGD',lm.SGDRegressor,lm.SGDClassifier,parameters={'alpha':ALPHA,'max_iter':ITER,'penalty':choice('Регуляризация',['l2','l1','elasticnet'])})
register('Logistic Regression',cls=lm.LogisticRegression,parameters={'C':C,'max_iter':ITER})
register('Decision Tree',tree.DecisionTreeRegressor,tree.DecisionTreeClassifier,group='Деревья и ансамбли',parameters={'max_depth':DEPTH,'min_samples_leaf':LEAF})
for name,reg,cls in [('Random Forest',en.RandomForestRegressor,en.RandomForestClassifier),('Extra Trees',en.ExtraTreesRegressor,en.ExtraTreesClassifier)]:
    register(name,reg,cls,group='Деревья и ансамбли',parameters={'n_estimators':TREES,'max_depth':DEPTH,'min_samples_leaf':LEAF})
register('Gradient Boosting',en.GradientBoostingRegressor,en.GradientBoostingClassifier,group='Деревья и ансамбли',parameters={'n_estimators':TREES,'learning_rate':RATE,'max_depth':number('Глубина деревьев',1,12,3),'min_samples_leaf':LEAF})
register('Hist Gradient Boosting',en.HistGradientBoostingRegressor,en.HistGradientBoostingClassifier,group='Деревья и ансамбли',parameters={'max_iter':ITER,'learning_rate':RATE,'max_depth':DEPTH,'l2_regularization':number('L2-регуляризация',0.,10.,.0,.01)})
register('AdaBoost',en.AdaBoostRegressor,en.AdaBoostClassifier,group='Деревья и ансамбли',parameters={'n_estimators':TREES,'learning_rate':RATE})
register('Bagging',en.BaggingRegressor,en.BaggingClassifier,group='Деревья и ансамбли',parameters={'n_estimators':number('Количество моделей',5,100,10,5),'max_samples':number('Доля строк для модели',.1,1.,1.,.1)})
register('SVR',svm.SVR,group='Методы опорных векторов',parameters={'C':C,'epsilon':number('Epsilon · допустимая ошибка',0.,10.,.1,.01),'kernel':choice('Ядро',['rbf','linear','poly','sigmoid'])})
register('SVC',cls=svm.SVC,group='Методы опорных векторов',parameters={'C':C,'kernel':choice('Ядро',['rbf','linear','poly','sigmoid'])})
register('Linear SVM',svm.LinearSVR,svm.LinearSVC,group='Методы опорных векторов',parameters={'C':C,'max_iter':ITER})
register('KNN',neighbors.KNeighborsRegressor,neighbors.KNeighborsClassifier,parameters={'n_neighbors':number('Количество соседей',1,50,5),'weights':choice('Веса соседей',['uniform','distance'])})
register('Gaussian Naive Bayes',cls=naive_bayes.GaussianNB,parameters={'var_smoothing':number('Сглаживание дисперсии',1e-12,.01,1e-9,1e-12,True)})
register('Bernoulli Naive Bayes',cls=naive_bayes.BernoulliNB,parameters={'alpha':ALPHA,'binarize':number('Порог бинаризации',-5.,5.,0.,.1)})
register('Linear Discriminant Analysis',cls=da.LinearDiscriminantAnalysis,parameters={'tol':number('Допуск',.000001,.1,.0001,.000001,True)})
register('Quadratic Discriminant Analysis',cls=da.QuadraticDiscriminantAnalysis,parameters={'reg_param':number('Регуляризация ковариации',0.,1.,.1,.05)})
register('MLP Neural Network',MLPRegressor,MLPClassifier,group='Нейронные сети',parameters=DENSE)
for name in ['LSTM','GRU','RNN']:
    register(name,RecurrentRegressor,group='Нейронные сети',parameters=RECURRENT)

# Additional installed-library estimators share the same configuration/search schema.
register('ARD Regression',lm.ARDRegression,group='Линейные модели',parameters={'tol':number('Допуск',.000001,.1,.001,.000001,True)})
register('Lars',lm.Lars,group='Линейные модели',parameters={'fit_intercept':FIT})
register('Lasso Lars',lm.LassoLars,group='Линейные модели',parameters={'alpha':ALPHA,'max_iter':ITER})
register('Passive Aggressive',lm.PassiveAggressiveRegressor,lm.PassiveAggressiveClassifier,group='Линейные модели',parameters={'C':C,'max_iter':ITER})
register('Perceptron',cls=lm.Perceptron,group='Линейные модели',parameters={'alpha':ALPHA,'max_iter':ITER})
register('Ridge Classifier',cls=lm.RidgeClassifier,group='Линейные модели',parameters={'alpha':ALPHA,'fit_intercept':FIT})
register('Extra Tree',tree.ExtraTreeRegressor,tree.ExtraTreeClassifier,group='Деревья и ансамбли',parameters={'max_depth':DEPTH,'min_samples_leaf':LEAF})
register('Nearest Centroid',cls=neighbors.NearestCentroid,group='Соседние объекты',parameters={'metric':choice('Расстояние',['euclidean','manhattan'])})
register('Kernel Ridge',KernelRidge,group='Ядерные методы',parameters={'alpha':ALPHA,'kernel':choice('Ядро',['linear','rbf','poly']),'gamma':number('Gamma',.0001,10.,.1,.0001,True)})
register('Gaussian Process',GaussianProcessRegressor,GaussianProcessClassifier,group='Гауссовские процессы',parameters={'n_restarts_optimizer':number('Перезапусков оптимизатора',0,3,0)})
register('Dummy Baseline',DummyRegressor,DummyClassifier,group='Базовые модели')

for name in ['Linear Regression','Ridge','Lasso','Elastic Net','Bayesian Ridge','Huber Regression','SGD','Logistic Regression']:
    MODEL_REGISTRY[name]['group']='Линейные модели'
for name in ['Gradient Boosting','Hist Gradient Boosting','AdaBoost']:
    MODEL_REGISTRY[name]['group']='Бустинг'
MODEL_REGISTRY['KNN']['group']='Соседние объекты'
for name in ['Gaussian Naive Bayes','Bernoulli Naive Bayes']: MODEL_REGISTRY[name]['group']='Наивный Байес'
for name in ['Linear Discriminant Analysis','Quadratic Discriminant Analysis']: MODEL_REGISTRY[name]['group']='Дискриминантный анализ'

OPTIONAL_MODELS={
    'XGBoost':dict(package='xgboost',reg='XGBRegressor',cls='XGBClassifier',url='https://xgboost.readthedocs.io/en/stable/python/python_api.html',
        parameters={'n_estimators':TREES,'max_depth':number('Глубина деревьев',1,16,6),'learning_rate':RATE,
            'subsample':number('Доля строк',.1,1.,1.,.1),'colsample_bytree':number('Доля признаков',.1,1.,1.,.1),'reg_lambda':ALPHA}),
    'CatBoost':dict(package='catboost',reg='CatBoostRegressor',cls='CatBoostClassifier',url='https://catboost.ai/docs/en/concepts/python-reference_catboost',
        parameters={'iterations':number('Количество итераций',10,500,100,10),'depth':number('Глубина деревьев',2,10,6),'learning_rate':RATE,'l2_leaf_reg':ALPHA}),
    'LightGBM':dict(package='lightgbm',reg='LGBMRegressor',cls='LGBMClassifier',url='https://lightgbm.readthedocs.io/en/stable/Python-API.html',
        parameters={'n_estimators':TREES,'num_leaves':number('Количество листьев',2,128,31),'learning_rate':RATE,
            'min_child_samples':number('Минимум строк в листе',2,100,20),'colsample_bytree':number('Доля признаков',.1,1.,1.,.1)}),
}
OPTIONAL_STATUS={}
for name,spec in OPTIONAL_MODELS.items():
    try:
        module=import_module(spec['package'])
        register(name,getattr(module,spec['reg']),getattr(module,spec['cls']),group='Бустинг',parameters=spec['parameters'])
        OPTIONAL_STATUS[name]='Доступно'
    except (ImportError,OSError,AttributeError,ValueError,RuntimeError) as exc:
        OPTIONAL_STATUS[name]=f'Недоступно: {exc}'

for name,spec in MODEL_REGISTRY.items():
    spec['package']=OPTIONAL_MODELS[name]['package'] if name in OPTIONAL_MODELS else ('torch' if name in ('LSTM','GRU','RNN') else 'scikit-learn')
    if name in ('Gaussian Process','Kernel Ridge'):
        spec['max_rows']=2000 if name=='Gaussian Process' else 5000


def available_models(task,split='time',category=None,query=''):
    return [name for name,spec in MODEL_REGISTRY.items() if task in spec['classes']
        and (split=='time' or name not in RECURRENT_MODELS)
        and (not category or category=='Все категории' or spec['group']==category)
        and (not query or query.casefold() in (name+' '+spec['group']+' '+spec['package']).casefold())]


def catalog_rows(task):
    rows=[{'Модель':name,'Категория':spec['group'],'Библиотека':spec['package'],'Статус':'Доступно',
        'Документация':documentation(name,task)} for name,spec in MODEL_REGISTRY.items() if task in spec['classes']]
    for name,spec in OPTIONAL_MODELS.items():
        if name not in MODEL_REGISTRY:
            rows.append({'Модель':name,'Категория':'Бустинг','Библиотека':spec['package'],
                'Статус':OPTIONAL_STATUS[name],'Документация':spec['url']})
    return rows


MODELS = {task:[name for name,spec in MODEL_REGISTRY.items() if task in spec['classes']] for task in [REG,CLS]}
RECURRENT_MODELS = {'LSTM','GRU','RNN'}


def documentation(name,task):
    cls=MODEL_REGISTRY[name]['classes'][task]
    if name in RECURRENT_MODELS:
        return f'https://docs.pytorch.org/docs/stable/generated/torch.nn.{name}.html'
    if name in OPTIONAL_MODELS: return OPTIONAL_MODELS[name]['url']
    module=cls.__module__.split('.')[1]
    return f'https://scikit-learn.org/stable/modules/generated/sklearn.{module}.{cls.__name__}.html'


def model_defaults(name,task):
    return MODEL_REGISTRY[name]['classes'][task]().get_params(deep=False)


def build_estimator(name,task,params):
    spec=MODEL_REGISTRY[name]
    cls=spec['classes'][task]
    # Compatibility with configs saved by the previous UI.
    aliases={'n_estimators':'trees','max_depth':'depth','min_samples_leaf':'min_samples_leaf',
             'max_iter':'iterations','hidden_layer_sizes':'hidden_layers','learning_rate_init':'learning_rate','n_neighbors':'neighbors'}
    values={k:params.get(k, params.get(aliases.get(k,''),v['default'])) for k,v in spec['parameters'].items()}
    if 'max_depth' in values and values['max_depth']==0: values['max_depth']=None
    if 'hidden_layer_sizes' in values: values['hidden_layer_sizes']=tuple(values['hidden_layer_sizes'])
    supported=cls().get_params(deep=False)
    if 'random_state' in supported: values['random_state']=params.get('seed',42)
    if 'n_jobs' in supported: values['n_jobs']=2
    if name=='SGD' and task==CLS: values['loss']='log_loss'
    if name=='MLP Neural Network':
        values.update(early_stopping=False,shuffle=params.get('split')!='time')
    if name=='Hist Gradient Boosting': values['early_stopping']=False
    if name in RECURRENT_MODELS: values.update(seed=params.get('seed',42),cell=name)
    if name=='XGBoost': values.update(tree_method='hist',verbosity=0)
    if name=='CatBoost': values.update(random_seed=params.get('seed',42),thread_count=2,verbose=False,allow_writing_files=False)
    if name=='LightGBM': values['verbosity']=-1
    estimator=cls(**values)
    if name=='XGBoost' and task==CLS: estimator=EncodedClassifier(estimator)
    if name=='MLP Neural Network' and task==REG:
        estimator=TransformedTargetRegressor(regressor=estimator,transformer=StandardScaler())
    return estimator


def sample_parameters(name, task, base, space, trial=None, rng=None):
    """One schema drives both random range exploration and Optuna suggestions."""
    values=dict(base)|space.get('__fixed__',{})
    for key,s in space.items():
        if key=='__fixed__': continue
        label=f'{name}:{key}'
        if s['type']=='choice':
            options=s['choices']
            values[key]=trial.suggest_categorical(label,options) if trial else options[int(rng.integers(len(options)))]
        else:
            low,high=s['min'],s['max']
            if s['type']=='int':
                step=s.get('step',1)
                values[key]=trial.suggest_int(label,int(low),int(high),step=step) if trial else int(low+step*rng.integers(0,int((high-low)//step)+1))
            else:
                values[key]=trial.suggest_float(label,float(low),float(high),log=s.get('log',False)) if trial else float(10**rng.uniform(math.log10(low),math.log10(high)) if s.get('log') else rng.uniform(low,high))
    if name=='MLP Neural Network' and ('layer_count' in values or 'layer_units' in values):
        values['hidden_layer_sizes']=[values.get('layer_units',64)]*values.get('layer_count',2)
    return values


def online_catalog_status(include_extensions=False):
    """Read official package metadata only; never import code from a remote catalogue."""
    rows=[]
    packages=[('scikit-learn','https://scikit-learn.org/stable/supervised_learning.html'),('torch','https://docs.pytorch.org/docs/stable/nn.html')]
    if include_extensions: packages.extend((spec['package'],spec['url']) for spec in OPTIONAL_MODELS.values())
    for package,url in packages:
        try: installed=version(package)
        except PackageNotFoundError: installed='Не установлено'
        row={'Источник':package,'Установлено':installed,'Документация':url}
        try:
            response=requests.get(f'https://pypi.org/pypi/{package}/json',timeout=(10,15))
            response.raise_for_status()
            row.update({'Версия в PyPI':response.json()['info']['version'],'Статус':'Подключено'})
        except (requests.RequestException,KeyError,ValueError):
            row.update({'Версия в PyPI':'—','Статус':'Источник недоступен · локальный каталог сохранён'})
        rows.append(row)
    return rows
