"""Causal learned repair: labels and lag anchors are preceding measured observations only."""
import warnings
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = ('ridge', 'random_forest', 'extra_trees', 'hist_gradient_boosting', 'mlp', 'lstm', 'gru')
NEURAL = ('mlp', 'lstm', 'gru')


def integer(config, name, default, low, high):
    value=config.get(name, default)
    try:
        number=int(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f'Восстановление: {name} должно быть целым числом от {low} до {high}.') from None
    if isinstance(value, bool) or number!=value or not low<=number<=high:
        raise ValueError(f'Восстановление: {name} должно быть целым числом от {low} до {high}.')
    return number


class LearnedRepair:
    def __init__(self, config):
        self.name=config.get('model', 'ridge')
        if self.name not in MODELS:
            raise ValueError('Неизвестная модель восстановления.')
        self.lags=integer(config, 'lags', 6, 1, 48)
        self.minimum=integer(config, 'min_samples', 48, 16, 2048)
        self.train_window=integer(config, 'train_window', 256, self.minimum, 4096)
        self.refit=integer(config, 'refit_interval', 128, 1, 4096)
        self.max_fits=integer(config, 'max_fits', 16, 1, 64)
        self.seed=integer(config, 'seed', 42, 0, 2**31-1)
        self.trees=integer(config, 'trees', 50, 10, 200)
        self.depth=integer(config, 'depth', 6, 1, 16)
        self.units=integer(config, 'units', 16, 4, 128)
        self.epochs=integer(config, 'epochs', 10, 1, 100)
        self.iterations=integer(config, 'iterations', 150, 20, 1000)
        self.sequence=integer(config, 'sequence', 6, 2, 24)
        if self.name in ('lstm','gru') and self.minimum < self.sequence+12:
            raise ValueError('Для LSTM/GRU минимум примеров должен быть не меньше окна сети + 12.')
        self.fallback=config.get('fallback', 'forward')
        if self.fallback not in ('none','forward','linear'):
            raise ValueError('Неизвестный резервный способ восстановления.')
        self.observed=[];self.X=[];self.y=[];self.positions=[];self.context=[]
        self.accepted_count=0;self.attempted_count=None;self.model=None
        self.fits=0;self.attempts=0;self.samples=0;self.model_filled=0;self.fallback_filled=0
        self.reasons=set();self.fit_warning=''

    def features(self, stamp, seconds):
        if len(self.observed)<self.lags:
            return None
        hour=stamp.hour+stamp.minute/60+stamp.second/3600
        week=stamp.dayofweek+hour/24
        seasonal=[np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
                  np.sin(2*np.pi*week/7),np.cos(2*np.pi*week/7),
                  np.sin(2*np.pi*(stamp.dayofyear-1)/365.25),np.cos(2*np.pi*(stamp.dayofyear-1)/365.25),seconds/86400]
        history=self.observed[-self.lags:][::-1]
        return np.asarray(seasonal+[value for value,_ in history]+[(seconds-time)/3600 for _,time in history],dtype=float)

    def accept(self, vector, value, seconds, position):
        if vector is not None:
            self.X.append(vector);self.y.append(value);self.positions.append(position);self.accepted_count+=1
            self.X=self.X[-self.train_window:];self.y=self.y[-self.train_window:];self.positions=self.positions[-self.train_window:]
        self.observed.append((value,seconds));self.observed=self.observed[-self.lags:]

    def remember(self, vector):
        if vector is not None:
            self.context.append(vector);self.context=self.context[-self.sequence:]
        else:
            self.context=[]

    def build(self):
        estimators={
            'ridge':lambda:Ridge(alpha=1.),
            'random_forest':lambda:RandomForestRegressor(n_estimators=self.trees,max_depth=self.depth,min_samples_leaf=2,random_state=self.seed,n_jobs=1),
            'extra_trees':lambda:ExtraTreesRegressor(n_estimators=self.trees,max_depth=self.depth,min_samples_leaf=2,random_state=self.seed,n_jobs=1),
            'hist_gradient_boosting':lambda:HistGradientBoostingRegressor(max_iter=self.iterations,max_leaf_nodes=15,max_depth=self.depth,early_stopping=False,random_state=self.seed),
            'mlp':lambda:MLPRegressor(hidden_layer_sizes=(self.units,),max_iter=self.iterations,random_state=self.seed,early_stopping=False,shuffle=False),
        }
        if self.name in ('lstm','gru'):
            from ml_lab.neural import RecurrentRegressor
            model=RecurrentRegressor(units=self.units,layers=1,dropout=0.,dense_units=self.units,
                                     window=self.sequence,epochs=self.epochs,batch_size=64,seed=self.seed,cell=self.name.upper())
            model.fit_breaks_=np.r_[True,np.diff(self.positions)!=1]
        else:
            model=TransformedTargetRegressor(regressor=estimators[self.name](),transformer=StandardScaler())
        return make_pipeline(StandardScaler(),model)

    def predict(self, vector):
        if vector is None or len(self.y)<self.minimum:
            self.reasons.add('Недостаточно прошлых измерений для обучения')
            return None
        needs_fit=self.attempted_count is None or self.accepted_count-self.attempted_count>=self.refit
        if needs_fit and self.attempts<self.max_fits:
            self.attempted_count=self.accepted_count;self.attempts+=1
            try:
                model=self.build()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    if self.name in ('lstm','gru'):
                        import torch
                        with torch.random.fork_rng(devices=[]):
                            model.fit(np.asarray(self.X),np.asarray(self.y))
                    else:
                        model.fit(np.asarray(self.X),np.asarray(self.y))
                self.model=model;self.fits+=1;self.samples=len(self.y)
                self.fit_warning='; '.join(dict.fromkeys(str(w.message) for w in caught))[:500]
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                self.reasons.add('Ошибка обучения: '+str(exc)[:200])
        if self.model is None:
            return None
        try:
            if self.name in ('lstm','gru'):
                # Context uses preceding known feature vectors, not imputed target labels.
                history=self.context[-(self.sequence-1):]
                if not history:
                    self.reasons.add('Недостаточно контекста сети')
                    return None
                history=[history[0]]*(self.sequence-1-len(history))+history
                self.model.steps[-1][1].context_=self.model.steps[0][1].transform(np.asarray(history)).astype(np.float32)
            prediction=float(self.model.predict(np.asarray(vector).reshape(1,-1))[0])
            if np.isfinite(prediction):
                return prediction
            self.reasons.add('Модель вернула нечисловой прогноз')
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            self.reasons.add('Ошибка прогноза: '+str(exc)[:200])
        return None

    def diagnostics(self):
        return {'Модель восстановления':self.name,'Восстановлено моделью':self.model_filled,
                'Резервным способом':self.fallback_filled,'Обучений модели':self.fits,
                'Примеров при последнем обучении':self.samples,
                'Лимит обучений достигнут':self.attempts>=self.max_fits,
                'Причина резерва':'; '.join(sorted(self.reasons)), 'Предупреждение модели':self.fit_warning}
