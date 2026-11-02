"""Explanations never enter model selection. All tabular diagnostics use validation."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from ml_lab.neural import RecurrentRegressor


def permutation_scores(result):
    if isinstance(result['pipeline'].named_steps['model'],RecurrentRegressor):
        raise ValueError('Для рекуррентных сетей используйте Integrated Gradients: перестановка строк разрушает временные окна.')
    observed=result['explain_y'].notna().to_numpy()
    if not observed.any(): raise ValueError('Нет исходных измерений Y для оценки важности на validation.')
    scores=permutation_importance(result['pipeline'],result['explain_X'].iloc[observed],result['explain_y'].iloc[observed],n_repeats=5,
        random_state=result['params'].get('seed',42),scoring='neg_mean_absolute_error' if result['task']=='Регрессия' else 'f1_macro',n_jobs=1)
    return pd.DataFrame({'Признак':result['features'],'Важность':scores.importances_mean,'Разброс':scores.importances_std}).sort_values('Важность',ascending=False)


def shap_scores(result, count=12, class_index=0):
    import shap
    pipeline=result['pipeline']
    if isinstance(pipeline.named_steps['model'],RecurrentRegressor):
        raise ValueError('Для LSTM/GRU/RNN используется Integrated Gradients по временным окнам.')
    prep,model=pipeline.named_steps['preprocess'],pipeline.named_steps['model']
    background=np.asarray(prep.transform(result['explain_train']),dtype=float)
    values=np.asarray(prep.transform(result['explain_X'].head(count)),dtype=float)
    names=prep.get_feature_names_out().tolist()
    if len(names)>100:
        raise ValueError('После кодирования более 100 признаков. Уменьшите число категорий для SHAP.')
    label='Прогноз Y'
    if result['task']=='Регрессия':
        predict=model.predict
    elif hasattr(model,'predict_proba'):
        predict=lambda X: model.predict_proba(X)[:,class_index]
        label=f'Вероятность класса {model.classes_[class_index]}'
    elif hasattr(model,'decision_function'):
        def predict(X):
            scores=model.decision_function(X)
            if scores.ndim==1:
                return scores if class_index==1 else -scores
            return scores[:,class_index]
        label=f'Оценка decision_function класса {model.classes_[class_index]} (не вероятность)'
    else:
        raise ValueError('Эта модель не предоставляет непрерывную оценку класса для SHAP.')
    explainer=shap.PermutationExplainer(predict,background,feature_names=names,seed=result['params'].get('seed',42))
    explanation=explainer(values,max_evals=3*(2*len(names)+1),silent=True)
    contributions=np.asarray(explanation.values)
    return dict(names=names,values=contributions,bases=np.asarray(explanation.base_values),
                inputs=values,output=label,indices=result['explain_X'].head(count).index.tolist(),
                summary=pd.DataFrame({'Признак':names,'Важность':np.abs(contributions).mean(axis=0)}).sort_values('Важность',ascending=False))


def integrated_gradients(result, steps=24):
    import torch
    pipeline=result['pipeline']; model=pipeline.named_steps['model']
    if not isinstance(model,RecurrentRegressor): raise ValueError('Метод предназначен для рекуррентной сети.')
    prep=pipeline.named_steps['preprocess']
    X=np.asarray(prep.transform(result['explain_X'].head(24)),dtype=np.float32)
    # Independent copy of context; no state is changed by diagnostics.
    context=np.asarray(prep.transform(result['explain_context'].tail(model.window-1)),dtype=np.float32) if model.window>1 else X[:0]
    gaps=np.flatnonzero(np.diff(result.get('explain_context_positions',np.arange(len(result['explain_context']))))!=1)
    if len(gaps) and model.window>1:
        preceding=result['explain_context'].iloc[gaps[-1]+1:].tail(model.window-1)
        context=np.asarray(prep.transform(preceding),dtype=np.float32)
        context=np.concatenate([np.repeat(context[:1],model.window-1-len(context),axis=0),context])
    breaks=result.get('explain_breaks')
    windows=model._windows(X,context,breaks=breaks[:len(X)] if breaks is not None else None)
    total=torch.zeros_like(windows)
    model.rnn_.eval(); model.head_.eval()
    for alpha in np.linspace(0,1,steps):
        inputs=(windows*float(alpha)).detach().requires_grad_(True)
        output,_=model.rnn_(inputs)
        prediction=model.head_(output[:,-1]).sum()*model.y_scale_
        grad,=torch.autograd.grad(prediction,inputs)
        total+=grad.detach()
    contribution=(windows*total/steps).numpy()
    return pd.DataFrame({'Признак':prep.get_feature_names_out(),'Важность':np.abs(contribution).sum(axis=1).mean(axis=0)}).sort_values('Важность',ascending=False)
