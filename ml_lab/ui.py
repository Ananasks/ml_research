"""Declarative Streamlit controls matching the model catalogue."""
import copy
import json
import streamlit as st
from ml_lab.model_registry import MODEL_REGISTRY, RECURRENT_MODELS


def parameter_controls(name,task,mode,namespace='model'):
    parameters={}; space={}
    schema=MODEL_REGISTRY[name]['parameters']
    for key,s in schema.items():
        widget_key=f'{namespace}_{task}_{name}_{mode}_{key}'
        if s['type']=='layers':
            if mode=='Ручная настройка':
                count=st.slider('Скрытых слоёв',1,6,2,key=widget_key+'_count')
                columns=st.columns(min(count,3))
                parameters[key]=[columns[i%len(columns)].slider(f'Нейронов · слой {i+1}',8,256,64 if i==0 else 32,8,key=f'{widget_key}_{i}') for i in range(count)]
            else:
                space['layer_count']=dict(label='Скрытых слоёв',type='int',min=1,max=6,default=2,step=1)
                space['layer_units']=dict(label='Нейронов в слое',type='int',min=8,max=256,default=64,step=8)
            continue
        if mode=='Ручная настройка':
            if s['type']=='choice':
                parameters[key]=st.selectbox(s['label'],s['choices'],index=s['choices'].index(s['default']),key=widget_key)
            else:
                parameters[key]=st.slider(s['label'],s['min'],s['max'],s['default'],s['step'],key=widget_key,help='Логарифмическое распределение при автоматическом поиске.' if s.get('log') else None)
        else:
            space[key]=copy.deepcopy(s)
    if mode!='Ручная настройка':
        # Epoch/window budgets stay fixed, avoiding accidental multi-hour searches.
        for key in list(space):
            s=space[key]; widget_key=f'{namespace}_{task}_{name}_{mode}_{key}'
            if key in ('epochs','window','max_iter'):
                parameters[key]=st.slider(s['label']+' · лимит',s['min'],s['max'],s['default'],s['step'],key=widget_key)
                del space[key]
            elif s['type']=='choice':
                options=st.multiselect(s['label'],s['choices'],default=s['choices'],key=widget_key)
                if not options: st.error(f"Выберите хотя бы одно значение: {s['label']}")
                space[key]['choices']=options
            else:
                st.markdown(f"**{s['label']}**")
                low,high=st.slider('Диапазон',s['min'],s['max'],(s['min'],s['max']),s['step'],key=widget_key)
                space[key].update(min=low,max=high)
                if s.get('log'): st.caption('Логарифмическая шкала поиска')
    if name in RECURRENT_MODELS:
        st.caption('Последовательности строк → рекуррентные слои → Dense → прогноз. Ранняя остановка: 6 эпох без улучшения внутренней валидации.')
    return parameters,space


def json_table(candidates):
    return [{'Вариант':i+1,'Модель':c['name'],'Параметры':json.dumps(c['params'],ensure_ascii=False)} for i,c in enumerate(candidates)]
