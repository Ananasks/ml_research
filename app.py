"""ML Research Lab: local research workflow over the shared ML engine."""
import hashlib
import io
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from ml_lab.core import run_experiments, generate_candidates, split_data, eligible_rows
from ml_lab.data import prepare_frame, detect_time_columns, read_dataset
from ml_lab.dataset_ui import dataset_controls
from ml_lab.model_registry import (MODELS, RECURRENT_MODELS, MODEL_REGISTRY, available_models,
    catalog_rows, documentation, online_catalog_status)
from ml_lab.ui import parameter_controls, json_table
from ml_lab.chart_style import show_chart
from ml_lab.result_notes import show_result_chart
from ml_lab.experiments import ExperimentStore, html_report
from ml_lab.result_export import results_bundle
from ml_lab.model_theory import render_model_descriptions
from ml_lab.workflow_canvas import workflow_canvas
from ml_lab.explain import permutation_scores, shap_scores, integrated_gradients
from ml_lab.diagnostics import (regression_figures, classification_figures, probability_figures,
    daily_figure, timeline_figure, comparison_figures, training_figure, importance_figure, shap_local_figure)
from ml_lab.time_index import parse_datetime_values, normalize_time_index

@st.cache_data(show_spinner=False,max_entries=8)
def cached_prepare_frame(raw, preparation):
    return prepare_frame(raw,**preparation)


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='ML Research Lab', page_icon='🧪', layout='wide')
st.title('🧪 ML Research Lab')
st.caption('Данные → признаки → модели → честная оценка → исследовательский отчёт')
st.session_state.setdefault('history', [])
store = ExperimentStore(ROOT / '.local' / 'experiments.sqlite3')
navigation_sections = {
    'Эксперимент': {
        'Схема эксперимента': 'workflow', 'Данные и признаки': 'data',
        'Модель и обучение': 'model',
    },
    'Результаты': {
        'Качество и прогноз': 'results_quality', 'Обучение и метрики': 'results_learning',
        'Сравнение моделей': 'results_comparison', 'Объяснимость': 'results_explain',
        'Экспорт': 'results_exports',
    },
    'Библиотека': {
        'Описание моделей': 'descriptions', 'Каталог и источники': 'catalog',
        'Журнал': 'journal',
    },
}
st.sidebar.markdown('### :material/science: ML Research Lab')
st.sidebar.caption('Рабочее пространство исследования')
selected_section = st.sidebar.selectbox(
    'Раздел', list(navigation_sections), key='navigation_section',
)
section_pages = navigation_sections[selected_section]
selected_page = st.sidebar.radio(
    'Страница', list(section_pages), key=f"navigation_page_{selected_section}",
)
navigation_pages = {label: page for section in navigation_sections.values() for label, page in section.items()}
active_page_key = navigation_pages[selected_page]
page_visibility = ''.join(f'.st-key-page_{page_key} {{display:{"block" if page_key == active_page_key else "none"} !important;}}\n'
    for page_key in navigation_pages.values() if not page_key.startswith('results_'))
page_visibility += f'.st-key-page_results {{display:{"block" if active_page_key.startswith("results_") else "none"} !important;}}\n'
page_visibility += ''.join(f'.st-key-page_{page_key} {{display:{"block" if page_key == active_page_key else "none"} !important;}}\n'
    for page_key in navigation_pages.values() if page_key.startswith('results_'))
global_ui_style = """
[data-testid="stAppViewContainer"] .main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
[data-testid="stAppViewContainer"] .main h1 { margin-bottom: .45rem; }
[data-testid="stAppViewContainer"] .main h3 { margin-top: .8rem; }
[data-testid="stSidebar"] { min-width: 15rem; max-width: 15rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { font-size: .9rem; line-height: 1.35; }
[data-testid="stWidgetLabel"] p { font-size: .88rem; line-height: 1.3; }
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] { min-width: 0; }
"""
st.markdown(f'<style>{global_ui_style}{page_visibility}</style>', unsafe_allow_html=True)
with st.container(key='page_descriptions'):
    render_model_descriptions()
with st.container(key='page_journal'):
    if st.session_state.get('journal_clear_message'): st.success(st.session_state.pop('journal_clear_message'))
    st.caption('Локальный журнал SQLite сохраняется между перезапусками. Метрики разных данных или разбиений напрямую несопоставимы.')
    try:
        records=store.list()
        if records:
            st.dataframe(pd.DataFrame([dict(Время=r['created'],Модель=r['model'],Dataset=r['config']['dataset'],Цель=r['target'],**r['metrics']) for r in records]),width='stretch')
            st.download_button('Журнал JSON',json.dumps(records,ensure_ascii=False,indent=2),'history.json','application/json')
            st.caption('Очистка удаляет записи всех экспериментов из локального журнала. Текущий результат и библиотека датасетов сохраняются.')
            if st.button('Очистить журнал',key='clear_experiment_journal'):
                removed=store.clear()
                st.session_state.history=[]
                st.session_state.journal_clear_message=f'Удалено записей: {removed}.'
                st.rerun()
        else: st.info('Здесь появятся завершённые эксперименты.')
    except Exception as exc: st.error(f'Не удалось прочитать журнал: {exc}')

with st.container(key='page_catalog'):
    st.subheader('Полный каталог моделей и онлайн-источники')
    catalog_task=st.session_state.get('cfg_task','Регрессия')
    catalog_model=st.session_state.get('chosen_model',MODELS[catalog_task][0])
    if catalog_model not in MODELS[catalog_task]: catalog_model=MODELS[catalog_task][0]
    catalog_spec=MODEL_REGISTRY[catalog_model]
    st.caption(f"Выбрана модель: {catalog_model} · категория: {catalog_spec['group']} · библиотека: {catalog_spec['package']}")
    st.link_button('Официальная документация модели',documentation(catalog_model,catalog_task))
    st.dataframe(pd.DataFrame(catalog_rows(catalog_task)),width='stretch',hide_index=True,
        column_config={'Документация':st.column_config.LinkColumn('Документация')})
    st.caption('Показаны проверенные адаптеры установленных библиотек. Новые пакеты доступны после установки и перезапуска приложения. Сведения о версиях загружаются с PyPI по кнопке.')
    st.code('python -m pip install -r requirements-models.txt',language='bash')
    if st.button('Проверить версии библиотек онлайн'):
        with st.spinner('Проверка официальных метаданных PyPI…'):
            st.session_state.model_online_status=online_catalog_status(include_extensions=True)
    if st.session_state.get('model_online_status'):
        st.dataframe(st.session_state.model_online_status,width='stretch',hide_index=True)

with st.container(key='page_data'):
    raw, dataset_name, dataset_id = dataset_controls(ROOT)
    if raw is None: st.stop()
    if st.session_state.get('dataset_id') != dataset_id:
        for key in list(st.session_state):
            if key.startswith('feature_') or key in ('result','candidates','candidate_signature'):
                del st.session_state[key]
        st.session_state.pop('workflow_repair_override', None)
        st.session_state.pop('workflow_target_override', None)
        st.session_state.dataset_id = dataset_id
    with st.container(border=True):
        st.subheader('Целевая колонка и задача')
        st.caption('Выберите показатель Y, который модель должна прогнозировать. Эта цель используется для обучения и оценки результатов.')
        a,b = st.columns(2)
        targets=[c for c in raw.columns if c not in detect_time_columns(raw)]
        target = a.selectbox('Целевая колонка (Y)', targets,
            index=targets.index(next((c for c in ['CB','MUF','y'] if c in targets),targets[-1])),
            key='cfg_target', help='datetime используется как индекс и не выбирается в качестве цели.')
        target = st.session_state.get('workflow_target_override', target)
        task = b.radio('Тип задачи', ['Регрессия','Классификация'], horizontal=True,key='cfg_task')
    generated_index_settings={}
    time_ready = bool(detect_time_columns(raw)) or isinstance(raw.index,pd.DatetimeIndex)
    if not time_ready:
        st.info('Укажите datetime, если это временной ряд. Для обычных таблиц можно оставить индекс строк и использовать случайное разделение выборки.')
        start_date = st.date_input('Начальная дата измерений')
        start_time = st.time_input('Начальное время измерений')
        index_step = st.selectbox('Шаг исходных измерений', ['5min','15min','30min','1h','1D'],index=2)
        time_ready = st.checkbox('Использовать указанные начало и шаг для создания datetime')
        if time_ready:
            raw=raw.copy()
            generated_index_settings=dict(start_datetime=str(pd.Timestamp.combine(start_date,start_time)),index_frequency=index_step)
            raw['datetime']=pd.date_range(pd.Timestamp.combine(start_date,start_time),periods=len(raw),freq=index_step)
    if time_ready and not detect_time_columns(raw) and isinstance(raw.index,pd.DatetimeIndex):
        raw=raw.rename_axis('datetime').reset_index()
    if time_ready:
        try:
            original,original_times,diagnostics=normalize_time_index(raw,drop_invalid_time=True)
            original.index=pd.DatetimeIndex(original_times,name='datetime')
            st.dataframe(original.head(100),width='stretch')
        except ValueError as exc:
            st.error(str(exc)); st.stop()
    else:
        st.dataframe(raw.head(100),width='stretch')
    has_time = bool(detect_time_columns(raw)) or isinstance(raw.index,pd.DatetimeIndex)
    st.subheader('Исходный график')
    plot_columns = [c for c in targets if pd.to_numeric(raw[c], errors='coerce').notna().any()]
    if plot_columns:
        selected_series = st.multiselect('Показатели на исходном графике', plot_columns,
            default=[target] if target in plot_columns else plot_columns[:1],
            key='source_series_'+dataset_id+'_'+target)
        if selected_series:
            if time_ready:
                source_values=original[selected_series].apply(pd.to_numeric,errors='coerce')
            else:
                source_values=pd.DataFrame(columns=selected_series,index=pd.DatetimeIndex([],name='datetime'))
                st.info('Исходный график появится после задания datetime.')
            fig = px.line(source_values, y=selected_series,
                labels={'value':'Исходное значение', 'variable':'Показатель'}, title='Исходные измерения')
            fig.update_traces(connectgaps=False)
            show_chart(fig, key='source_chart')
            st.caption('Данные до заполнения пропусков, создания признаков и сдвига цели. Масштабируйте график мышью; показатели переключаются в легенде.')
        else:
            st.info('Выберите хотя бы один показатель для графика.')
    else:
        st.info('В исходном наборе нет числовых показателей для графика.')
    if dataset_name=='muf.csv' and st.button('Настройки для MUF'):
        st.session_state.preset_muf=True
        st.session_state.chosen_model='Hist Gradient Boosting'
    preset=st.session_state.get('preset_muf',False) and dataset_name=='muf.csv'
    grid=st.checkbox('Единая сетка времени',value=preset,disabled=not has_time)
    grid_frequency=st.selectbox('Шаг единой сетки времени',['5min','15min','30min','1h','1D'],index=2,disabled=not(grid and has_time))
    calendar = st.checkbox('Календарные признаки', value=True, disabled=not has_time)
    numeric_sources=[c for c in raw.select_dtypes('number') if c not in detect_time_columns(raw) and c.lower() not in ('index','id')]
    if target not in numeric_sources and pd.to_numeric(raw[target],errors='coerce').notna().any(): numeric_sources.append(target)
    repair_config=None
    with st.expander('Восстановление пропусков и аномалий',expanded=False):
        repair_enabled=st.checkbox('Восстанавливать пропуски и корректировать аномалии',value=True,disabled=not has_time)
        repair_candidates=[c for c in numeric_sources if task=='Регрессия' or c!=target]
        repair_columns=st.multiselect('Колонки восстановления',repair_candidates,default=repair_candidates,key='repair_columns_'+dataset_id+'_'+task)
        a,b=st.columns(2)
        fill_label=a.selectbox('Восстановление значений',['Линейная экстраполяция по прошлому','Предыдущее значение','Без восстановления'])
        anomaly_label=b.selectbox('Поиск аномалий',['IQR по прошлому окну','MAD по прошлому окну','Z-score по прошлому окну','Только физические границы'])
        a,b,c=st.columns(3)
        repair_window=int(a.number_input('Окно поиска аномалий · строк',8,1000,48))
        min_history=int(b.number_input('Минимальная история для поиска',2,repair_window,min(8,repair_window)))
        threshold=float(c.number_input('Порог аномалии',.1,20.,3.,.1))
        action=st.radio('Коррекция аномалий',['Ограничить границами','Заменить и восстановить'],horizontal=True)
        max_gap=int(st.number_input('Максимум подряд восстанавливаемых строк · 0 без ограничения',0,100000,0))
        st.caption('Физические границы необязательны и задаются для конкретных величин. Пустая ячейка — без ограничения.')
        bound_table=st.data_editor(pd.DataFrame({'Колонка':repair_columns,'Минимум':[float('nan')]*len(repair_columns),'Максимум':[float('nan')]*len(repair_columns)}),
            disabled=['Колонка'],hide_index=True,key='repair_bounds_'+dataset_id+'_'+task)
        bounds={row['Колонка']:{key:float(row[label]) for key,label in [('min','Минимум'),('max','Максимум')] if pd.notna(row[label])} for _,row in bound_table.iterrows()}
        scaler_label=st.selectbox('Масштабирование числовых X',['StandardScaler','RobustScaler','MinMaxScaler','Без масштабирования'])
        quality_flags=st.checkbox('Добавлять признаки пропусков и аномалий',value=True)
        if repair_enabled and has_time and repair_columns:
            repair_config=dict(columns=repair_columns,fill_method={'Линейная экстраполяция по прошлому':'linear','Предыдущее значение':'forward','Без восстановления':'none'}[fill_label],
                anomaly_method={'IQR по прошлому окну':'iqr','MAD по прошлому окну':'mad','Z-score по прошлому окну':'zscore','Только физические границы':'none'}[anomaly_label],
                anomaly_action='clip' if action=='Ограничить границами' else 'replace',window=repair_window,min_history=min_history,threshold=threshold,max_gap=max_gap,bounds=bounds)
        st.caption('Расчёт использует только прошлые наблюдения. Начальные пропуски без истории остаются неизвестными; X дополнительно заполняются статистиками train. '
            'Длинная экстраполяция — предположение о продолжении тренда. Восстановленные Y могут участвовать в train, но не считаются измеренными фактами для метрик. Категории целевой классификации не интерполируются.')
    with st.expander('Признаки истории и горизонт'):
        feature_sources=[c for c in numeric_sources if task=='Регрессия' or c!=target]
        selected_sources=st.multiselect('Колонки для признаков истории',feature_sources,default=[target] if target in feature_sources else feature_sources[:1],key='history_sources_'+dataset_id+'_'+task)
        lags = st.multiselect('Лаги выбранных колонок · строк', [1,2,3,6,12,24,48,96,168,288,336], default=[1,2,3] if preset else [], disabled=not has_time)
        extra_lags=st.text_input('Дополнительные лаги · через запятую',help='Например: 1440, 2016. Лаги задаются в строках.')
        try:
            lags=sorted(set(lags)|{int(value.strip()) for value in extra_lags.split(',') if value.strip()})
            if any(lag<1 for lag in lags): raise ValueError
        except ValueError:
            st.error('Укажите положительные целые лаги через запятую.');st.stop()
        rolling = st.multiselect('Скользящие окна · только прошлые значения', [3,6,12,24,48,96,168,288], default=[3,12] if preset else [], disabled=not has_time)
        rolling_stats=st.multiselect('Статистики скользящего окна',['mean','std','min','max','median','iqr','trend'],default=['mean','std'])
        ewm=st.multiselect('Экспоненциальное среднее · span',[3,6,12,24,48,96],disabled=not has_time)
        difference_lags=st.multiselect('Разности прошлых значений · лаг',[1,2,6,12,24,48,96,288],disabled=not has_time)
        partial_windows=st.checkbox('Использовать неполное окно в начале',value=False)
        horizon = st.number_input('Горизонт · строк', 0, 1000, 0, disabled=not has_time)
        st.caption('Все признаки истории сдвинуты в прошлое минимум на один шаг и доступны для выбранных числовых колонок. '
            'trend — среднее изменение между концами прошлого окна. Первые строки без истории не подменяются будущими значениями. Лаги Y предполагают пошаговое получение фактических наблюдений.')
    preparation = dict(calendar=calendar, target=target, lags=lags, rolling_windows=rolling,rolling_stats=rolling_stats,
        horizon=int(horizon), drop_invalid_time=True, frequency=grid_frequency if grid else None,repair=repair_config,
        feature_columns=selected_sources,ewm_spans=ewm,difference_lags=difference_lags,partial_windows=partial_windows,quality_flags=quality_flags,**generated_index_settings)
    if st.session_state.get('workflow_repair_override') is not None:
        preparation['repair'] = st.session_state.workflow_repair_override
    try: frame = cached_prepare_frame(raw,preparation)
    except Exception as exc:
        st.error(str(exc)); st.stop()
    if frame.attrs.get('repair_records'):
        with st.expander('Результат восстановления · до и после'):
            st.dataframe(pd.DataFrame(frame.attrs['repair_records']),width='stretch',hide_index=True)
            repaired_column=st.selectbox('Колонка для проверки восстановления',list(frame.attrs['repair_originals']))
            before=frame.attrs['repair_originals'][repaired_column].to_numpy()
            after=frame.attrs['repair_values'][repaired_column].to_numpy()
            repair_plot=pd.DataFrame({'Исходные значения':before,'После восстановления':after},index=frame.index)
            figure=px.line(repair_plot,x=repair_plot.index,y=repair_plot.columns,title='Восстановление · '+repaired_column)
            anomalous=frame.attrs['repair_masks'][repaired_column]['anomaly']
            figure.add_scatter(x=frame.index[anomalous],y=before[anomalous],mode='markers',name='Обнаруженные аномалии',marker=dict(color='#dc2626',size=7))
            show_chart(figure,key='repair_preview')
            st.download_button('Подготовленные данные CSV',frame.to_csv(index=True).encode('utf-8-sig'),'prepared_data.csv','text/csv')
    if frame.attrs.get('invalid_time_rows'): st.warning(f"Исключено некорректных дат: {frame.attrs['invalid_time_rows']}")
    excluded = set(detect_time_columns(raw)) | {target}
    options = [c for c in frame.columns if c not in excluded]
    st.subheader('Входные признаки X')
    a,b = st.columns(2)
    all_on = a.button('Выбрать все')
    all_off = b.button('Снять все')
    columns = st.columns(3)
    features = []
    for i,c in enumerate(options):
        key = 'feature_'+target+'_'+c
        if all_on or all_off: st.session_state[key] = all_on
        # Exact copies of the target are leakage, never selected by default.
        default = not frame[c].equals(frame[target])
        st.session_state.setdefault(key,default)
        if columns[i%3].checkbox(c, key=key): features.append(c)
    with st.expander('Профиль и статистика'):
        st.dataframe(pd.DataFrame({'Тип':frame.dtypes.astype(str),'Пропуски':frame.isna().sum(),'Уникальных':frame.nunique()}),width='stretch')
        numeric = frame.select_dtypes('number')
        if len(numeric.columns)>1:
            show_chart(px.imshow(numeric.corr(),zmin=-1,zmax=1,color_continuous_scale='RdBu_r',title='Корреляции'),key='data_correlation')
        if features: show_chart(px.histogram(frame,x=features[0],title='Распределение признака'),key='data_distribution')
    available=len(eligible_rows(frame,features,target,task)) if features else len(frame)
    st.subheader('Объём выборок')
    split_label = st.radio('Разделение', (['По времени'] if repair_config else ['По времени','Случайное']) if has_time else ['Случайное'], horizontal=True)
    params = dict(split='time' if split_label=='По времени' else 'random',scaler={'StandardScaler':'standard','RobustScaler':'robust','MinMaxScaler':'minmax','Без масштабирования':'none'}[scaler_label],missing_indicators=True)
    size_mode = st.radio('Размеры выборок', ['Количество строк','Проценты'], horizontal=True)
    if size_mode=='Количество строк':
        a,b,c = st.columns(3)
        params.update(size_mode='rows',train_rows=a.number_input('Строк для обучения',12,None,max(12,int(available*.6))),
            validation_rows=b.number_input('Строк для валидации',2,None,max(2,int(available*.2))),
            test_rows=c.number_input('Строк для теста',1,None,max(1,int(available*.2))))
    else:
        a,b = st.columns(2)
        params.update(test_size=a.number_input('Тест · доля',.05,.45,.2,.05),validation_size=b.number_input('Валидация · доля',.05,.45,.2,.05))
    if has_time and params['split']=='time':
        period = st.radio('Период прогноза',['Последняя часть набора','Один день','Диапазон дней'],horizontal=True)
        if period!='Последняя часть набора':
            times=frame.attrs['time']; day=times.iloc[int(len(times)*.8)].date()
            start=st.date_input('Дата прогноза',day)
            end=st.date_input('Конец периода',start) if period=='Диапазон дней' else start
            if params.get('size_mode')=='rows':
                st.caption('Для календарного периода тест задаётся датами; предшествующие строки делятся на обучение и валидацию в отношении 3:1.')
                params.pop('size_mode')
                params.update(test_size=.2,validation_size=.2)
            params.update(forecast_start=str(start),forecast_end=str(pd.Timestamp(end)+pd.Timedelta(days=1)-pd.Timedelta(nanoseconds=1)))
    params['seed']=int(st.number_input('Random seed',0,999999,42))
    show_unused=st.checkbox('Показывать количество незадействованных строк',value=True,key='show_unused_rows')
    try:
        _,_,indices=split_data(frame,features,target,task,params)
        st.caption('Фактические размеры: '+ ' · '.join(f'{name}: {len(idx)}' for name,idx in zip(['train','validation','test'],indices)))
        if show_unused:
            used=sum(len(idx) for idx in indices)
            a,b,c=st.columns(3)
            a.metric('Незадействованных строк',len(frame)-used)
            b.metric('Доступно вне выборок',available-used)
            c.metric('Не подходят для обучения',len(frame)-available)
            st.caption('Подсчёт по подготовленному набору: незадействованные = строки вне train/validation/test + строки с оставшейся неизвестной целью или недоступной историей. '
                'Строки вне выборок включают зазоры для горизонта и наблюдения вне выбранного периода. '
                'Пропуски обычных X восстанавливаются по прошлому, а оставшиеся заполняются внутри модели статистиками train и сами по себе не исключают строки.')
            if frame.attrs.get('invalid_time_rows'):
                st.caption(f"До формирования набора отдельно исключено строк с некорректным datetime: {frame.attrs['invalid_time_rows']}.")
    except ValueError as exc: st.info(str(exc))

with st.container(key='page_model'):
    if st.session_state.get('training_notice'): st.success(st.session_state.pop('training_notice'))
    available=available_models(task,params['split'])
    categories=['Все категории']+sorted({MODEL_REGISTRY[name]['group'] for name in available})
    left,right=st.columns([1,2])
    category=left.selectbox('Категория модели',categories,key='model_category_'+task)
    query=right.text_input('Поиск модели',placeholder='Название, категория или библиотека',key='model_search_'+task)
    names=available_models(task,params['split'],category,query)
    if not names:
        st.info('По заданным фильтрам моделей нет. Измените категорию или поисковый запрос.')
        st.stop()
    if st.session_state.get('chosen_model') not in names: st.session_state.chosen_model=names[0]
    model_name=st.selectbox('Модель',names,key='chosen_model')
    spec=MODEL_REGISTRY[model_name]
    if spec.get('max_rows'):
        st.caption(f"Для {model_name} поддерживается не более {spec['max_rows']} строк train из-за затрат памяти. Размеры выборок задаются в разделе «Данные и признаки».")
    mode=st.radio('Режим настройки',['Ручная настройка','Исследование диапазонов','Автоматическая оптимизация'],horizontal=True)
    model_params, space=parameter_controls(model_name,task,mode)
    params.update(model_params)
    count=st.number_input('Количество вариантов',1,100,5) if mode!='Ручная настройка' else 1
    config=dict(index=dict(name='datetime',start=str(frame.index.min()) if time_ready else None,end=str(frame.index.max()) if time_ready else None),dataset=dataset_name,dataset_id=dataset_id,features=features,target=target,task=task,preparation=preparation,params=params,model=model_name,mode=mode,space=space,count=count)
    signature=hashlib.sha256(json.dumps(config,sort_keys=True,default=str).encode()).hexdigest()
    if mode=='Исследование диапазонов':
        if st.button('✧ Сгенерировать модели'):
            st.session_state.candidates=generate_candidates(task,[model_name],int(count),params,{model_name:space|{'__fixed__':model_params}})
            st.session_state.candidate_signature=signature
        if st.session_state.get('candidates'): st.dataframe(json_table(st.session_state.candidates),width='stretch')
    stale_candidates=mode=='Исследование диапазонов' and st.session_state.get('candidate_signature')!=signature
    if st.button('▶ Обучить и оценить' if mode=='Ручная настройка' else '▶ Запустить поиск',type='primary',disabled=(not time_ready and params['split']=='time') or not features or stale_candidates):
        try:
            progress=st.progress(0)
            candidates=st.session_state.get('candidates',[]) if mode=='Исследование диапазонов' else [{'name':model_name,'params':params}]
            search=dict(count=int(count),families=[model_name],spaces={model_name:space|{'__fixed__':model_params}}) if mode=='Автоматическая оптимизация' else None
            result=run_experiments(frame,features,target,task,candidates,params,progress=lambda i,n,name:progress.progress(i/n,text=name),search=search)
            st.session_state.pop('results_export',None)
            st.session_state.result=result
            result.update(config=config,signature=signature,settings=preparation | {'time_columns':detect_time_columns(raw)}, raw_history=raw.copy())
            if 'error' not in result:
                st.session_state.history.append(dict(Модель=result['name'],**result['metrics']))
                try: store.save(result,config)
                except Exception as exc: st.warning(f'Модель обучена; история не сохранена: {exc}')
                st.session_state.training_notice='Обучение завершено. Откройте результаты.'
                st.rerun()
        except Exception as exc: st.error(f'Ошибка обучения: {exc}')
    with st.expander('Конфигурация эксперимента'): st.json(config)

with st.container(key='page_workflow'):
    st.subheader('Рабочее пространство эксперимента')
    st.caption('Визуальная схема обработки, модели и исследовательских графиков. Выберите узел на холсте, чтобы открыть его настройки.')
    if st.session_state.get('workflow_panel_size_version') != 2:
        st.session_state.pop('workflow_library_width', None)
        st.session_state.pop('workflow_inspector_width', None)
        st.session_state.workflow_panel_size_version = 2
    toolbar = st.columns([2.6, 1, 0.9], vertical_alignment='center')
    toolbar[0].caption(f'{dataset_name}  ·  {len(raw):,} строк  ·  цель {target}  ·  {task}')
    with toolbar[0].popover('Настроить панели'):
        library_width = st.slider('Ширина библиотеки',0.8,2.2,1.45,0.1,key='workflow_library_width')
        inspector_width = st.slider('Ширина настроек',0.8,2.2,1.75,0.1,key='workflow_inspector_width')
    run_action = toolbar[1].empty()
    save_action = toolbar[2].empty()
    block_specs = {
        'Загрузка данных': ('📄', 'Источник таблицы', 'Передаёт загруженный набор следующим блокам.'),
        'Очистка данных': ('🧹', 'Пропуски и аномалии', 'Использует настройки восстановления из «Данные и признаки».'),
        'Инженерия признаков': ('🛠️', 'Календарь и история', 'Создаёт выбранные лаги, окна и календарные признаки.'),
        'Выбор X и Y': ('🔎', 'Цель и входы', 'Определяет, что модель прогнозирует и какие X получает.'),
        'Разбиение выборки': ('✂️', 'Train / validation / test', 'Отделяет финальный тест от подбора модели.'),
        'Предобработка': ('🧽', 'Импутация, кодирование, масштабирование', 'Обучает преобразования только на train.'),
        'Стандартизация': ('⚖️', 'Масштабирование числовых признаков', 'Сравнивает исходный масштаб с преобразованием, обученным на train.'),
        'Обучение модели': ('🧠', 'Алгоритм', 'Обучает выбранный классический алгоритм или нейросеть.'),
        'Оценка модели': ('📊', 'Метрики', 'Выбирает модель по validation и оценивает её на test.'),
        'Анализ результатов': ('📈', 'Интерпретация', 'Сводит метрики и сравнение кандидатов; подробные графики доступны во вкладке «Результаты».'),
        'Просмотр данных': ('📋', 'Таблица', 'Показывает подготовленные строки и признаки.'),
        'Экспорт данных': ('⬇️', 'CSV', 'Скачивает подготовленную таблицу.'),
        'Экспорт модели': ('📦', 'JOBLIB', 'Скачивает обученный pipeline и его конфигурацию.'),
        'Профиль данных': ('🧾', 'Профиль данных', 'Типы, пропуски, уникальные значения и описательные статистики.'),
        'Проверка пропусков': ('🕳️', 'Пропуски', 'Сводка пропусков до и после подготовки данных.'),
        'Корреляции': ('🧩', 'Корреляции', 'Корреляционная матрица числовых признаков.'),
        'Распределение': ('📊', 'Распределение', 'Гистограмма выбранного признака с разбиением по классу.'),
        'Диаграмма рассеяния': ('🔬', 'Связь признаков', 'Интерактивная диаграмма X против Y.'),
        'Ящик с усами': ('📦', 'Размах и выбросы', 'Показывает медиану, квартили и отдельные наблюдения.'),
        'Проверка разбиения': ('📐', 'Размеры частей', 'Показывает диапазоны времени и объёмы train/validation/test.'),
        'Диагностика ошибок': ('🩺', 'Ошибки модели', 'Остатки, накопленная ошибка и матрицы ошибок.'),
        'Важность признаков': ('🎚️', 'Permutation importance', 'Оценивает вклад перестановкой признака на validation.'),
        'SHAP объяснение': ('🧮', 'SHAP', 'Глобальные и локальные вклады признаков на validation.'),
        'Integrated Gradients': ('🧠', 'Integrated Gradients', 'Вклады признаков и временных шагов для LSTM/GRU/RNN.'),
        'Экспорт отчёта': ('📝', 'HTML-отчёт', 'Скачивает автономный отчёт по завершённому эксперименту.'),
        'Экспорт архива': ('🗜️', 'Архив эксперимента', 'Собирает графики, прогнозы, конфигурацию и модель в ZIP.'),
    }
    block_groups = {
        'Данные и исследование': ['Загрузка данных', 'Просмотр данных', 'Профиль данных', 'Проверка пропусков', 'Корреляции', 'Распределение', 'Диаграмма рассеяния', 'Ящик с усами', 'Экспорт данных'],
        'Подготовка': ['Очистка данных', 'Инженерия признаков', 'Выбор X и Y', 'Разбиение выборки', 'Предобработка', 'Стандартизация'],
        'Модели': ['Обучение модели'],
        'Оценка и объяснимость': ['Проверка разбиения', 'Оценка модели', 'Анализ результатов', 'Диагностика ошибок', 'Важность признаков', 'SHAP объяснение', 'Integrated Gradients'],
        'Экспорт и воспроизводимость': ['Экспорт отчёта', 'Экспорт архива', 'Экспорт модели'],
    }
    block_colors = {
        'Загрузка данных':('#d7ebfb','#4c91c9'), 'Очистка данных':('#fce0c5','#e59a45'),
        'Инженерия признаков':('#d9eed9','#61a86a'), 'Выбор X и Y':('#d7ebfb','#4c91c9'),
        'Разбиение выборки':('#d9eed9','#61a86a'), 'Предобработка':('#fce0c5','#e59a45'),
        'Обучение модели':('#d7ebfb','#4c91c9'), 'Оценка модели':('#f8dadd','#d5747c'),
        'Стандартизация':('#fce0c5','#e59a45'),
        'Анализ результатов':('#f8dadd','#d5747c'), 'Просмотр данных':('#f8dadd','#d5747c'),
        'Экспорт данных':('#d7ebfb','#4c91c9'), 'Экспорт модели':('#f8dadd','#d5747c'),
        'Профиль данных':('#d7ebfb','#4c91c9'), 'Проверка пропусков':('#fce0c5','#e59a45'),
        'Корреляции':('#d9eed9','#61a86a'), 'Распределение':('#f8dadd','#d5747c'),
        'Диаграмма рассеяния':('#f8dadd','#d5747c'), 'Ящик с усами':('#f8dadd','#d5747c'), 'Проверка разбиения':('#d9eed9','#61a86a'),
        'Диагностика ошибок':('#f8dadd','#d5747c'), 'Важность признаков':('#f8dadd','#d5747c'),
        'SHAP объяснение':('#f8dadd','#d5747c'), 'Integrated Gradients':('#d7ebfb','#4c91c9'),
        'Экспорт отчёта':('#f8dadd','#d5747c'), 'Экспорт архива':('#f8dadd','#d5747c'),
    }
    required_blocks = ['Загрузка данных', 'Очистка данных', 'Инженерия признаков', 'Выбор X и Y',
                       'Разбиение выборки', 'Предобработка', 'Обучение модели', 'Оценка модели', 'Анализ результатов']
    if 'workflow_nodes' not in st.session_state:
        st.session_state.workflow_nodes = [dict(id=f'node_{i+1}', type=block) for i, block in enumerate(required_blocks)]
        st.session_state.workflow_edges = [(f'node_{i+1}', f'node_{i+2}') for i in range(len(required_blocks)-1)]
    st.session_state.setdefault('workflow_canvas_instance', 0)
    st.session_state.setdefault('workflow_scaler_value', st.session_state.get('workflow_scaler', params.get('scaler','standard')))
    nodes = st.session_state.workflow_nodes
    edges = st.session_state.workflow_edges
    if st.session_state.get('workflow_layout_version') != 3:
        for index, node in enumerate(nodes):
            column = index % 3
            row = index // 3
            if row % 2:
                column = 2 - column
            node['x'] = 35 + column * 200
            node['y'] = 32 + row * 195
        st.session_state.workflow_layout_version = 3
    if not any(node['id'] == st.session_state.get('workflow_canvas_selected_node') for node in nodes):
        model_node = next((node['id'] for node in nodes if node['type'] == 'Обучение модели'), None)
        if model_node:
            st.session_state.workflow_canvas_selected_node = model_node
    left, canvas, inspector = st.columns([library_width, 4.2, inspector_width], gap='small')
    with left:
        st.markdown('#### Библиотека инструментов')
        st.caption('Выберите виджет, чтобы добавить его на холст.')
        def append_workflow_block(block_type):
            max_existing = max((int(node['id'].rsplit('_',1)[-1]) for node in nodes), default=0)
            next_id = max(max_existing + 1, int(st.session_state.get('workflow_next_id', 1)))
            node_id = f'node_{next_id}'
            st.session_state.workflow_next_id = next_id + 1
            nodes.append(dict(id=node_id, type=block_type))
            st.session_state.workflow_canvas_selected_node = node_id
        tool_query=st.text_input('Найти инструмент',placeholder='Например, SHAP или распределение',key='workflow_tool_search').strip().casefold()
        favorites = st.session_state.setdefault('workflow_favorites', [])
        favorites_only = st.checkbox('Только избранное',key='workflow_favorites_only')
        for group_name, group_blocks in block_groups.items():
            matching_blocks=[block_type for block_type in group_blocks if not tool_query or tool_query in ' '.join((block_type,*block_specs[block_type])).casefold()]
            if favorites_only: matching_blocks=[block_type for block_type in matching_blocks if block_type in favorites]
            if matching_blocks:
                with st.expander(group_name, expanded=bool(tool_query) or group_name in ('Данные и исследование','Модели')):
                    for block_type in matching_blocks:
                        icon, title, help_text = block_specs[block_type]
                        add_col, favorite_col = st.columns([0.86,0.14])
                        if add_col.button(f'{icon}  {title}',key=f'workflow_add_{block_type}',width='stretch',wrap=True,help=help_text):
                            append_workflow_block(block_type)
                            st.rerun()
                        if favorite_col.button('★' if block_type in favorites else '☆',key=f'workflow_favorite_{block_type}',help='Убрать из избранного' if block_type in favorites else 'Добавить в избранное'):
                            st.session_state.workflow_favorites = [item for item in favorites if item != block_type] if block_type in favorites else favorites + [block_type]
                            st.rerun()
        if tool_query and not any(tool_query in ' '.join((block,*block_specs[block])).casefold() for group in block_groups.values() for block in group):
            st.info('Инструмент не найден.')
        st.divider()
        if nodes:
            st.caption('Соединяйте блоки на холсте перетаскиванием выходного порта на входной. Щёлкните по линии, чтобы удалить связь.')
            selected_id = st.session_state.get('workflow_canvas_selected_node')
            selected_for_copy = next((node for node in nodes if node['id'] == selected_id), None)
            if st.button('⧉ Копировать выбранный блок', key='workflow_copy_node', width='stretch', disabled=selected_for_copy is None):
                max_existing = max((int(node['id'].rsplit('_',1)[-1]) for node in nodes), default=0)
                node_id = f'node_{max_existing + 1}'
                duplicate = selected_for_copy.copy()
                duplicate.update(id=node_id, x=(duplicate.get('x') or 45) + 35, y=(duplicate.get('y') or 160) + 35)
                nodes.append(duplicate)
                st.session_state.workflow_canvas_selected_node = node_id
                st.rerun()
            if st.button('Удалить выбранный блок', key='workflow_remove_node', width='stretch', disabled=selected_id is None):
                nodes[:] = [node for node in nodes if node['id'] != selected_id]
                edges[:] = [edge for edge in edges if selected_id not in edge]
                st.session_state.workflow_canvas_selected_node = None
                st.rerun()
        workflow_feature_key = f'workflow_features_{dataset_id}_{target}'
        workflow_target_key = f'workflow_target_{dataset_id}'
        project_data = {'version': 1, 'dataset_id': dataset_id, 'nodes': nodes, 'edges': edges,
            'model': st.session_state.get('workflow_model_name', model_name),
            'scaler': st.session_state.get('workflow_scaler_value', params.get('scaler', 'standard')),
            'model_mode': st.session_state.get('workflow_model_mode', 'Ручная настройка'),
            'features': st.session_state.get(workflow_feature_key, features),
            'target': target,
            'model_params': st.session_state.get('workflow_model_params_cache', {}),
            'repair': st.session_state.get('workflow_repair_override')}
        save_action.download_button('Сохранить схему', json.dumps(project_data, ensure_ascii=False, indent=2),
            'experiment_schema.json', 'application/json', key='workflow_save_project')
        project_file = st.file_uploader('Загрузить схему JSON', type=['json'], key='workflow_project_upload')
        if project_file and st.button('Открыть схему', key='workflow_load_project'):
            try:
                imported = json.loads(project_file.getvalue().decode('utf-8'))
                imported_nodes = imported.get('nodes', [])
                valid_types = set(block_specs)
                if imported.get('version') != 1 or not isinstance(imported_nodes, list) or not all(
                    isinstance(node, dict) and node.get('type') in valid_types and isinstance(node.get('id'), str) and node.get('id') for node in imported_nodes
                ):
                    raise ValueError('Формат схемы или тип блока не поддерживается.')
                ids = {node['id'] for node in imported_nodes}
                if len(ids) != len(imported_nodes) or not all(isinstance(node_id, str) for node_id in ids):
                    raise ValueError('Идентификаторы блоков должны быть уникальными строками.')
                imported_edges = imported.get('edges', [])
                if not isinstance(imported_edges, list) or any(not isinstance(edge, (list, tuple)) or len(edge) != 2 or not all(isinstance(endpoint, str) for endpoint in edge) or edge[0] not in ids or edge[1] not in ids for edge in imported_edges):
                    raise ValueError('Связи должны ссылаться на существующие блоки.')
                st.session_state.workflow_nodes = imported_nodes
                st.session_state.workflow_edges = [tuple(edge) for edge in imported_edges]
                st.session_state.workflow_canvas_selected_node = None
                st.session_state.workflow_canvas_instance += 1
                if imported.get('model') in MODELS[task]: st.session_state.workflow_model_name = imported['model']
                if imported.get('scaler') in ('standard', 'robust', 'minmax', 'none'): st.session_state.workflow_scaler_value = imported['scaler']
                if isinstance(imported.get('features'),list):
                    st.session_state[workflow_feature_key] = [column for column in imported['features'] if column in options]
                if imported.get('target') in targets: st.session_state.workflow_target_override = imported['target']
                if isinstance(imported.get('model_params'),dict): st.session_state.workflow_model_params_cache = imported['model_params']
                if imported.get('repair') is None or isinstance(imported.get('repair'),dict):
                    st.session_state.workflow_repair_override = imported.get('repair')
                st.session_state.workflow_canvas_revision = None
                st.rerun()
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                st.error(f'Не удалось загрузить схему: {exc}')

    with canvas:
        st.markdown('#### Экспериментальный холст')
        st.caption('Выберите узел, чтобы открыть его параметры и визуальный результат. Графики показаны прямо в карточках узлов.')
        node_by_id = {node['id']: node for node in nodes}
        flow_nodes = []
        numeric_preview_columns = list(frame.select_dtypes(include='number').columns)
        preview_column = target if target in numeric_preview_columns else (numeric_preview_columns[0] if numeric_preview_columns else None)
        def sampled_series(column, count=28):
            if column not in frame:
                return []
            series = pd.to_numeric(frame[column], errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
            if series.empty:
                return []
            indices = np.linspace(0, len(series)-1, min(count, len(series)), dtype=int)
            return [float(value) for value in series.iloc[indices].tolist()]
        workflow_result_preview = st.session_state.get('result')
        if not workflow_result_preview or workflow_result_preview.get('config', {}).get('dataset_id') != dataset_id:
            workflow_result_preview = None
        def node_preview_for(block_type):
            if block_type in ('Диаграмма рассеяния', 'Корреляции') and len(numeric_preview_columns) >= 2:
                if block_type == 'Корреляции':
                    matrix = frame[numeric_preview_columns[:7]].corr().fillna(0).round(3).values.tolist()
                    return {'kind':'heatmap', 'matrix':matrix, 'label':'Корреляции признаков'}
                x_column, y_column = numeric_preview_columns[:2]
                pair=frame[[x_column,y_column]].replace([np.inf,-np.inf],np.nan).dropna()
                if not pair.empty:
                    indices=np.linspace(0,len(pair)-1,min(28,len(pair)),dtype=int)
                    pair=pair.iloc[indices]
                return {'kind':'scatter','values':pair[x_column].astype(float).tolist(),'secondary':pair[y_column].astype(float).tolist(),'label':f'{x_column} × {y_column}'}
            if block_type == 'Распределение' and preview_column:
                counts = pd.to_numeric(frame[preview_column], errors='coerce').dropna().pipe(lambda s: pd.cut(s, bins=12, duplicates='drop').value_counts().sort_index()).tolist()
                return {'kind':'bars', 'values':[float(value) for value in counts], 'label':f'Распределение · {preview_column}'}
            if block_type in ('Диагностика ошибок', 'Анализ результатов') and workflow_result_preview and 'metrics' in workflow_result_preview:
                metrics = list(workflow_result_preview['metrics'].items())[:3]
                max_value = max((abs(float(v)) for _,v in metrics), default=1) or 1
                return {'kind':'metrics', 'metrics':[[str(name),f'{float(value):.3f}',18+125*abs(float(value))/max_value] for name,value in metrics]}
            if block_type == 'Оценка модели' and workflow_result_preview and 'metrics' in workflow_result_preview:
                metrics = list(workflow_result_preview['metrics'].items())[:3]
                max_value = max((abs(float(v)) for _,v in metrics), default=1) or 1
                return {'kind':'metrics', 'metrics':[[str(name),f'{float(value):.3f}',18+125*abs(float(value))/max_value] for name,value in metrics]}
            if block_type == 'Разбиение выборки':
                try:
                    _, _, split_parts_preview = split_data(frame, features, target, task, params)
                    total = max(1, sum(len(part) for part in split_parts_preview))
                    return {'kind':'split', 'parts':[len(part)/total for part in split_parts_preview]}
                except ValueError:
                    pass
            if block_type == 'Обучение модели':
                fitted_values = []
                fitted_secondary = []
                if workflow_result_preview and 'predictions' in workflow_result_preview:
                    predictions = workflow_result_preview['predictions']
                    actual_column = next((c for c in predictions.columns if c.casefold() in ('actual','y_true','факт')), None)
                    predicted_column = next((c for c in predictions.columns if c.casefold() in ('predicted','y_pred','прогноз')), None)
                    if actual_column and predicted_column:
                        pair=predictions[[actual_column,predicted_column]].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).dropna().head(28)
                        fitted_values=pair[actual_column].astype(float).tolist()
                        fitted_secondary=pair[predicted_column].astype(float).tolist()
                return {'kind':'model', 'family':st.session_state.get('workflow_model_name',model_name),
                    'values':fitted_values or (sampled_series(preview_column) if preview_column else []), 'secondary':fitted_secondary,
                    'label':'Факт и прогноз' if fitted_values else 'Структура модели'}
            if block_type in ('Профиль данных', 'Проверка пропусков', 'Загрузка данных', 'Просмотр данных'):
                if block_type == 'Проверка пропусков':
                    missing_share = (raw.isna().mean().sort_values(ascending=False).head(16)*100).tolist()
                    return {'kind':'bars','values':[float(v) for v in missing_share], 'label':'Пропуски по столбцам'}
                if block_type == 'Профиль данных':
                    missing_share = (frame.isna().mean().sort_values(ascending=False).head(16)*100).tolist()
                    return {'kind':'bars','values':[float(v) for v in missing_share], 'label':'Качество признаков'}
                return {'kind':'line','values':sampled_series(preview_column), 'label':f'Сигнал · {preview_column}' if preview_column else 'Исходный набор'}
            if block_type in ('Стандартизация','Предобработка'):
                return {'kind':'line','values':sampled_series(preview_column), 'secondary':[], 'label':'Масштабирование признаков'}
            return {'kind':'bars','values':[float(frame[c].notna().mean()) for c in numeric_preview_columns[:12]],'label':'Состояние данных'}
        for node in nodes:
            icon, title, _ = block_specs[node['type']]
            detail = {'Загрузка данных':dataset_name, 'Выбор X и Y':f'{len(features)} признаков → {target}',
                      'Обучение модели':st.session_state.get('workflow_model_name', model_name),
                      'Разбиение выборки':params['split']}.get(node['type'], '')
            fill, accent = block_colors.get(node['type'],('#e2e8f0','#64748b'))
            result_ready = bool(workflow_result_preview and 'error' not in workflow_result_preview)
            status = ('Выполнен' if result_ready and node['type'] in ('Обучение модели','Оценка модели','Анализ результатов')
                else 'Ожидает запуска' if node['type'] in required_blocks else 'Готов к исследованию')
            flow_nodes.append(dict(id=node['id'], type=node['type'], icon=icon, label=title, detail=detail, status=status,
                fill=fill, accent=accent, preview=node_preview_for(node['type']), x=node.get('x'), y=node.get('y')))
        canvas_state = workflow_canvas(flow_nodes,
            [dict(id=f'{source}-{target}',source=source,target=target) for source,target in edges
             if source in node_by_id and target in node_by_id],
            selected=st.session_state.get('workflow_canvas_selected_node'), height=680,
            key=f"workflow_canvas_{st.session_state.workflow_canvas_instance}")
        if canvas_state and canvas_state.get('revision') != st.session_state.get('workflow_canvas_revision'):
            positions = {node['id']:(node.get('x'),node.get('y')) for node in canvas_state.get('nodes',[])}
            for node in nodes:
                if node['id'] in positions:
                    node['x'],node['y'] = positions[node['id']]
            new_edges = [(edge['source'],edge['target']) for edge in canvas_state.get('edges',[])
                         if edge.get('source') in node_by_id and edge.get('target') in node_by_id]
            st.session_state.workflow_edges = list(dict.fromkeys(new_edges))
            if canvas_state.get('selected') in node_by_id:
                st.session_state.workflow_canvas_selected_node = canvas_state['selected']
            st.session_state.workflow_canvas_revision = canvas_state['revision']
            st.rerun()

    required_ids = {}
    for block in required_blocks:
        ids = [node['id'] for node in nodes if node['type'] == block]
        if ids: required_ids[block] = ids[0]
    missing = [block for block in required_blocks if block not in required_ids]
    missing_edges = [(a,b) for a,b in zip(required_blocks, required_blocks[1:])
                     if a in required_ids and b in required_ids and (required_ids[a], required_ids[b]) not in edges]
    adjacency = {node['id']: [] for node in nodes}
    for source_id, target_id in edges:
        if source_id in adjacency and target_id in adjacency:
            adjacency[source_id].append(target_id)
    visiting, visited = set(), set()
    def visit(node_id):
        if node_id in visiting: return True
        if node_id in visited: return False
        visiting.add(node_id)
        if any(visit(next_id) for next_id in adjacency[node_id]): return True
        visiting.remove(node_id); visited.add(node_id)
        return False
    has_cycle = any(visit(node_id) for node_id in adjacency if node_id not in visited)
    if missing:
        st.warning('Для запуска добавьте обязательные блоки: ' + ', '.join(missing))
    elif missing_edges:
        st.warning('Соедините обязательные блоки по порядку: ' + ' → '.join(a for a,b in missing_edges) + ' → ' + missing_edges[-1][1])
    elif has_cycle:
        st.warning('В схеме есть цикл. Удалите обратное соединение, чтобы получить выполнимый граф.')

    with inspector:
        st.markdown('#### Исследование узла')
        selected_node = node_by_id.get(st.session_state.get('workflow_canvas_selected_node')) if nodes else None
        if selected_node:
            selected_type = selected_node['type']
            icon, title, description = block_specs[selected_type]
            with st.expander(f'{icon} {selected_type}', expanded=True):
                st.caption(description)
                result = st.session_state.get('result')
                if raw is not None and selected_type in ('Выбор X и Y', 'Обучение модели'):
                    target_choice = st.selectbox('Целевой столбец Y',targets,index=targets.index(target),key=workflow_target_key)
                    if st.button('Применить целевой столбец',key=f"workflow_target_apply_{selected_node['id']}"):
                        st.session_state.workflow_target_override = target_choice
                        st.rerun()
                    valid_features = [column for column in frame.columns if column not in set(detect_time_columns(raw)) | {target}]
                    features = st.multiselect('Входные признаки X', valid_features, default=[column for column in features if column in valid_features], key=workflow_feature_key)
                if selected_type == 'Загрузка данных' and raw is not None:
                    a,b,c=st.columns(3);a.metric('Набор',dataset_name);b.metric('Строк',f'{len(raw):,}');c.metric('Колонок',len(raw.columns))
                    st.caption(f'Источник ID: {dataset_id}')
                    numeric=frame.select_dtypes('number')
                    if not numeric.empty:
                        column=st.selectbox('Предпросмотр сигнала',numeric.columns,key=f'workflow_source_plot_{selected_node["id"]}')
                        st.plotly_chart(px.line(frame.head(1500),y=column,title=f'Исходный сигнал · {column}'),width='stretch')
                    with st.expander('Открыть исходную таблицу'):
                        st.dataframe(raw.head(30),width='stretch')
                elif selected_type == 'Профиль данных' and raw is not None:
                    st.subheader('Структура и качество данных')
                    profile=pd.DataFrame({'Тип':frame.dtypes.astype(str),'Пропуски':frame.isna().sum(),
                        'Пропуски, %':(frame.isna().mean()*100).round(2),'Уникальных':frame.nunique()})
                    profile_plot=profile.assign(Признак=profile.index.astype(str)).sort_values('Пропуски, %',ascending=False).head(20)
                    st.plotly_chart(px.bar(profile_plot,x='Пропуски, %',y='Признак',orientation='h',title='Полнота признаков · доля пропусков'),width='stretch')
                    with st.expander('Подробный профиль столбцов'):
                        st.dataframe(profile,width='stretch')
                    numeric=frame.select_dtypes('number')
                    if not numeric.empty:
                        with st.expander('Распределения признаков'):
                            column=st.selectbox('Числовой признак',numeric.columns,key=f'workflow_profile_plot_{selected_node["id"]}')
                            st.plotly_chart(px.histogram(frame,x=column,marginal='box',title=f'Распределение · {column}'),width='stretch')
                        with st.expander('Описательные статистики'):
                            st.dataframe(numeric.describe().T,width='stretch')
                elif selected_type == 'Проверка пропусков' and raw is not None:
                    missing_table=pd.DataFrame({'До подготовки':raw.isna().sum(),'После подготовки':frame.isna().sum()})
                    missing_table['Восстановлено']=(missing_table['До подготовки']-missing_table['После подготовки']).clip(lower=0)
                    missing_plot=missing_table.assign(Признак=missing_table.index.astype(str)).sort_values('До подготовки',ascending=False).head(20)
                    st.plotly_chart(px.bar(missing_plot,x='Признак',y=['До подготовки','После подготовки'],barmode='group',title='Пропуски до и после обработки'),width='stretch')
                    with st.expander('Таблица восстановления'):
                        st.dataframe(missing_table.sort_values('До подготовки',ascending=False),width='stretch')
                    if frame.attrs.get('repair_records'):
                        with st.expander('Журнал изменённых наблюдений'):
                            st.dataframe(pd.DataFrame(frame.attrs['repair_records']),width='stretch',hide_index=True)
                elif selected_type == 'Очистка данных' and raw is not None:
                    st.markdown('**Параметры обработки пропусков и аномалий**')
                    base_repair = preparation.get('repair') or {
                        'columns':[c for c in numeric_sources if task == 'Регрессия' or c != target],
                        'fill_method':'linear','anomaly_method':'iqr','anomaly_action':'clip',
                        'window':48,'min_history':8,'threshold':3.0,'max_gap':0,'bounds':{}}
                    repair_columns_node = st.multiselect('Колонки для обработки',
                        [c for c in numeric_sources if task == 'Регрессия' or c != target],
                        default=[c for c in base_repair.get('columns',[]) if c in numeric_sources],
                        key=f"workflow_repair_columns_{selected_node['id']}")
                    repair_a, repair_b = st.columns(2)
                    fill_method_node = repair_a.selectbox('Заполнение', ['linear','forward','none'],
                        index=['linear','forward','none'].index(base_repair.get('fill_method','linear')),
                        format_func=lambda value:{'linear':'Линейно по прошлым точкам','forward':'Предыдущее значение','none':'Не заполнять'}[value],
                        key=f"workflow_repair_fill_{selected_node['id']}")
                    anomaly_method_node = repair_b.selectbox('Поиск аномалий', ['iqr','mad','zscore','none'],
                        index=['iqr','mad','zscore','none'].index(base_repair.get('anomaly_method','iqr')),
                        format_func=lambda value:{'iqr':'IQR','mad':'MAD','zscore':'Z-score','none':'Только физические границы'}[value],
                        key=f"workflow_repair_method_{selected_node['id']}")
                    repair_a, repair_b, repair_c = st.columns(3)
                    repair_window_node = int(repair_a.number_input('Окно',8,1000,int(base_repair.get('window',48)),key=f"workflow_repair_window_{selected_node['id']}"))
                    repair_history_node = int(repair_b.number_input('История',2,repair_window_node,min(int(base_repair.get('min_history',8)),repair_window_node),key=f"workflow_repair_history_{selected_node['id']}"))
                    repair_threshold_node = float(repair_c.number_input('Порог',0.1,20.0,float(base_repair.get('threshold',3.0)),0.1,key=f"workflow_repair_threshold_{selected_node['id']}"))
                    repair_action_node = st.selectbox('Действие с аномалиями',['clip','replace'],
                        index=['clip','replace'].index(base_repair.get('anomaly_action','clip')),
                        format_func=lambda value:{'clip':'Ограничить оценкой','replace':'Удалить и восстановить'}[value],
                        key=f"workflow_repair_action_{selected_node['id']}")
                    if st.button('Применить обработку к данным',key=f"workflow_repair_apply_{selected_node['id']}"):
                        st.session_state.workflow_repair_override = dict(columns=repair_columns_node,
                            fill_method=fill_method_node,anomaly_method=anomaly_method_node,
                            anomaly_action=repair_action_node,window=repair_window_node,
                            min_history=repair_history_node,threshold=repair_threshold_node,
                            max_gap=int(base_repair.get('max_gap',0)),bounds=base_repair.get('bounds',{}))
                        st.rerun()
                    st.caption('Обработчик пересчитывает входные данные, используемые при обучении.')
                    repair_compare=pd.DataFrame({'До обработки':raw.isna().sum(),'После обработки':frame.isna().sum()})
                    repair_compare['Признак']=repair_compare.index.astype(str)
                    st.plotly_chart(px.bar(repair_compare.head(20),x='Признак',y=['До обработки','После обработки'],barmode='group',title='Пропуски до и после очистки'),width='stretch')
                    with st.expander('Сводка и настройки обработки'):
                        st.write({'Настройки восстановления':preparation.get('repair'),
                                  'Исключено некорректных временных меток':frame.attrs.get('invalid_time_rows',0)})
                        st.dataframe(repair_compare.drop(columns='Признак'),width='stretch')
                    if frame.attrs.get('repair_records'):
                        with st.expander('Журнал исправлений'):
                            st.dataframe(pd.DataFrame(frame.attrs['repair_records']),width='stretch',hide_index=True)
                elif selected_type == 'Корреляции' and raw is not None:
                    numeric=frame.select_dtypes('number')
                    if len(numeric.columns)>=2:
                        corr=numeric.corr()
                        st.plotly_chart(px.imshow(corr,zmin=-1,zmax=1,text_auto='.2f',color_continuous_scale='RdBu_r',title='Корреляции Пирсона'),width='stretch')
                    else: st.info('Для корреляционной матрицы нужны как минимум два числовых признака.')
                elif selected_type == 'Распределение' and raw is not None:
                    numeric=frame.select_dtypes('number').columns.tolist()
                    if numeric:
                        column=st.selectbox('Числовой признак',numeric,key=f'workflow_dist_col_{selected_node["id"]}')
                        bins=st.slider('Число интервалов',5,100,30,key=f'workflow_dist_bins_{selected_node["id"]}')
                        color=target if task=='Классификация' and target in frame and frame[target].nunique()<=12 else None
                        st.plotly_chart(px.histogram(frame,x=column,color=color,nbins=bins,marginal='box',title=f'Распределение · {column}'),width='stretch')
                    else: st.info('В наборе нет числовых столбцов для гистограммы.')
                elif selected_type == 'Диаграмма рассеяния' and raw is not None:
                    numeric=frame.select_dtypes('number').columns.tolist()
                    if len(numeric)>=2:
                        a,b=st.columns(2)
                        x_col=a.selectbox('X',numeric,key=f'workflow_scatter_x_{selected_node["id"]}')
                        y_col=b.selectbox('Y',numeric,index=min(1,len(numeric)-1),key=f'workflow_scatter_y_{selected_node["id"]}')
                        color=target if target in frame and frame[target].nunique()<=20 else None
                        st.plotly_chart(px.scatter(frame,x=x_col,y=y_col,color=color,title=f'{y_col} от {x_col}'),width='stretch')
                    else: st.info('Для диаграммы рассеяния нужны два числовых столбца.')
                elif selected_type == 'Ящик с усами' and raw is not None:
                    numeric=frame.select_dtypes('number').columns.tolist()
                    if numeric:
                        column=st.selectbox('Числовой признак',numeric,key=f'workflow_box_col_{selected_node["id"]}')
                        group_options=['Без группировки']+[c for c in frame.columns if c != column and frame[c].nunique(dropna=True)<=20]
                        group=st.selectbox('Разбить по категории',group_options,key=f'workflow_box_group_{selected_node["id"]}')
                        category=None if group=='Без группировки' else group
                        st.plotly_chart(px.box(frame,x=category,y=column,color=category,points='outliers',
                            title=f'Размах и выбросы · {column}'),width='stretch')
                        st.caption('Коробка показывает межквартильный размах, линия — медиану; точки за усами помогают заметить потенциальные выбросы.')
                    else: st.info('Для ящика с усами нужен числовой столбец.')
                elif selected_type == 'Просмотр данных' and raw is not None:
                    st.caption(f'Подготовленная таблица: {len(frame):,} строк × {len(frame.columns)} колонок.')
                    numeric=frame.select_dtypes('number')
                    if len(numeric.columns)>=2:
                        st.plotly_chart(px.scatter(frame.sample(min(1500,len(frame)),random_state=42),x=numeric.columns[0],y=numeric.columns[1],
                            color=target if task=='Классификация' and target in frame and frame[target].nunique()<=15 else None,
                            title='Карта наблюдений по двум признакам'),width='stretch')
                    elif len(numeric.columns)==1:
                        st.plotly_chart(px.line(frame.head(1500),y=numeric.columns[0],title=f'Сигнал · {numeric.columns[0]}'),width='stretch')
                    with st.expander('Открыть подготовленную таблицу'):
                        st.dataframe(frame.head(100),width='stretch')
                elif selected_type == 'Экспорт данных' and raw is not None:
                    st.download_button('Скачать подготовленные данные CSV', frame.to_csv(index=True).encode('utf-8-sig'),
                        'prepared_data.csv', 'text/csv', key='workflow_download_data')
                elif selected_type == 'Инженерия признаков' and raw is not None:
                    numeric=frame.select_dtypes('number')
                    if preview_column:
                        st.plotly_chart(px.line(frame.head(1500),y=preview_column,title=f'Результирующий признак · {preview_column}'),width='stretch')
                    with st.expander('Признаки и параметры подготовки'):
                        st.caption(f'Числовых признаков: {len(numeric.columns)} · категориальных: {frame.shape[1]-len(numeric.columns)}')
                        st.json(preparation)
                        st.dataframe(frame.head(30),width='stretch')
                elif selected_type == 'Выбор X и Y' and raw is not None:
                    st.metric('Входные признаки',len(features),help='Переменные X, передаваемые модели')
                    st.caption(f'Цель **{target}** · задача {task}')
                    numeric_features=[column for column in features if column in frame and pd.api.types.is_numeric_dtype(frame[column])]
                    if numeric_features and target in frame and pd.api.types.is_numeric_dtype(frame[target]):
                        x_feature=st.selectbox('Исследовать связь X с Y',numeric_features,key=f'workflow_xy_plot_{selected_node["id"]}')
                        st.plotly_chart(px.scatter(frame.sample(min(2000,len(frame)),random_state=42),x=x_feature,y=target,
                            trendline='ols',title=f'{target} от {x_feature}'),width='stretch')
                    with st.expander('Входная таблица X и Y'):
                        st.dataframe(frame[features+[target]].head(100),width='stretch')
                elif selected_type == 'Разбиение выборки' and raw is not None:
                    try:
                        X,y,parts=split_data(frame,features,target,task,params)
                        split_rows=[]
                        for name,indices in zip(['Обучение','Валидация','Тест'],parts):
                            subset=X.iloc[indices]
                            split_rows.append({'Часть':name,'Строк':len(indices),'Начало':str(subset.index.min()),'Конец':str(subset.index.max())})
                        st.bar_chart(pd.DataFrame({'Наблюдения':[row['Строк'] for row in split_rows]},index=[row['Часть'] for row in split_rows]))
                        with st.expander('Границы выборок'):
                            st.dataframe(pd.DataFrame(split_rows),width='stretch',hide_index=True)
                        st.caption('Разбиение соответствует текущим настройкам раздела «Данные и признаки» и не переобучает модель.')
                    except ValueError as exc: st.warning(str(exc))
                elif selected_type == 'Проверка разбиения' and raw is not None:
                    try:
                        X,y,parts=split_data(frame,features,target,task,params)
                        st.bar_chart(pd.DataFrame({'Строк':[len(part) for part in parts]},index=['train','validation','test']))
                        st.dataframe(pd.DataFrame([{'Часть':name,'Строк':len(idx),'Первый индекс':str(X.iloc[idx].index.min()),'Последний индекс':str(X.iloc[idx].index.max())}
                            for name,idx in zip(['train','validation','test'],parts)]),width='stretch',hide_index=True)
                    except ValueError as exc: st.warning(str(exc))
                elif selected_type == 'Экспорт модели':
                    saved_result = result
                    if saved_result and 'pipeline' in saved_result:
                        model_file = io.BytesIO(); joblib.dump({'pipeline':saved_result['pipeline'],'config':saved_result.get('config',{})}, model_file)
                        st.download_button('Скачать обученную модель', model_file.getvalue(), 'model.joblib', 'application/octet-stream', key='workflow_download_model')
                    else: st.info('Сначала обучите модель.')
                elif selected_type == 'Анализ результатов':
                    saved_result = result
                    if saved_result and 'metrics' in saved_result:
                        for metric_col, (metric_name, metric_value) in zip(st.columns(len(saved_result['metrics'])), saved_result['metrics'].items()):
                            metric_col.metric(metric_name, f'{metric_value:.4f}')
                        st.dataframe(pd.DataFrame(saved_result.get('leaderboard',[])),width='stretch',hide_index=True)
                        st.plotly_chart(timeline_figure(saved_result),width='stretch',key=f'workflow_timeline_{selected_node["id"]}')
                    else: st.info('После выполнения пайплайна здесь появятся метрики.')
                elif selected_type == 'Оценка модели':
                    if result and 'metrics' in result:
                        for metric_col,(metric_name,metric_value) in zip(st.columns(len(result['metrics'])),result['metrics'].items()): metric_col.metric(metric_name,f'{metric_value:.4f}')
                        st.dataframe(pd.DataFrame(result.get('leaderboard',[])),width='stretch',hide_index=True)
                    else: st.info('Выполните эксперимент, чтобы получить validation- и test-метрики.')
                elif selected_type == 'Диагностика ошибок':
                    if result and 'predictions' in result:
                        figures=regression_figures(result['predictions']) if result['task']=='Регрессия' else classification_figures(result)
                        for name,figure in figures.items(): st.plotly_chart(figure,width='stretch',key=f'workflow_diagnostic_{selected_node["id"]}_{name}')
                    else: st.info('Диагностика появится после обучения модели.')
                elif selected_type == 'Важность признаков':
                    if result and 'pipeline' in result:
                        cache_key=f'workflow_importance_{selected_node["id"]}_{str(result.get("signature",""))[:12]}'
                        if st.button('Рассчитать permutation importance',key=cache_key+'_run'):
                            try: st.session_state[cache_key]=permutation_scores(result)
                            except Exception as exc: st.error(str(exc))
                        if cache_key in st.session_state:
                            scores=st.session_state[cache_key]
                            st.dataframe(scores,width='stretch',hide_index=True)
                            st.plotly_chart(importance_figure(scores,'Permutation importance · validation'),width='stretch',key=cache_key+'_chart')
                    else: st.info('Важность рассчитывается после обучения модели.')
                elif selected_type == 'SHAP объяснение':
                    if result and 'pipeline' in result:
                        cache_key=f'workflow_shap_{selected_node["id"]}_{str(result.get("signature",""))[:12]}'
                        count_rows=st.slider('Строк validation для SHAP',2,24,8,key=cache_key+'_count')
                        if st.button('Рассчитать SHAP',key=cache_key+'_run'):
                            try: st.session_state[cache_key]=shap_scores(result,count=count_rows)
                            except Exception as exc: st.error(str(exc))
                        if cache_key in st.session_state:
                            explanation=st.session_state[cache_key]
                            st.dataframe(explanation['summary'].head(30),width='stretch',hide_index=True)
                            row=st.selectbox('Локальная строка validation',range(len(explanation['indices'])),key=cache_key+'_row')
                            st.plotly_chart(shap_local_figure(explanation,row),width='stretch',key=cache_key+'_local')
                    else: st.info('SHAP можно рассчитать после обучения классической модели.')
                elif selected_type == 'Integrated Gradients':
                    if result and 'pipeline' in result:
                        cache_key=f'workflow_ig_{selected_node["id"]}_{str(result.get("signature",""))[:12]}'
                        steps=st.slider('Шаги интегрирования',8,64,24,key=cache_key+'_steps')
                        if st.button('Рассчитать Integrated Gradients',key=cache_key+'_run'):
                            try: st.session_state[cache_key]=integrated_gradients(result,steps=steps)
                            except Exception as exc: st.error(str(exc))
                        if cache_key in st.session_state:
                            scores=st.session_state[cache_key]
                            st.dataframe(scores,width='stretch',hide_index=True)
                            st.plotly_chart(importance_figure(scores,'Integrated Gradients'),width='stretch',key=cache_key+'_chart')
                    else: st.info('Integrated Gradients доступен после обучения LSTM, GRU или RNN.')
                elif selected_type == 'Экспорт отчёта':
                    if result and 'metrics' in result:
                        st.download_button('Скачать HTML-отчёт',html_report(result,result.get('config',{})),
                            'experiment_report.html','text/html',key='workflow_html_report')
                    else: st.info('Отчёт появится после обучения модели.')
                elif selected_type == 'Экспорт архива':
                    if result and 'metrics' in result:
                        include=st.checkbox('Включить обученную модель',value=True,key='workflow_archive_include_model')
                        archive_key=f'workflow_archive_{selected_node["id"]}_{str(result.get("signature",""))[:12]}_{include}'
                        if st.button('Подготовить архив',key=archive_key+'_prepare'):
                            try: st.session_state[archive_key]=results_bundle(result,st.session_state.get('chart_styles',{}),include_model=include)
                            except Exception as exc: st.error(f'Не удалось собрать архив: {exc}')
                        if archive_key in st.session_state:
                            st.download_button('Скачать архив эксперимента',st.session_state[archive_key],
                                'experiment_results.zip','application/zip',key=archive_key+'_download')
                    else: st.info('Архив появится после обучения модели.')
                elif selected_type in ('Предобработка', 'Стандартизация') and raw is not None:
                    scaler_name = st.selectbox('Преобразование числовых признаков', ['standard','robust','minmax','none'],
                        index=['standard','robust','minmax','none'].index(st.session_state.get('workflow_scaler_value',params.get('scaler','standard'))),
                        format_func=lambda x:{'standard':'Стандартизация · StandardScaler','robust':'Устойчивое масштабирование · RobustScaler','minmax':'Нормализация 0–1 · MinMaxScaler','none':'Без масштабирования'}[x], key='workflow_scaler_widget')
                    st.session_state.workflow_scaler_value = scaler_name
                    st.caption('Параметры преобразователя обучаются только на train и применяются при запуске.')
                    numeric_features = [c for c in features if c in frame and pd.api.types.is_numeric_dtype(frame[c])]
                    if numeric_features:
                        scaled_column = st.selectbox('Признак для сравнения до и после',numeric_features,key=f"workflow_scale_column_{selected_node['id']}")
                        try:
                            split_X, _, split_parts = split_data(frame,features,target,task,params)
                            values = pd.to_numeric(frame[scaled_column],errors='coerce')
                            train_values = pd.to_numeric(split_X[scaled_column].iloc[split_parts[0]],errors='coerce').dropna().to_numpy().reshape(-1,1)
                            median = float(pd.Series(train_values.ravel()).median())
                            filled = values.fillna(median).to_numpy().reshape(-1,1)
                            scaler_class = {'standard':StandardScaler,'robust':RobustScaler,'minmax':MinMaxScaler,'none':None}[scaler_name]
                            transformed = filled.ravel() if scaler_class is None else scaler_class().fit(train_values).transform(filled).ravel()
                            comparison = pd.DataFrame({'До преобразования':values.to_numpy(),'После преобразования':transformed},index=frame.index)
                            st.plotly_chart(px.line(comparison.head(500),title=f'До и после · {scaled_column} · scaler обучен на train'),width='stretch',key=f"workflow_scale_plot_{selected_node['id']}")
                            st.dataframe(comparison.head(20),width='stretch')
                        except ValueError as exc:
                            st.info(f'Сравнение появится после настройки разбиения: {exc}')
                elif selected_type == 'Обучение модели' and raw is not None:
                    workflow_names = available_models(task, params['split'])
                    if st.session_state.get('workflow_model_name') not in workflow_names:
                        st.session_state.workflow_model_name = model_name if model_name in workflow_names else workflow_names[0]
                    workflow_model = st.selectbox('Алгоритм', workflow_names, key='workflow_model_name')
                    st.caption(MODEL_REGISTRY[workflow_model].get('description', f"{workflow_model} · {MODEL_REGISTRY[workflow_model]['group']} · {MODEL_REGISTRY[workflow_model]['package']}"))
                    workflow_mode=st.selectbox('Стратегия обучения',['Ручная настройка','Исследование диапазонов','Автоматическая оптимизация'],key='workflow_model_mode')
                    workflow_model_params, workflow_space = parameter_controls(workflow_model, task, workflow_mode, namespace='workflow')
                    st.session_state.workflow_model_params_cache = workflow_model_params
                    st.session_state.workflow_space_cache = workflow_space
                    workflow_count=st.number_input('Число вариантов / испытаний',1,100,5,key='workflow_model_count') if workflow_mode!='Ручная настройка' else 1
                    search_config=dict(model=workflow_model,task=task,mode=workflow_mode,params=workflow_model_params,space=workflow_space,count=int(workflow_count),split=params)
                    search_signature=hashlib.sha256(json.dumps(search_config,sort_keys=True,default=str).encode()).hexdigest()
                    st.session_state.workflow_search_signature_current=search_signature
                    if workflow_mode=='Исследование диапазонов' and st.button('Сгенерировать варианты',key='workflow_generate_candidates'):
                        st.session_state.workflow_candidates=generate_candidates(task,[workflow_model],int(workflow_count),params,
                            {workflow_model:workflow_space|{'__fixed__':workflow_model_params}})
                        st.session_state.workflow_candidates_signature=search_signature
                    if workflow_mode=='Исследование диапазонов' and st.session_state.get('workflow_candidates_signature')==search_signature:
                        st.dataframe(json_table(st.session_state.workflow_candidates),width='stretch',hide_index=True)

    if workflow_feature_key in st.session_state:
        features = [column for column in st.session_state[workflow_feature_key] if column in options]
    active_workflow_mode=st.session_state.get('workflow_model_mode','Ручная настройка')
    candidates_ready=active_workflow_mode!='Исследование диапазонов' or st.session_state.get('workflow_candidates_signature')==st.session_state.get('workflow_search_signature_current')
    ready = raw is not None and (time_ready or params['split'] != 'time') and bool(features) and not missing and not missing_edges and not has_cycle and candidates_ready
    if run_action.button('▶ Запустить', type='primary', key='run_visual_pipeline', disabled=not ready, width='stretch'):
        try:
            active_model = st.session_state.get('workflow_model_name', model_name)
            pipeline_params = params | st.session_state.get('workflow_model_params_cache', model_params) | {'scaler':st.session_state.get('workflow_scaler_value',params.get('scaler','standard'))}
            workflow_space=st.session_state.get('workflow_space_cache',{})
            workflow_count=int(st.session_state.get('workflow_model_count',5)) if active_workflow_mode!='Ручная настройка' else 1
            workflow_config = dict(index=config['index'], dataset=dataset_name, dataset_id=dataset_id, features=features,
                target=target, task=task, preparation=preparation, params=pipeline_params, model=active_model,
                mode=active_workflow_mode, space=workflow_space, count=workflow_count, nodes=nodes.copy(), edges=edges.copy())
            workflow_signature = hashlib.sha256(json.dumps(workflow_config, sort_keys=True, default=str).encode()).hexdigest()
            progress=st.progress(0)
            candidates=st.session_state.get('workflow_candidates',[]) if active_workflow_mode=='Исследование диапазонов' else [{'name':active_model,'params':pipeline_params}]
            search=dict(count=workflow_count,families=[active_model],spaces={active_model:workflow_space|{'__fixed__':st.session_state.get('workflow_model_params_cache',{})}}) if active_workflow_mode=='Автоматическая оптимизация' else None
            result=run_experiments(frame,features,target,task,candidates,pipeline_params,
                progress=lambda current,total,name:progress.progress(current/total,text=name),search=search)
            st.session_state.pop('results_export',None); st.session_state.result=result
            result.update(config=workflow_config, signature=workflow_signature, source='workflow',
                settings=preparation | {'time_columns':detect_time_columns(raw)}, raw_history=raw.copy())
            if 'error' not in result:
                st.session_state.history.append(dict(Модель=result['name'],**result['metrics']))
                try: store.save(result,workflow_config)
                except Exception as exc: st.warning(f'Модель обучена; история не сохранена: {exc}')
                st.session_state.training_notice='Эксперимент выполнен. Откройте результаты.'
                st.rerun()
        except Exception as exc:
            st.error(f'Ошибка выполнения пайплайна: {exc}')

    with st.expander('Данные, результаты и журнал', expanded=False):
        bottom_panel = st.segmented_control('Раздел панели результатов', ['Графики', 'Выход эксперимента', 'Входные данные', 'Журнал и сравнение'],
            default='Графики',key='workflow_bottom_panel',label_visibility='collapsed')
        if bottom_panel == 'Входные данные':
            st.caption(f'Подготовленные входные данные · {len(frame):,} строк × {len(frame.columns)} столбцов')
            st.dataframe(frame.head(100),width='stretch')
        workflow_result = st.session_state.get('result')
        same_dataset_result = bool(workflow_result and workflow_result.get('config',{}).get('dataset_id') == dataset_id)
        if bottom_panel == 'Выход эксперимента':
            if same_dataset_result and 'error' not in workflow_result:
                st.subheader(f"{workflow_result.get('name','Модель')} · test")
                metric_columns=st.columns(max(1,len(workflow_result.get('metrics',{}))))
                for column,(metric_name,metric_value) in zip(metric_columns,workflow_result.get('metrics',{}).items()):
                    column.metric(metric_name,f'{metric_value:.4f}')
                st.dataframe(workflow_result.get('predictions',pd.DataFrame()).head(200),width='stretch')
            elif workflow_result and 'error' in workflow_result:
                st.error(workflow_result['error'])
            else:
                st.info('Результаты этого набора появятся после запуска эксперимента.')
        elif bottom_panel == 'Графики':
            if same_dataset_result and 'error' not in workflow_result:
                st.plotly_chart(timeline_figure(workflow_result),width='stretch',key='workflow_bottom_timeline')
                if workflow_result.get('predictions') is not None and not workflow_result['predictions'].empty:
                    figures=regression_figures(workflow_result['predictions']) if workflow_result['task']=='Регрессия' else classification_figures(workflow_result)
                    for figure_name,figure in figures.items():
                        st.plotly_chart(figure,width='stretch',key=f'workflow_bottom_{figure_name}')
            else:
                numeric=frame.select_dtypes('number')
                if len(numeric.columns)>=2:
                    st.plotly_chart(px.imshow(numeric.corr(),zmin=-1,zmax=1,text_auto='.2f',color_continuous_scale='RdBu_r',title='Корреляции подготовленных признаков'),width='stretch',key='workflow_bottom_correlation')
                else: st.info('Графики оценки модели появятся после запуска.')
        elif bottom_panel == 'Журнал и сравнение':
            try:
                dataset_runs=[record for record in store.list() if record.get('config',{}).get('dataset_id')==dataset_id]
                if dataset_runs:
                    table=[]
                    for record in dataset_runs[:20]:
                        parameters=record.get('config',{}).get('params',{})
                        table.append({'Время':record.get('created'),'Модель':record.get('model'),
                            'Параметры':json.dumps(parameters,ensure_ascii=False,sort_keys=True),**record.get('metrics',{})})
                    st.dataframe(pd.DataFrame(table),width='stretch',hide_index=True)
                    st.caption('Каждый запуск сохраняется в SQLite. Измените параметры, выполните схему ещё раз и сравните строки при одинаковых данных и разбиении.')
                else:
                    st.info('Для выбранного набора пока нет сохранённых запусков.')
            except Exception as exc:
                st.error(f'Не удалось прочитать журнал экспериментов: {exc}')
with st.container(key='page_results'):
    result=st.session_state.get('result')
    if not result: st.info('Запустите эксперимент в разделе «Схема эксперимента» или «Модель и обучение».')
    elif 'error' in result:
        st.error(result['error']); st.dataframe(result['leaderboard'])
    else:
        if result.get('source')!='workflow' and result['signature']!=signature: st.warning('Настройки изменены. Здесь показан предыдущий эксперимент.')
        st.subheader(result['name']+' · финальный test')
        if 'evaluation_sizes' in result:
            sizes=result['evaluation_sizes']
            st.caption(f"Метрики по исходным измерениям: validation {sizes['validation']}, test {sizes['test']}. Восстановленных Y в train: {sizes['restored_train']}.")
        for column,(name,value) in zip(st.columns(len(result['metrics'])),result['metrics'].items()): column.metric(name,f'{value:.4f}')
        with st.container(key='page_results_quality'):
            st.info('Временной ход показывает, насколько модель повторяет измерения на train и на отложенном test. '
                'Сравните прогноз с фактом и базовой линией: преимущество над простым базовым прогнозом должно сохраняться на test. '
                'Хорошее совпадение на train при заметном ухудшении на test может указывать на переобучение или изменение режима данных.')
            if 'train_predictions' not in result:
                st.caption('Для отображения обучающей выборки запустите обучение заново: текущий результат сохранён до обновления графиков.')
            show_result_chart(timeline_figure(result),key='train_test_timeline')
            st.caption('Train — оценка на тех же данных, по которым модель обучалась; это не независимая проверка. '
                'Разрывы факта означают отсутствие исходных измерений; восстановленные Y не подменяют факт. '
                'При случайном разбиении периоды train и test могут перекрываться по времени.')
            predictions=result['predictions']
            figures=regression_figures(predictions) if result['task']=='Регрессия' else classification_figures(result)
            for name,figure in figures.items():
                show_result_chart(figure,key='quality_'+name)
            if result['task']=='Регрессия':
                st.caption('Остаток = факт − прогноз. Накопленные MAE/RMSE учитывают все test-наблюдения до указанной даты. Автокорреляция показывает зависимость ошибок через заданное число строк, включая неравномерные ряды.')
            elif result.get('test_probabilities') is not None:
                positive=st.selectbox('Класс для ROC, PR и калибровки',result['probability_classes'])
                result['display_probability_class']=positive
                probability_plots=probability_figures(result,positive)
                for name,figure in probability_plots.items():
                    show_result_chart(figure,key='probability_'+name)
                if not probability_plots: st.info('Для этих кривых в test нужны и выбранный класс, и другие классы.')
                st.caption('ROC и PR рассчитаны по принципу «выбранный класс против остальных». Линии калибровки на небольшом test могут быть нестабильны.')
            else:
                st.caption('Модель не предоставляет predict_proba: ROC, PR и калибровка вероятностей недоступны.')
            st.caption('Базовый прогноз: '+result.get('baseline_label','предыдущее доступное значение цели'))
            if 'baseline' in result: st.write('Метрики базового прогноза и модели на одинаковых измерениях test',result['baseline'])
            st.dataframe(predictions,width='stretch')
        with st.container(key='page_results_learning'):
            if result['task']=='Регрессия':
                st.info('Столбцы показывают качество отдельно за каждый день train и test. MAE — средняя абсолютная ошибка; '
                    'RMSE сильнее реагирует на крупные промахи: обе метрики лучше при меньших значениях. '
                    'R² ближе к 1 — лучше; отрицательный R² означает качество хуже постоянного среднего на оцениваемом участке. '
                    'MAPE — средняя относительная ошибка в процентах, показана линией на правой оси.')
            else:
                st.info('Столбцы показывают Accuracy и F1 macro по дням train и test. Accuracy — доля верных ответов; '
                    'F1 macro усредняет F1 равноправно по классам. Обе метрики лучше при значениях ближе к 1; '
                    'при дисбалансе ориентируйтесь также на F1 и метрики каждого класса.')
            daily=daily_figure(result)
            if daily is not None: show_result_chart(daily,key='daily_metrics_train_test')
            curve=training_figure(result)
            if curve is not None:
                show_result_chart(curve,key='training_curve')
                st.caption('Loss — функция потерь модели, её масштаб может отличаться от MAE/RMSE. Для LSTM/GRU/RNN validation loss относится к внутренней части train, не к внешней validation; пунктир отмечает выбранную эпоху. Потери MLP относятся к итерациям обучения. Рост validation loss при снижении train loss может указывать на переобучение.')
            else: st.info('Эта модель не сохраняет покадровую историю loss. Доступны метрики test по дням и сравнение кандидатов на validation.')
            st.caption('MAE/RMSE выражены в единицах Y, R² безразмерен: сравнивайте высоты только для одной и той же метрики. '
                'Для крупных величин R² может быть плохо виден рядом с ошибками — скрывайте другие ряды в легенде. '
                'MAPE не учитывает нулевые факты и нестабилен около нуля. При одном измерении R² не определён; '
                'пропуски метрик не заменяются нулями.')
        with st.container(key='page_results_comparison'):
            st.info('Кандидаты сравниваются на общей validation: выше R² или F1 macro — лучше. '
                'Линия лучшего результата показывает прогресс поиска, а график времени помогает оценить цену улучшения. '
                'Ошибки отдельных кандидатов перечислены в таблице. Test используется для финальной оценки победителя.')
            for name,figure in comparison_figures(result).items():
                show_result_chart(figure,key='comparison_'+name)
            st.caption('Все кандидаты сравниваются на validation. Test не участвует в выборе победителя.')
            st.dataframe(json_table([{'name':r['Модель'],'params':r['Параметры']} for r in result['leaderboard']]),width='stretch')
            st.dataframe(pd.DataFrame(result['leaderboard']).drop(columns='Параметры'),width='stretch')
        with st.container(key='page_results_explain'):
            st.info('Общая важность показывает, какие признаки сильнее влияют на модель. '
                'Локальный SHAP объясняет один прогноз: положительный вклад повышает выход относительно базового значения, отрицательный — снижает. '
                'Permutation Importance показывает потерю качества при перемешивании признака; большая положительная важность означает большую полезность.')
            recurrent=result['name'] in RECURRENT_MODELS
            methods=['Integrated Gradients'] if recurrent else ['Permutation Importance','SHAP']
            method=st.selectbox('Метод',methods)
            class_index=0
            if method=='SHAP' and result['task']=='Классификация':
                model_classes=result['pipeline'].classes_.astype(str).tolist()
                explained_class=st.selectbox('Объясняемый класс',model_classes)
                class_index=model_classes.index(explained_class)
            explanation_key=method+':'+str(class_index)
            if st.button('Рассчитать объяснение'):
                try:
                    with st.spinner('Расчёт на validation…'):
                        if method=='Permutation Importance': payload=permutation_scores(result)
                        elif method=='SHAP': payload=shap_scores(result,class_index=class_index)
                        else: payload=integrated_gradients(result)
                    result.setdefault('explanations',{})[explanation_key]=payload
                except Exception as exc: st.error(str(exc))
            payload=result.get('explanations',{}).get(explanation_key)
            if payload is not None:
                table=payload['summary'] if method=='SHAP' else payload
                show_result_chart(importance_figure(table,method+' · важность признаков на validation'),key='importance',note_key='importance_'+method)
                st.dataframe(table,width='stretch')
                if method=='SHAP':
                    row=st.selectbox('Наблюдение для локального объяснения',list(range(len(payload['indices']))),
                        format_func=lambda i:str(payload['indices'][i]))
                    show_result_chart(shap_local_figure(payload,row),key='shap_local')
                    feature=st.selectbox('Признак для графика зависимости SHAP',payload['names'])
                    result.setdefault('explanation_views',{})[explanation_key]=dict(row=row,feature=feature)
                    feature_index=payload['names'].index(feature)
                    dependence=pd.DataFrame({'Значение признака':payload['inputs'][:,feature_index],
                        'SHAP-вклад':payload['values'][:,feature_index]})
                    show_result_chart(px.scatter(dependence,x='Значение признака',y='SHAP-вклад',title='SHAP · зависимость вклада от значения признака'),key='shap_dependence')
                    st.caption('SHAP объясняет кодированные/нормированные признаки ограниченной выборки validation. Waterfall: базовое значение + вклады = выход модели для выбранной строки; это может быть вероятность класса или decision score.')
                elif method=='Permutation Importance':
                    st.caption('Важность — увеличение MAE или уменьшение F1 macro после перестановки признака. Полосы показывают стандартное отклонение пяти перестановок, не доверительный интервал. Отрицательное значение возможно.')
                else:
                    st.caption('Средние абсолютные вклады Integrated Gradients суммируются по шагам окна относительно нулевой точки в нормированном пространстве.')
            else: st.info('Нажмите «Рассчитать объяснение». Результат сохранится при переключении графиков этого эксперимента.')
            st.caption('Вклады описывают поведение модели и не доказывают причинность. Коррелирующие признаки могут делить вклад. Объяснения используют validation и не меняют выбор победителя.')
        with st.container(key='page_results_exports'):
            st.markdown('**Полученные результаты эксперимента**')
            st.caption('Архив включает метрики train/validation/test, прогнозы, дневные метрики, сравнение кандидатов, рассчитанные объяснения и интерактивные графики с выбранным оформлением.')
            include_model=st.checkbox('Включить обученную модель в архив',value=True,key='export_include_model')
            if st.button('Подготовить архив результатов',key='prepare_results_export'):
                with st.spinner('Формирование архива…'):
                    st.session_state.results_export=dict(signature=result['signature'],include_model=include_model,
                        styles=json.dumps(st.session_state.get('chart_styles',{}),sort_keys=True),
                        explanations=list(result.get('explanations',{})),views=json.dumps(result.get('explanation_views',{}),sort_keys=True),probability_class=result.get('display_probability_class'),
                        data=results_bundle(result,st.session_state.get('chart_styles',{}),include_model))
            bundle=st.session_state.get('results_export')
            if bundle and bundle['signature']==result['signature'] and bundle['include_model']==include_model:
                if bundle['styles']!=json.dumps(st.session_state.get('chart_styles',{}),sort_keys=True) or bundle['explanations']!=list(result.get('explanations',{})) or bundle.get('views')!=json.dumps(result.get('explanation_views',{}),sort_keys=True) or bundle.get('probability_class')!=result.get('display_probability_class'):
                    st.caption('Оформление или объяснения изменены. Подготовьте архив заново для включения изменений.')
                st.download_button('Скачать результаты ZIP',bundle['data'],'ml_research_results.zip','application/zip')
            st.caption('CSV содержит значения прогнозов и исходные факты, JSON — настройки эксперимента, joblib — обученную модель с preprocessing. '
                'HTML-отчёт объединяет метрики и таблицы; для воспроизведения нужны исходные данные и сохранённая конфигурация.')
            st.download_button('Прогнозы CSV',predictions.to_csv(index=True).encode('utf-8-sig'),'predictions.csv','text/csv')
            if 'train_predictions' in result:
                st.download_button('Временной ход train CSV',result['train_predictions'].to_csv(index=True).encode('utf-8-sig'),'train_predictions.csv','text/csv')
            st.download_button('Сравнение CSV',pd.DataFrame(result['leaderboard']).to_csv(index=False).encode('utf-8-sig'),'comparison.csv','text/csv')
            st.download_button('Конфигурация JSON',json.dumps(result['config'],ensure_ascii=False,indent=2),'experiment.json','application/json')
            artifact=io.BytesIO();joblib.dump(dict(pipeline=result['pipeline'],config=result['config']),artifact)
            st.download_button('Модель и preprocessing',artifact.getvalue(),'model.joblib','application/octet-stream')
            st.download_button('HTML-отчёт',html_report(result,result['config']),'report.html','text/html')
