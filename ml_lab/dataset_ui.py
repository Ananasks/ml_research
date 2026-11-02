"""Dataset upload, optional persistence and analysis selection on the data page."""
import hashlib
import io
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from ml_lab.datasets import DatasetLibrary, dataset_record, combine_datasets
from ml_lab.data import read_dataset


def dataset_controls(root):
    library=DatasetLibrary(Path(root)/'.local'/'datasets')
    session=st.session_state.setdefault('datasets_session',{})
    examples={'Демонстрационный набор':'demo.csv','MUF · временной ряд':'muf.csv','TEC / CB · станции':'tec_cb.csv','Афины · ионосфера':'athens.csv'}
    with st.container(border=True):
        st.subheader('Загрузка данных')
        st.caption('Загрузите один или несколько наборов для анализа. Сохранение в библиотеку — по вашему выбору; временные наборы доступны только в текущей сессии.')
        uploads=st.file_uploader('Датасеты · CSV, TSV, Excel',type=['csv','tsv','xlsx'],accept_multiple_files=True,key='datasets_uploads')
        with st.expander('Настройки чтения загружаемых файлов'):
            a,b,c=st.columns(3)
            sep=a.selectbox('Разделитель CSV',['Автоматически',',',';','Табуляция','|'],key='datasets_separator')
            encoding=b.selectbox('Кодировка',['utf-8-sig','utf-8','cp1251','latin-1'],key='datasets_encoding')
            decimal=c.selectbox('Десятичный разделитель',['.',','],key='datasets_decimal')
            separator={'Автоматически':None,'Табуляция':'\t'}.get(sep,sep)
        uploaded_ids=[]
        for upload in uploads or []:
            try:
                raw=upload.getvalue()
                sheet=0
                if upload.name.lower().endswith('.xlsx'):
                    sheets=pd.ExcelFile(io.BytesIO(raw)).sheet_names
                    sheet=st.selectbox(f'Лист · {upload.name}',sheets,key='datasets_sheet_'+hashlib.sha256(raw).hexdigest())
                settings=dict(separator=separator,encoding=encoding,decimal=decimal,sheet=sheet)
                identifier=hashlib.sha256(raw+json.dumps(settings,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                if identifier not in session: session[identifier]=dataset_record(raw,upload.name,settings)
                uploaded_ids.append(identifier)
            except Exception as exc: st.error(f'{upload.name}: {exc}')
        saved={r['id']:r for r in library.list()}
        if session:
            st.markdown('**Наборы текущей сессии**')
            for identifier,record in list(session.items()):
                a,b=st.columns([4,1])
                a.write(f"**{record['name']}** · {record['rows']} строк · {len(record['columns'])} колонок")
                if identifier in saved: b.caption('✓ Сохранён')
                elif b.button('Сохранить',key='datasets_save_'+identifier):
                    try:
                        library.save(record)
                        saved={r['id']:r for r in library.list()}
                        st.success(f"{record['name']} сохранён в локальную библиотеку.")
                    except Exception as exc: st.error(f'Не удалось сохранить: {exc}')
            if uploaded_ids and st.button('Использовать загруженные наборы',key='datasets_use_uploads',type='primary'):
                st.session_state.datasets_source='Загруженные и сохранённые'
                st.session_state.datasets_selected=uploaded_ids
            if len(uploaded_ids)>1 and st.button('Сохранить все загруженные',key='datasets_save_all'):
                try:
                    for identifier in uploaded_ids: library.save(session[identifier])
                    saved={r['id']:r for r in library.list()}
                    st.success('Загруженные наборы сохранены в локальную библиотеку.')
                except Exception as exc: st.error(f'Не удалось сохранить все наборы: {exc}')
        st.divider()
        source=st.radio('Источник данных',['Пример из исследований','Загруженные и сохранённые'],horizontal=True,key='datasets_source')
        if source=='Пример из исследований':
            example=st.selectbox('Набор данных',list(examples),key='datasets_example')
            name=examples[example];raw=(Path(root)/'data'/name).read_bytes()
            return read_dataset(raw,name),name,hashlib.sha256(raw+name.encode()).hexdigest()
        available=saved|{key:{k:r[k] for k in ['id','name','rows','columns']} for key,r in session.items()}
        if not available:
            st.info('Загрузите файлы выше. Сохранённые наборы появятся здесь при следующем запуске.')
            return None,'',None
        label=lambda key:f"{available[key]['name']} [{key[:6]}] · {'сохранён' if key in saved else 'временно'}"
        if 'datasets_selected' in st.session_state:
            st.session_state.datasets_selected=[key for key in st.session_state.datasets_selected if key in available]
        selected=st.multiselect('Наборы для анализа',list(available),format_func=label,key='datasets_selected')
        if not selected:
            st.info('Выберите один или несколько наборов. Их колонки появятся ниже в выборе цели Y и признаков X.')
            return None,'',None
        try:
            records=[session[key] if key in session else library.load(key) for key in selected]
            method='datetime';dayfirst=False;drop_invalid=True
            if len(records)>1:
                primary=st.selectbox('Основной dataset · его строки сохраняем',selected,format_func=label,key='datasets_primary')
                records.sort(key=lambda r:r['id']!=primary)
                mode=st.radio('Как совместить наборы?',['По datetime','По номеру строки'],horizontal=True,key='datasets_join')
                method='datetime' if mode=='По datetime' else 'rows'
                if method=='datetime':
                    a,b=st.columns(2)
                    dayfirst=a.checkbox('При объединении день перед месяцем',key='datasets_dayfirst')
                    drop_invalid=b.checkbox('При объединении исключать плохие даты',True,key='datasets_drop_invalid')
                    st.caption('Точное совпадение datetime. Сохраняются строки основного набора; пропуски других наборов видны в предпросмотре. Интерполяция не выполняется.')
                elif not st.checkbox('Строки наборов соответствуют одним и тем же наблюдениям',key='datasets_rows_confirm'):
                    st.info('Проверьте соответствие наблюдений перед объединением по строкам.')
                    return None,'',None
            frame,diagnostics=combine_datasets(records,method,dayfirst,drop_invalid)
            for message in diagnostics: st.caption(message)
            if len(records)>1: st.caption('Имена колонок содержат название и ID исходного dataset. Ниже можно выбрать Y из одного набора, а X — из нескольких.')
            with st.expander('Состав выбранных наборов'):
                st.dataframe(pd.DataFrame([{'Dataset':r['name'],'Строк':r['rows'],'Колонки':', '.join(r['columns']),'Хранение':'Библиотека' if r['id'] in saved else 'Текущая сессия'} for r in records]),width='stretch',hide_index=True)
            signature=hashlib.sha256(repr(([r['id'] for r in records],method,dayfirst,drop_invalid)).encode()).hexdigest()
            name=records[0]['name'] if len(records)==1 else f'Сборка · {len(records)} наборов'
            return frame,name,signature
        except Exception as exc:
            st.error(f'Не удалось подготовить данные: {exc}')
            return None,'',None
