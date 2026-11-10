"""TEC adapters reuse the dataset library and full saved ML predictions."""
from typing import Literal
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from ml_lab.tec_sources import prepare_tec, summarize_pair, serialize_series
from ml_lab.radio_channels.tec import ITU_SOURCE

router=APIRouter(prefix='/tec')


class Source(BaseModel):
    kind: Literal['dataset','run'] = 'dataset'
    dataset: str = ''
    time_column: str = ''
    value_column: str = ''
    unit: Literal['TECU','electrons/m2'] = 'TECU'
    tec_type: Literal['VTEC','STEC'] = 'VTEC'
    run_id: str = ''
    split: Literal['test','validation'] = 'test'
    run_field: Literal['Факт','Прогноз'] = 'Прогноз'


class AnalysisRequest(BaseModel):
    measured: Source | None = None
    predicted: Source | None = None
    carrier_mhz: float = Field(1575.42,ge=100,le=12000)
    bandwidth_mhz: float = Field(1,ge=.001,le=200)
    elevation: float = Field(60,ge=5,le=90)
    height_km: float = Field(350,ge=100,le=1000)


def resolve(source, request):
    from api.main import get_dataset, RUNS
    if source.kind == 'dataset':
        frame=get_dataset(source.dataset)
        time_col, value_col=source.time_column, source.value_column
        if time_col=='__auto__':
            from ml_lab.time_index import detect_time_columns
            columns=detect_time_columns(frame)
            if not columns:
                raise ValueError('Не найдены колонки datetime или date+time. Выберите колонку времени вручную.')
            frame=frame.copy()
            frame['__radio_datetime__']=frame[columns].astype(str).agg(' '.join,axis=1)
            time_col='__radio_datetime__'
        provenance={'kind':'dataset','dataset':source.dataset,'column':value_col,'time_column':time_col}
    else:
        run=RUNS.get(source.run_id)
        if run is None:
            raise ValueError('Сохранённый ML-запуск не найден. Выполните обучение в разделе «Модели и подбор».')
        if run.get('task') != 'Регрессия':
            raise ValueError('Для ПЭС нужен регрессионный прогноз.')
        from ml_lab.result_series import winning_validation
        frame=run.get('predictions') if source.split=='test' else winning_validation(run)
        if not isinstance(frame,pd.DataFrame) or frame.empty:
            raise ValueError('В выбранной части запуска нет прогнозов.')
        frame=frame.copy()
        if 'Время' not in frame and isinstance(frame.index,pd.DatetimeIndex):
            frame['Время']=frame.index
        time_col, value_col='Время',source.run_field
        provenance={'kind':'run','run_id':source.run_id,'model':run.get('name'),'target':run.get('target'),
                    'split':source.split,'field':source.run_field,'full_saved_predictions':True}
    clean, quality=prepare_tec(frame,time_col,value_col,source.unit,source.tec_type,request.elevation,request.height_km)
    return clean, {**provenance,'input_unit':source.unit,'tec_type':source.tec_type,'quality':quality}


@router.post('/analyze')
def analyze(request: AnalysisRequest):
    if request.bandwidth_mhz >= 2*request.carrier_mhz:
        raise HTTPException(422,'Полоса должна быть меньше удвоенной несущей.')
    if request.measured is None and request.predicted is None:
        raise HTTPException(422,'Выберите источник измерений или прогноза ПЭС.')
    try:
        frames, sources = {}, {}
        for name in ('measured','predicted'):
            source=getattr(request,name)
            if source:
                frames[name],sources[name]=resolve(source,request)
        fc,bw=request.carrier_mhz*1e6,request.bandwidth_mhz*1e6
        metrics=None; common=0; residual=None
        if len(frames)==2:
            pair,metrics=summarize_pair(frames['measured'],frames['predicted'],fc,bw)
            common=len(pair)
            residual_frame=pair[['time']].copy()
            residual_frame['stec']=np.abs(pair.stec_real-pair.stec_pred)
            residual_frame['tec']=residual_frame.stec
            residual_frame['rate']=np.nan
            residual=serialize_series(residual_frame,fc,bw)
        return {'config':request.model_dump(),'sources':sources,
            'series':{key:serialize_series(frame,fc,bw) for key,frame in frames.items()},
            'residual':residual, 'comparison':metrics,'alignment':{'common_rows':common,'method':'exact UTC timestamps; no interpolation',
                'unmatched':{key:len(frame)-common for key,frame in frames.items()} if common else {}},
            'model':'A: first-order cold plasma','source':ITU_SOURCE,
            'convention':'Re{x exp(+j2πft)}; transfer phase advance +K/f, phase range negative',
            'notes':['Производная — левая разность по соседним точкам; через разрывы >3 медианных шагов не рассчитывается.',
                'Наивные даты трактуются как UTC. Интерполяция и экстраполяция рядов не выполняются.',
                'Ошибки Доплера сравниваются на одинаковых интервалах общих временных меток.',
                'Полоса когерентности — дисперсионный критерий Bk=√(cf³/(80.5πNₜ)); S4 и быстрые замирания из ПЭС не выводятся.',
                'При Nₜ=0 полоса не ограничена этой моделью (null в JSON); остаточная полоса использует |STEC измерений − STEC прогноза| на общих UTC-метках.',
                'Геометрия — заданный угол места и тонкий сферический слой; эфемериды и магнитное поле не моделируются.']}
    except ValueError as e:
        raise HTTPException(422,str(e)) from e


@router.get('/runs')
def available_tec_runs():
    """Read app-created local snapshots; never accept uploaded serialized models."""
    from api.main import RUNS
    directory=getattr(RUNS,'directory',None)
    paths=sorted(directory.glob('*.joblib'),key=lambda p:p.stat().st_mtime,reverse=True)[:30] if directory else []
    results=[]
    for path in paths:
        run=RUNS.get(path.stem)
        if not run or run.get('task')!='Регрессия':
            continue
        frame=run.get('predictions')
        if not isinstance(frame,pd.DataFrame) or frame.empty:
            continue
        has_time='Время' in frame or isinstance(frame.index,pd.DatetimeIndex)
        results.append({'id':path.stem,'model':run.get('name','Модель'),'target':run.get('target',''),
            'timeAvailable':has_time,'rows':len(frame)})
    return results
