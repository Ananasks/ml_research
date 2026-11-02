"""Local dataset library and explicit, reproducible multi-dataset alignment."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import pandas as pd
from ml_lab.data import read_dataset, detect_time_columns, prepare_frame


def dataset_record(raw, name, settings=None):
    settings=dict(settings or {})
    frame=read_dataset(raw,name,**settings)
    identifier=hashlib.sha256(raw+json.dumps(settings,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return dict(id=identifier,name=Path(name).name,raw=raw,settings=settings,frame=frame,
                rows=len(frame),columns=frame.columns.tolist())


class DatasetLibrary:
    def __init__(self,root): self.root=Path(root)

    def list(self):
        if not self.root.exists(): return []
        records=[]
        for path in sorted(self.root.glob('*.json')):
            try:
                record=json.loads(path.read_text(encoding='utf-8'))
                if not isinstance(record,dict) or not isinstance(record.get('name'),str) or not isinstance(record.get('settings'),dict) or not isinstance(record.get('columns'),list) or not isinstance(record.get('rows'),int):
                    continue
                if record.get('id')==path.stem and len(path.stem)==64 and all(c in '0123456789abcdef' for c in path.stem) and (self.root/(path.stem+'.bin')).is_file():
                    records.append(record)
            except (OSError,ValueError,KeyError): continue
        return sorted(records,key=lambda r:r.get('created',''),reverse=True)

    def save(self,record):
        self.root.mkdir(parents=True,exist_ok=True)
        identifier=record['id']
        if len(identifier)!=64 or any(c not in '0123456789abcdef' for c in identifier): raise ValueError('Некорректный ID датасета.')
        destination=self.root/(identifier+'.json')
        if destination.exists(): return identifier
        metadata={k:record[k] for k in ['id','name','settings','rows','columns']}
        metadata['created']=datetime.now(timezone.utc).isoformat()
        # Publish metadata last, so readers never observe an incomplete saved dataset.
        for suffix,content in [('.bin',record['raw']),('.json',json.dumps(metadata,ensure_ascii=False,indent=2).encode())]:
            temp=self.root/(uuid4().hex+'.tmp')
            try:
                temp.write_bytes(content)
                os.replace(temp,self.root/(identifier+suffix))
            finally:
                if temp.exists(): temp.unlink()
        return identifier

    def load(self,identifier):
        metadata=next((r for r in self.list() if r['id']==identifier),None)
        if metadata is None: raise ValueError('Сохранённый dataset не найден.')
        record=dataset_record((self.root/(identifier+'.bin')).read_bytes(),metadata['name'],metadata['settings'])
        if record['id']!=identifier: raise ValueError('Содержимое сохранённого dataset изменено.')
        return record


def combine_datasets(records,method='datetime',dayfirst=False,drop_invalid_time=True):
    if not records: raise ValueError('Выберите хотя бы один dataset.')
    if len(records)==1: return records[0]['frame'].copy(),[]
    pieces=[]; diagnostics=[]
    primary_time = records[0]['frame'][detect_time_columns(records[0]['frame'])].reset_index(drop=True) if method=='rows' else None
    for record in records:
        frame=record['frame'].copy();frame.attrs={}
        time_columns=detect_time_columns(frame)
        if method=='datetime':
            if not time_columns: raise ValueError(f"В {record['name']} нет datetime или date + time. Для объединения по времени даты нужны во всех наборах.")
            frame=prepare_frame(frame,calendar=False,dayfirst=dayfirst,drop_invalid_time=drop_invalid_time)
            invalid=frame.attrs['invalid_time_rows']; frame.attrs={}
            if frame.index.has_duplicates:
                raise ValueError(f"В {record['name']} повторяются даты. Сначала устраните дубликаты, чтобы объединение не размножало строки.")
            if invalid: diagnostics.append(f"{record['name']}: исключено некорректных дат — {invalid}.")
        else: frame=frame.reset_index(drop=True)
        prefix=f"{record['name']} [{record['id'][:6]}]"
        drop=[c for c in frame if c.lower() in ('datetime','date','time','index','unnamed: 0')]
        frame=frame.drop(columns=drop)
        frame=frame.rename(columns={c:f'{prefix} · {c}' for c in frame})
        pieces.append(frame)
    base=pieces[0].copy()
    for record,piece in zip(records[1:],pieces[1:]):
        if method=='rows' and len(piece)!=len(base):
            raise ValueError('Для объединения по строкам количество строк должно совпадать во всех наборах.')
        if method=='datetime':
            matched=int(base.index.isin(piece.index).sum())
            diagnostics.append(f"{record['name']}: совпало по datetime {matched} из {len(base)} строк основного набора.")
            if not matched: raise ValueError(f"У {record['name']} нет общих дат с основным набором.")
        base=base.join(piece,how='left',validate='one_to_one')
    if method=='datetime': base.insert(0,'datetime',base.index)
    elif primary_time is not None:
        for column in reversed(primary_time.columns.tolist()): base.insert(0,column,primary_time[column].to_numpy())
    return base.reset_index(drop=True),diagnostics
