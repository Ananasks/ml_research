"""Durable metadata journal and portable, escaped research reports."""
import html
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import pandas as pd


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if hasattr(value, 'item'): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


class ExperimentStore:
    def __init__(self, path): self.path = Path(path)

    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute('CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, created TEXT, payload TEXT)')
        return connection

    def save(self, result, config):
        identifier = uuid4().hex
        record = clean(dict(id=identifier, created=datetime.now(timezone.utc).isoformat(),
            model=result['name'], task=result['task'], target=result['target'], features=result['features'],
            metrics=result['metrics'], validation=result['validation'], sizes=result['sizes'], evaluation_sizes=result.get('evaluation_sizes'), config=config))
        with self.connect() as connection:
            connection.execute('INSERT INTO experiments VALUES (?, ?, ?)',
                (identifier, record['created'], json.dumps(record, ensure_ascii=False, allow_nan=False)))
        return record

    def list(self):
        with self.connect() as connection:
            rows = connection.execute('SELECT payload FROM experiments ORDER BY created DESC, id DESC').fetchall()
        return [json.loads(row[0]) for row in rows]

    def clear(self):
        """Delete journal records only; dataset files and current results are untouched."""
        with self.connect() as connection:
            count=connection.execute('SELECT COUNT(*) FROM experiments').fetchone()[0]
            connection.execute('DELETE FROM experiments')
        return count


def html_report(result, config):
    escape = lambda value: html.escape(str(value))
    sections = [('Test', pd.DataFrame([result['metrics']])),
                ('Validation · сравнение кандидатов', pd.DataFrame(result['leaderboard'])),
                ('Размеры выборок', pd.DataFrame([result['sizes']])),
                ('Прогнозы теста', result['predictions'])]
    if result.get('evaluation_sizes'):
        sections.insert(3, ('Число исходных измерений для метрик',pd.DataFrame([result['evaluation_sizes']])))
    body = ''.join('<h2>'+escape(title)+'</h2>'+frame.to_html(index=False, escape=True) for title,frame in sections)
    return ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>ML Research</title>'
        '<style>body{font:15px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#182536}'
        'table{border-collapse:collapse;width:100%;overflow-wrap:anywhere}td,th{border:1px solid #ddd;padding:8px;text-align:left}'
        'pre{white-space:pre-wrap}h1,h2{color:#176878}</style><h1>ML Research</h1><p>'
        +escape(result['name'])+' · '+escape(result['target'])+'</p>'
        '<p>Победитель выбран по validation. Финальные метрики относятся к test. '
        'Восстановленные Y не входят в метрики validation/test: используются исходные измерения. '
        'Сравнивайте запуски только при одинаковых данных и разбиении.</p>'+body+'<h2>Конфигурация</h2><pre>'
        +escape(json.dumps(clean(config),ensure_ascii=False,indent=2,allow_nan=False))+'</pre></html>').encode('utf-8')
