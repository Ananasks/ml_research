"""Plot data spans complete splits and only the winning validation configuration."""
import numpy as np
import pandas as pd


def winning_validation(result):
    frame = result.get('comparison', pd.DataFrame())
    if 'Модель' in frame and result.get('variant') is not None:
        frame = frame.loc[frame['Модель'].eq(f"{result['variant']} · {result['name']}")]
    return frame


def plot_records(frame, limit=6000):
    if frame is None or frame.empty:
        return []
    frame = frame.copy()
    frame.attrs = {}
    if 'Время' not in frame and isinstance(frame.index, pd.DatetimeIndex):
        frame.insert(0, 'Время', frame.index)
    if 'Время' in frame:
        frame = frame.sort_values('Время', kind='stable')
        times = pd.to_datetime(frame['Время'], utc=True)
        delta = times.diff()
        step = delta[delta > pd.Timedelta(0)].median()
        frame['_gapBefore'] = delta > 3 * step if pd.notna(step) else False
    elif 'Индекс' in frame:
        frame = frame.sort_values('Индекс', kind='stable')
    if len(frame) > limit:
        # Keep the full period and extrema of fact, prediction and baseline in each bin.
        fields = [c for c in ('Факт', 'Прогноз', 'Базовый прогноз') if c in frame]
        positions = {0, len(frame)-1}
        for block in np.array_split(np.arange(len(frame)), max(1, limit//(2*max(1,len(fields))))):
            for column in fields:
                values = pd.to_numeric(frame.iloc[block][column], errors='coerce').to_numpy()
                if np.isfinite(values).any():
                    positions.update((int(block[np.nanargmin(values)]), int(block[np.nanargmax(values)])))
            positions.add(int(block[0]))
        if '_gapBefore' in frame:
            for pos in np.flatnonzero(frame['_gapBefore'].to_numpy()):
                positions.update((max(0, int(pos)-1), int(pos)))
        frame = frame.iloc[sorted(positions)]
    return __import__('json').loads(frame.to_json(orient='records', date_format='iso'))


def result_series(result):
    frames = {'train': result.get('train_predictions'), 'validation': winning_validation(result),
              'test': result.get('predictions')}
    return {'splits': {name: plot_records(frame) for name, frame in frames.items()},
            'totals': {name: len(frame) if frame is not None else 0 for name, frame in frames.items()},
            'splitMode': result.get('config', {}).get('params', {}).get('split', 'time'),
            'timeConvention': 'UTC', 'version': 1}
