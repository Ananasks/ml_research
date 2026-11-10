"""Matched packet comparisons at actual UTC observations; never stretches a packet to a day."""
from datetime import timezone
from pydantic import BaseModel, Field, AwareDatetime
from ml_lab.radio_simulator import RadioConfig, experiment


class DailySample(BaseModel):
    time: AwareDatetime
    measured: float | None = Field(None, ge=0, le=2000)
    predicted: float | None = Field(None, ge=0, le=2000)
    measured_rate: float = Field(0, ge=-100, le=100)
    predicted_rate: float = Field(0, ge=-100, le=100)


class DailyRequest(BaseModel):
    config: RadioConfig
    samples: list[DailySample] = Field(min_length=1, max_length=2048)


def compare_daily(request: DailyRequest):
    rows = []
    samples = sorted([s.model_copy(update={'time': s.time.astimezone(timezone.utc)}) for s in request.samples], key=lambda s: s.time)
    if len({s.time for s in samples}) != len(samples):
        raise ValueError('Временные метки должны быть уникальны.')
    if any(s.measured is None and s.predicted is None for s in samples):
        raise ValueError('В каждой точке нужен реальный или прогнозный ПЭС.')
    for sample in samples:
        seed = (request.config.seed + int(sample.time.timestamp())) % 2147483648
        for case in ('measured', 'clarke', 'predicted'):
            value = sample.measured if case == 'clarke' else getattr(sample, case, None)
            if value is None:
                continue
            rate = sample.measured_rate if case == 'clarke' else getattr(sample, case + '_rate', 0)
            cfg = request.config.model_copy(update={
                'model': 'combined' if case == 'clarke' else 'tec',
                'tec_stec_tecu': value or 0, 'tec_rate_tecu_s': rate,
                'compensation_stec_tecu': sample.predicted,
                'compensation_rate_tecu_s': sample.predicted_rate,
                'channel_estimation': 'ideal' if request.config.channel_estimation == 'forecast' and sample.predicted is None else request.config.channel_estimation,
                'seed': seed, 'sweep': False, 'realizations': 1})
            score = experiment(cfg, compact=True)
            rows.append({'time': sample.time.isoformat(), 'day': sample.time.date().isoformat(),
                         'case': case, 'model': cfg.model, 'stec_tecu':cfg.tec_stec_tecu, 'seed': seed, **score})
    summaries = []
    for day in sorted({r['day'] for r in rows}) + ['all']:
        for case in ('measured', 'clarke', 'predicted'):
            selected = [r for r in rows if r['case'] == case and (day == 'all' or r['day'] == day)]
            if not selected:
                continue
            bits = sum(r['metrics']['bits'] for r in selected)
            errors = sum(r['metrics']['errors'] for r in selected)
            evm = 100*(sum(r['error_energy'] for r in selected)/sum(r['reference_energy'] for r in selected))**.5
            summaries.append({'day': day, 'case': case, 'packets': len(selected),
                              'bits': bits, 'errors': errors, 'ber': errors/bits, 'evm_percent': evm})
    return {'rows': rows, 'summaries': summaries, 'config': request.config.model_dump(),
            'notes': ['Один короткий пакет в каждой исходной UTC-точке; непрерывная суточная реализация не моделируется.',
                      'В одной точке у трёх случаев одинаковые символы, шум и seed. Между точками пакеты независимы.',
                      'Итоговая BER — сумма ошибок / сумма битов; EVM — корень из отношения суммарной энергии ошибки к суммарной энергии опорных символов.',
                      'Сценарии 1 и 3 используют физический слой измеренного и прогнозного ПЭС; сценарий 2 добавляет Кларка к реальному ПЭС. Эхо OFDM задаётся общими настройками.',
                      'При точной компенсации фазовые эффекты ПЭС могут устраняться полностью: совпадение BER допустимо.',
                      'Каждая точка использует одну реализацию; настройка числа реализаций относится к детальному пакету.']}
