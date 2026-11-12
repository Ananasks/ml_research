/** Reject the legacy API response before it reaches the signal-result renderer. */
export function requireChannelTransmission(value: unknown): void {
  if (value == null || typeof value !== 'object' || !('transmission' in value) ||
      value.transmission == null || typeof value.transmission !== 'object') {
    throw new Error('Python API возвращает прежний формат моделирования без сигнального тракта. Перезапустите сервер API и повторите расчёт.')
  }
}
