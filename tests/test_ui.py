"""End-to-end Streamlit widget workflows (without a browser dependency)."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


def by_label(elements, label):
    return next(element for element in elements if element.label == label)


class InterfaceTests(unittest.TestCase):
    def test_examples_train_and_reset_previous_results(self):
        at = AppTest.from_file(APP).run(timeout=60)
        by_label(at.checkbox,'Использовать указанные начало и шаг для создания datetime').check().run()
        for name in ['Демонстрационный набор', 'MUF · временной ряд', 'TEC / CB · станции', 'Афины · ионосфера']:
            with self.subTest(dataset=name):
                by_label(at.selectbox, 'Набор данных').select(name).run(timeout=60)
                if name=='Демонстрационный набор':
                    by_label(at.checkbox,'Использовать указанные начало и шаг для создания datetime').check().run()
                self.assertFalse(at.exception)
                self.assertFalse(at.error)
                by_label(at.button, '▶ Обучить и оценить').click().run(timeout=60)
                self.assertFalse(at.exception)
                self.assertFalse(at.error)
                self.assertIn('metrics', at.session_state.result)
                self.assertTrue(at.session_state.result['sizes']['test'] > 1)
        self.assertEqual(len(at.session_state.history),4)
        # MUFD is an exact duplicate of MUF and must not be a default feature.
        self.assertNotIn('MUFD',at.session_state.result['features'])

    def test_generation_and_stale_result_indicator(self):
        at = AppTest.from_file(APP).run(timeout=60)
        by_label(at.checkbox,'Использовать указанные начало и шаг для создания datetime').check().run()
        by_label(at.selectbox,'Модель').select('Ridge').run()
        by_label(at.radio,'Режим настройки').set_value('Исследование диапазонов').run()
        by_label(at.button,'✧ Сгенерировать модели').click().run()
        self.assertEqual(len(at.session_state.candidates),5)
        by_label(at.button,'▶ Запустить поиск').click().run(timeout=60)
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertEqual(len(at.session_state.result['leaderboard']),5)
        by_label(at.number_input,'Random seed').set_value(123).run()
        self.assertTrue(any('Настройки изменены' in warning.value for warning in at.warning))
        self.assertTrue(by_label(at.button,'▶ Запустить поиск').disabled)

    def test_checkboxes_and_date_forecast(self):
        at = AppTest.from_file(APP).run(timeout=60)
        by_label(at.checkbox,'Использовать указанные начало и шаг для создания datetime').check().run()
        by_label(at.button,'Снять все').click().run()
        self.assertTrue(by_label(at.button,'▶ Обучить и оценить').disabled)
        by_label(at.button,'Выбрать все').click().run()
        self.assertFalse(by_label(at.button,'▶ Обучить и оценить').disabled)
        by_label(at.selectbox,'Набор данных').select('MUF · временной ряд').run(timeout=60)
        by_label(at.radio,'Период прогноза').set_value('Один день').run()
        day=by_label(at.date_input,'Дата прогноза').value
        by_label(at.button,'▶ Обучить и оценить').click().run(timeout=60)
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        dates=at.session_state.result['predictions']['Время'].dt.date
        self.assertTrue((dates==day).all())

    def test_lstm_controls_require_time_split(self):
        at = AppTest.from_file(APP).run(timeout=60)
        by_label(at.checkbox,'Использовать указанные начало и шаг для создания datetime').check().run()
        self.assertIn('LSTM',by_label(at.selectbox,'Модель').options)
        by_label(at.selectbox,'Набор данных').select('MUF · временной ряд').run(timeout=60)
        by_label(at.selectbox,'Модель').select('LSTM').run()
        self.assertFalse(at.exception)
        self.assertEqual(by_label(at.slider,'Рекуррентных слоёв').value,1)
        self.assertEqual(by_label(at.slider,'Окно последовательности · строк').value,12)
        self.assertEqual(by_label(at.selectbox,'Оптимизатор').value,'Adam')


if __name__ == '__main__': unittest.main()
