# ML Research Lab — Streamlit edition

Предыдущая полноценная версия ML Research Lab на Streamlit. Эта ветка сохраняет интерфейс приложения `app.py` и общий ML-движок; текущий интерфейс React находится в ветке `main`.

## Запуск

Создайте окружение, установите зависимости и запустите приложение:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

В Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

После запуска Streamlit откройте `http://localhost:8501/`.

Дополнительные библиотеки моделей устанавливаются отдельно:

```bash
pip install -r requirements-models.txt
```

Локальные результаты экспериментов хранятся в `.local/` и исключены из Git.
