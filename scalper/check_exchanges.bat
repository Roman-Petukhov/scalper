@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv (
  echo Создаю окружение Python...
  %PY% -m venv .venv || (echo Не найден Python 3.11+. & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install -q --disable-pip-version-check -r requirements-trader.txt || (echo Не удалось установить зависимости. & pause & exit /b 1)
echo Проверяю доступ к биржам с этого компьютера...
echo.
python -m trader.check
echo.
pause
