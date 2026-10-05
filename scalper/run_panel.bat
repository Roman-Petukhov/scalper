@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv (
  echo Создаю окружение Python...
  %PY% -m venv .venv || (echo Не найден Python 3.11+. Установите с python.org и отметьте "Add python.exe to PATH". & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
echo Проверяю зависимости...
python -m pip install -q --disable-pip-version-check -r requirements-trader.txt || (echo Не удалось установить зависимости. & pause & exit /b 1)
python -m trader.local
pause
