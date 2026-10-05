@echo off
chcp 65001 >nul
cd /d "%~dp0"
where git >nul 2>nul && git rev-parse --is-inside-work-tree >nul 2>nul && (
  echo Обновляю панель с GitHub...
  git pull --ff-only --quiet || echo Обновиться не удалось ^(нет сети или есть свои правки^) — запускаю текущую версию.
)
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv (
  echo Создаю окружение Python...
  %PY% -m venv .venv || (echo Не найден Python 3.11+. Установите с python.org и отметьте "Add python.exe to PATH". & pause & exit /b 1)
)
if not exist "%USERPROFILE%\Desktop\Трендовые пробои.lnk" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_shortcut.ps1" >nul 2>nul && echo На рабочем столе появился ярлык «Трендовые пробои».
)
call .venv\Scripts\activate.bat
echo Проверяю зависимости...
python -m pip install -q --disable-pip-version-check -r requirements-trader.txt || (echo Не удалось установить зависимости. & pause & exit /b 1)
python -m trader.local
pause
