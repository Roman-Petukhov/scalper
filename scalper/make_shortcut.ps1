# Ярлык «Трендовые пробои» на рабочем столе: запускает панель и открывает её отдельным окном.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut((Join-Path $desktop "Трендовые пробои.lnk"))
$lnk.TargetPath = Join-Path $here "run_panel.bat"
$lnk.WorkingDirectory = $here
$lnk.IconLocation = (Join-Path $here "trader\web\static\icons\app.ico") + ",0"
$lnk.WindowStyle = 7
$lnk.Description = "Сигналы пробоя трендовых линий"
$lnk.Save()
