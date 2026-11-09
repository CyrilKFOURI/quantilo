@echo off
rem Quantilo: start the app on this computer (Windows). Usage: run, or run notebook
rem Installs uv if needed (no admin rights), which fetches Python and the libraries on first run.
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    if not exist "%USERPROFILE%\.local\bin\uv.exe" (
        echo Installing uv, first run only...
        powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    )
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

if /i "%~1"=="notebook" (
    rem The library in notebooks: the same environment, plus plotly and JupyterLab.
    echo Opening the notebooks, the first run downloads JupyterLab...
    uv run --project backend --frozen --no-dev --with plotly --with jupyterlab --with ipykernel jupyter lab packages\quantilo\examples\app_tour.ipynb
) else (
    echo Preparing Quantilo, the first run downloads the libraries ^(a few minutes^)...
    uv run --project backend --frozen --no-dev python backend\launch.py %*
)
