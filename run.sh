#!/usr/bin/env sh
# Quantilo: start the app on this computer (macOS, Linux). Usage: ./run.sh, or ./run.sh notebook
# Installs uv if needed (no admin rights), which fetches Python and the libraries on first run.
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    if [ ! -x "$HOME/.local/bin/uv" ]; then
        echo "Installing uv, first run only..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    PATH="$HOME/.local/bin:$PATH"
fi

if [ "$1" = "notebook" ]; then
    # The library in notebooks: the same environment, plus plotly and JupyterLab.
    echo "Opening the notebooks, the first run downloads JupyterLab..."
    exec uv run --project backend --frozen --no-dev --with plotly --with jupyterlab --with ipykernel \
        jupyter lab packages/quantilo/examples/app_tour.ipynb
fi

echo "Preparing Quantilo, the first run downloads the libraries (a few minutes)..."
exec uv run --project backend --frozen --no-dev python backend/launch.py "$@"
