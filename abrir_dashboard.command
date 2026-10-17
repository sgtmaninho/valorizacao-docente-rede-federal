#!/bin/bash
# Dê dois cliques para abrir o dashboard no navegador.
cd "$(dirname "$0")"
source ./_pysetup.sh
preparar_ambiente || { read -n1; exit 1; }
"$PY" -m streamlit run dashboard.py
