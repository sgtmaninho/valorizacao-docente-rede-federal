#!/bin/bash
cd "$(dirname "$0")"
echo "=== Recuperar currículos que falharam na coleta (IFSP) ==="
source ./_pysetup.sh
preparar_ambiente || { read -n1; exit 1; }
caffeinate -i "$PY" refazer_cv_faltantes.py IFSP 2>&1 | tee -a data/refazer_cv.log
read -n1 -p "Pressione qualquer tecla para fechar..."
