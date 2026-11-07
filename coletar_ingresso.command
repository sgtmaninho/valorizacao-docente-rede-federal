#!/bin/bash
# Dê dois cliques: coleta o ano de ingresso na Rede Federal de cada docente (várias horas; retoma de onde parou).
cd "$(dirname "$0")"
echo "=== TCC – Coleta do ano de ingresso na Rede Federal ==="
echo "Pode deixar rodando. Se fechar ou o Mac dormir, é só abrir de novo: continua de onde parou."
source ./_pysetup.sh
preparar_ambiente || { read -n1 -p "Pressione uma tecla..."; exit 1; }
mkdir -p data/raw_ingresso
caffeinate -i "$PY" extrair_ingresso.py "$@" 2>&1 | tee -a data/ingresso.log
echo; echo "Coleta concluída. Pode fechar esta janela."
read -n1 -p "Pressione qualquer tecla para fechar..."
