#!/bin/bash
# Dê dois cliques depois de terminar a coleta do ingresso: recalcula o cenário "desde o ingresso" e o integral.
cd "$(dirname "$0")"
source ./_pysetup.sh
preparar_ambiente || { read -n1; exit 1; }
"$PY" recorte_ingresso.py
for C in integral ingresso; do
  echo "=== Cenário $C ==="
  TCC_CENARIO=$C "$PY" transformer.py && TCC_CENARIO=$C "$PY" analise_exploratoria.py > data/analise_log_$C.txt && TCC_CENARIO=$C "$PY" analise_ranking.py > /dev/null
done
echo "Pronto. Abra o painel com abrir_dashboard.command"; read -n1 -p "Pressione qualquer tecla para fechar..."
