#!/bin/bash
# Dê dois cliques: (1) tenta de novo os portais que faltavam e (2) extrai as séries temporais (produção por ano).
cd "$(dirname "$0")"
echo "=== TCC – Atualização da base (Rede Federal) e séries temporais ==="
source ./_pysetup.sh
preparar_ambiente || { read -n1 -p "Pressione uma tecla..."; exit 1; }
"$PY" -m pip install -q -r requirements.txt 2>/dev/null
mkdir -p data/raw data/raw_series
echo "1) Portais que faltavam (em segundo plano): IFPI, IFMT, CEFET-RJ, CEFET-MG, IFES, IFSul, IFNMG, IFBaiano"
caffeinate -i "$PY" extractor.py --inst IFPI IFMT CEFETRJ CEFETMG IFES IFSUL IFNMG IFBAIANO >> data/extracao_v3.log 2>&1 &
PID_EXT=$!
echo "2) Séries temporais de todos os docentes efetivos (retoma de onde parou)"
caffeinate -i "$PY" extrair_series.py 2>&1 | tee -a data/series.log
wait $PID_EXT
echo "3) Séries dos institutos recém-extraídos"
caffeinate -i "$PY" extrair_series.py 2>&1 | tee -a data/series.log
echo; echo "Pronto. Pode fechar esta janela."
read -n1 -p "Pressione qualquer tecla para fechar..."
