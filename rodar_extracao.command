#!/bin/bash
# Dê dois cliques neste arquivo no Finder para extrair os dados da Rede Federal.
cd "$(dirname "$0")"
echo "=== TCC – Extração de dados da Rede Federal (Portal Integra) ==="
source ./_pysetup.sh
preparar_ambiente || { read -n1 -p "Pressione uma tecla..."; exit 1; }
"$PY" -m pip install -q -r requirements.txt 2>/dev/null
mkdir -p data/raw data/processed
caffeinate -i "$PY" extractor.py "$@" 2>&1 | tee -a data/extracao.log
echo; echo "=== Processando dados (tabelas + PCA/clusters) ==="
"$PY" transformer.py && "$PY" analise_pca.py | tee data/processed/resumo_pca.txt && "$PY" analise_exploratoria.py > data/analise_log.txt
echo; echo "Pronto! Para ver o dashboard, dê dois cliques em abrir_dashboard.command"
read -n1 -p "Pressione qualquer tecla para fechar..."
