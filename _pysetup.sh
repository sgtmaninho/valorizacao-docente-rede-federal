# Encontra um Python 3 que funcione e prepara o ambiente (.venv ou Python do conda)
achar_python() {
  local cands=()
  local conda_py
  conda_py=$(zsh -ic 'command -v python3 || command -v python' 2>/dev/null | tail -1)
  [ -n "$conda_py" ] && cands+=("$conda_py")
  cands+=("$HOME/anaconda3/bin/python3" "$HOME/opt/anaconda3/bin/python3" "$HOME/miniconda3/bin/python3" \
          "/opt/anaconda3/bin/python3" "/opt/homebrew/anaconda3/bin/python3" "/usr/local/bin/python3" \
          "/opt/homebrew/bin/python3" "/usr/bin/python3")
  for c in "${cands[@]}"; do
    if [ -x "$c" ] && "$c" -c "import sys, venv; assert sys.version_info>=(3,9)" 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}

preparar_ambiente() {
  if [ -x .venv/bin/python ] && .venv/bin/python -m pip --version >/dev/null 2>&1; then
    PY=.venv/bin/python; return 0
  fi
  rm -rf .venv
  local base
  base=$(achar_python) || { echo "Nenhum Python 3 encontrado. Instale em python.org"; return 1; }
  echo "Usando Python: $base"
  if "$base" -m venv --system-site-packages .venv 2>/dev/null && .venv/bin/python -m pip --version >/dev/null 2>&1; then
    PY=.venv/bin/python
  else
    rm -rf .venv
    echo "(venv indisponível — usando o Python diretamente)"
    PY="$base"
  fi
  echo "Instalando dependências (só na primeira vez)..."
  "$PY" -m pip install -q -r requirements.txt || "$PY" -m pip install -q --user -r requirements.txt
}
