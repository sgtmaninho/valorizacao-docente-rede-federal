"""Extrai SÉRIES TEMPORAIS (contagem por ano) para os docentes efetivos já extraídos.

Para cada docente de data/raw/<SIGLA>.jsonl:
  1. /api/portfolio/indicador/pessoa/<id>/-1/-1  -> itens por rótulo (estrato Qualis etc.) com o ano
  2. /api/portfolio/pessoa/s/<slug>              -> ano de cada item do currículo (ensino, projetos,
                                                     orientações, bancas, eventos, extensão, gestão...)
Saída: data/raw_series/<SIGLA>.jsonl  (1 linha por docente; retoma de onde parou)

Uso:  python extrair_series.py            (todos)
      python extrair_series.py --inst IFB  (só alguns)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import config as C
import extractor as E

OUT = C.BASE_DIR / "data" / "raw_series"
ANO_MIN, ANO_MAX = 1980, datetime.now().year + 1
CHAVES_ANO = ["anoDoArtigo", "anoDoTrabalho", "anoDePublicacao", "anoDaPublicacao", "ano", "anoDeRealizacao",
              "anoDesenvolvimento", "anoDeConclusao", "anoInicio", "anoDeInicio"]
RX_ITEM = re.compile(r"\b([A-Z][A-Za-z]+)\(id=")
RX_ANO = re.compile(r"\b(ano[A-Za-z]*)=(\d{4})\b")


def efetivo(r: dict) -> bool:
    cargo = (r.get("cargo") or "").strip()
    sit = ((r.get("cv") or {}).get("situacao") or "").upper().replace("_", " ").strip()
    if not re.search(r"docente|professor", cargo, re.I):
        return False
    return "PERMANENTE" in sit or sit == "ATIVO" or (sit == "" and cargo == "Docente")


def _ano_valido(v):
    try:
        a = int(str(v)[:4])
        return a if ANO_MIN <= a <= ANO_MAX else None
    except (TypeError, ValueError):
        return None


# ---------------- indicadores (Qualis) por ano
def series_indicadores(ind) -> dict:
    """ind = [rótulos, contagens, textos]; cada texto lista os itens com seus campos ano*."""
    out = {}
    if not (isinstance(ind, list) and len(ind) >= 3):
        return out
    for rot, txt in zip(ind[0], ind[2]):
        txt = str(txt)
        anos = RX_ANO.findall(txt)
        if not anos:
            continue
        # o campo de ano mais frequente no rótulo é o ano do item (ex.: anoDoArtigo)
        chave = Counter(k for k, _ in anos).most_common(1)[0][0]
        cont = Counter(int(a) for k, a in anos if k == chave and _ano_valido(a))
        if cont:
            out[str(rot)] = {str(a): n for a, n in sorted(cont.items())}
    return out


# ---------------- currículo por ano (mesma lógica de folhas do extractor)
def _ano_do_item(d, prof=0):
    if not isinstance(d, dict) or prof > 3:
        return None
    for k in CHAVES_ANO:
        if k in d:
            a = _ano_valido(d[k])
            if a:
                return a
    for v in d.values():
        if isinstance(v, dict):
            a = _ano_do_item(v, prof + 1)
            if a:
                return a
        elif isinstance(v, list) and v and isinstance(v[0], dict) and prof < 2:
            a = _ano_do_item(v[0], prof + 1)
            if a:
                return a
    return None


def _walk(node, prefixo, out, prof=0):
    if prof > 10:
        return
    ultimo = prefixo.rsplit(".", 1)[-1]
    if isinstance(node, dict):
        for k, v in node.items():
            if k in E.IGNORAR or k.endswith("ByDate"):
                continue
            if isinstance(v, (list, dict)):
                _walk(v, f"{prefixo}.{k}" if prefixo else k, out, prof + 1)
    elif isinstance(node, list) and node:
        dicts = [d for d in node if isinstance(d, dict)]
        wrapper = ultimo not in E.FOLHAS and dicts and all(
            "sequenciaProducao" not in d and any(isinstance(v, (list, dict)) and k not in E.IGNORAR
                                                 for k, v in d.items()) for d in dicts)
        if wrapper:
            for d in dicts:
                _walk(d, prefixo, out, prof + 1)
        else:
            for d in dicts:
                a = _ano_do_item(d)
                if a:
                    out[prefixo][str(a)] += 1


def series_cv(cv: dict) -> dict:
    out = defaultdict(Counter)
    for secao, ap in (("producaoBibliografica", "pb"), ("producaoTecnica", "pt"),
                      ("outraProducao", "op"), ("dadosComplementares", "dc")):
        _walk(cv.get(secao) or {}, ap, out)
    _walk(((cv.get("dadosGerais") or {}).get("atuacoesProfissionais") or {}), "atu", out)
    return {k: dict(sorted(v.items())) for k, v in out.items() if v}


# ---------------- extração
def ja_feitos(sigla):
    arq = OUT / f"{sigla}.jsonl"
    s = set()
    if arq.exists():
        for l in arq.open(encoding="utf-8"):
            try:
                r = json.loads(l)
                if r.get("ok_ind") or r.get("ok_cv"):      # falhas (portal fora do ar) são tentadas de novo
                    s.add(r["pessoa_uid"])
            except (ValueError, KeyError):
                pass
    return s


def extrair_inst(sigla):
    raw = C.RAW_DIR / f"{sigla}.jsonl"
    if not raw.exists():
        return sigla, 0
    pessoas = []
    for l in raw.open(encoding="utf-8"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if efetivo(r):
            pessoas.append(r)
    feitos = ja_feitos(sigla)
    pend = [p for p in pessoas if p["pessoa_uid"] not in feitos]
    random.Random(42).shuffle(pend)   # ordem aleatória: qualquer extração parcial é uma amostra aleatória
    E.log(f"{sigla}: {len(pessoas)} docentes efetivos, {len(feitos)} já feitos, {len(pend)} pendentes")
    if not pend:
        return sigla, 0
    sess = E.nova_sessao()
    base = C.base_url(sigla)

    def uma(p):
        ind = E.get_json(sess, f"{base}/api/portfolio/indicador/pessoa/{p['id_integra']}/-1/-1")
        cv = E.get_json(sess, f"{base}/api/portfolio/pessoa/s/{p['slug']}") if p.get("slug") else None
        return {"pessoa_uid": p["pessoa_uid"], "sigla": sigla,
                "qualis_ano": series_indicadores(ind),
                "cv_ano": series_cv(cv) if isinstance(cv, dict) else None,
                "ok_ind": ind is not None, "ok_cv": isinstance(cv, dict)}

    lock, n = threading.Lock(), 0
    with (OUT / f"{sigla}.jsonl").open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=C.WORKERS_POR_INST) as ex:
        for i, fut in enumerate(as_completed([ex.submit(uma, p) for p in pend]), 1):
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                E.log(f"{sigla}: erro ({e})")
                continue
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            n += 1
            if i % 200 == 0 or i == len(pend):
                E.log(f"{sigla}: {i}/{len(pend)}")
    return sigla, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    siglas = [s.upper() for s in a.inst] if a.inst else sorted(p.stem for p in C.RAW_DIR.glob("*.jsonl"))
    t0 = time.time()
    E.log(f"Séries temporais: {len(siglas)} instituições")
    with ThreadPoolExecutor(max_workers=C.INSTS_PARALELAS) as ex:
        res = list(ex.map(extrair_inst, siglas))
    E.log(f"FIM em {(time.time() - t0) / 60:.1f} min — {sum(n for _, n in res)} docentes nesta rodada")


if __name__ == "__main__":
    main()
