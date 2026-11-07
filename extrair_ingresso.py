"""Coleta o ANO DE INGRESSO NA REDE FEDERAL de cada docente efetivo (pedido do orientador, 06/11/2026).

Para cada docente de data/raw/<SIGLA>.jsonl busca o currículo (/api/portfolio/pessoa/s/<slug>) e guarda:
  - atuacoes: instituição + vínculos (ano de início/fim, tipo, enquadramento) de cada atuação profissional
  - cv_ano:   contagem por ano de cada item do currículo (mesma regra de extrair_series.py)
O ano de ingresso é decidido depois, em recorte_ingresso.py (sem precisar coletar de novo).

Saída: data/raw_ingresso/<SIGLA>.jsonl (1 linha por docente; retoma de onde parou)
Uso:  python extrair_ingresso.py            (todos)
      python extrair_ingresso.py --inst IFB  (só alguns)
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config as C
import extractor as E
from extrair_series import efetivo, series_cv

OUT = C.BASE_DIR / "data" / "raw_ingresso"


def _escalares(d: dict) -> dict:
    return {k: v for k, v in d.items() if isinstance(v, (str, int, float, bool)) and v not in ("", None)
            and k not in ("id",)}


def atuacoes(cv: dict) -> list:
    """Lista compacta das atuações profissionais: campos simples da atuação + vínculos."""
    ap = ((cv.get("dadosGerais") or {}).get("atuacoesProfissionais") or {})
    lst = ap.get("atuacaoProfissional") if isinstance(ap, dict) else ap
    if isinstance(lst, dict):
        lst = [lst]
    out = []
    for a in lst or []:
        if not isinstance(a, dict):
            continue
        reg = _escalares(a)
        vinc = a.get("vinculos") or []
        if isinstance(vinc, dict):
            vinc = [vinc]
        reg["vinculos"] = [_escalares(v) for v in vinc if isinstance(v, dict)]
        out.append(reg)
    return out


def ja_feitos(sigla):
    arq, s = OUT / f"{sigla}.jsonl", set()
    if arq.exists():
        for l in arq.open(encoding="utf-8"):
            try:
                r = json.loads(l)
                if r.get("ok_cv"):
                    s.add(r["pessoa_uid"])
            except (ValueError, KeyError):
                pass
    return s


def extrair_inst(sigla):
    raw = C.RAW_DIR / f"{sigla}.jsonl"
    if not raw.exists() or sigla.endswith("_cv_refeito"):
        return sigla, 0
    pessoas, vistos = [], set()
    for l in raw.open(encoding="utf-8"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if efetivo(r) and r.get("slug") and r["pessoa_uid"] not in vistos:
            vistos.add(r["pessoa_uid"]); pessoas.append(r)
    feitos = ja_feitos(sigla)
    pend = [p for p in pessoas if p["pessoa_uid"] not in feitos]
    random.Random(7).shuffle(pend)
    E.log(f"{sigla}: {len(pessoas)} docentes efetivos, {len(feitos)} já feitos, {len(pend)} pendentes")
    if not pend:
        return sigla, 0
    sess, base = E.nova_sessao(), C.base_url(sigla)

    def uma(p):
        cv = E.get_json(sess, f"{base}/api/portfolio/pessoa/s/{p['slug']}")
        ok = isinstance(cv, dict)
        return {"pessoa_uid": p["pessoa_uid"], "sigla": sigla, "ok_cv": ok,
                "atuacoes": atuacoes(cv) if ok else None,
                "cv_ano": series_cv(cv) if ok else None}

    lock, n = threading.Lock(), 0
    with (OUT / f"{sigla}.jsonl").open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=C.WORKERS_POR_INST) as ex:
        for i, fut in enumerate(as_completed([ex.submit(uma, p) for p in pend]), 1):
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                E.log(f"{sigla}: erro ({e})"); continue
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            n += rec["ok_cv"]
            if i % 200 == 0 or i == len(pend):
                E.log(f"{sigla}: {i}/{len(pend)}")
    return sigla, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", nargs="*")
    ap.add_argument("--limite", type=int, help="teste: só N docentes por instituição")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    siglas = [s.upper() for s in a.inst] if a.inst else sorted(
        p.stem for p in C.RAW_DIR.glob("*.jsonl") if not p.stem.endswith("_cv_refeito"))
    t0 = time.time()
    E.log(f"Ano de ingresso na Rede: {len(siglas)} instituições")
    with ThreadPoolExecutor(max_workers=C.INSTS_PARALELAS) as ex:
        res = list(ex.map(extrair_inst, siglas))
    E.log(f"FIM em {(time.time() - t0) / 60:.1f} min — {sum(n for _, n in res)} currículos nesta rodada")


if __name__ == "__main__":
    main()
