"""Busca de novo o currículo Lattes dos docentes cuja coleta falhou (cv vazio) e grava em data/raw/<SIGLA>_cv_refeito.jsonl.
Uso: python refazer_cv_faltantes.py IFSP   (padrão: todas as instituições com falhas)"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import config as C
import extractor as E

def faltantes(sigla):
    arq = C.RAW_DIR / f"{sigla}.jsonl"
    feitos = set()
    out = C.RAW_DIR / f"{sigla}_cv_refeito.jsonl"
    if out.exists():
        for l in out.open(encoding="utf-8"):
            try:
                r = json.loads(l)
                if r.get("cv"): feitos.add(r["pessoa_uid"])
            except ValueError: pass
    regs = []
    for l in arq.open(encoding="utf-8"):
        try: r = json.loads(l)
        except ValueError: continue
        if not r.get("cv") and r.get("slug") and r["pessoa_uid"] not in feitos:
            regs.append(r)
    return regs

def main():
    siglas = [s.upper() for s in sys.argv[1:]] or ["IFSP"]
    for sigla in siglas:
        regs = faltantes(sigla)
        E.log(f"{sigla}: {len(regs)} currículos para buscar de novo")
        sess = E.nova_sessao(); base = C.base_url(sigla)
        ok = 0
        with (C.RAW_DIR / f"{sigla}_cv_refeito.jsonl").open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=4) as ex:
            def uma(r):
                cv = E.get_json(sess, f"{base}/api/portfolio/pessoa/s/{r['slug']}")
                r = dict(r); r["cv"] = E.extrair_features_cv(cv) if isinstance(cv, dict) else None
                return r
            futs = [ex.submit(uma, r) for r in regs]
            for i, f in enumerate(as_completed(futs), 1):
                try: r = f.result()
                except Exception as e: E.log(f"erro: {e}"); continue
                if r.get("cv"):
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush(); ok += 1
                if i % 25 == 0 or i == len(regs): E.log(f"{sigla}: {i}/{len(regs)} (com CV: {ok})")
        E.log(f"{sigla}: concluído — {ok} currículos recuperados")

if __name__ == "__main__":
    main()
