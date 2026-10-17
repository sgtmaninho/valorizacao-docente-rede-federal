"""Extrator multi-instituição do Portal Integra (Rede Federal).

Para cada instituição:
  1. lista todas as pessoas   -> /api/portfolio//pessoa/data?start=&length=
  2. currículo Lattes completo -> /api/portfolio/pessoa/s/<slug>
  3. indicadores (c/ Qualis)  -> /api/portfolio/indicador/pessoa/<id>/-1/-1

Salva um registro compacto por pessoa em data/raw/<SIGLA>.jsonl (append).
Se for interrompido, é só rodar de novo: quem já foi extraído é pulado.

Uso:
  python extractor.py                    # todas as instituições
  python extractor.py --inst IFB IFSP    # só algumas
  python extractor.py --limite 20        # teste rápido (20 pessoas por instituição)
  python extractor.py --sem-cv           # mais rápido: só indicadores (sem titulação)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

import config as C

# Usa o repositório de certificados do sistema (macOS/Windows), que completa cadeias
# intermediárias como o navegador faz — alguns portais Integra falham só no Python sem isso.
try:
    import truststore
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "truststore"], check=False)
    try:
        import truststore
    except ImportError:
        truststore = None
if truststore:
    truststore.inject_into_ssl()

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------- HTTP
def nova_sessao() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": C.USER_AGENT, "Accept": "application/json, text/plain, */*",
                      "Accept-Language": "pt-BR,pt;q=0.9"})
    adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=16)
    s.mount("https://", adapter)
    return s


ULTIMO_ERRO: dict = {}


def get_json(sess: requests.Session, url: str):
    for tentativa in range(1, C.MAX_TENTATIVAS + 1):
        try:
            r = sess.get(url, timeout=C.TIMEOUT)
            if r.status_code == 200:
                return r.json()
            ULTIMO_ERRO[url.split("/api/")[0]] = f"HTTP {r.status_code}"
            if r.status_code in (404, 400):
                return None
            espera = 2 ** tentativa + random.random()
        except (requests.RequestException, ValueError) as e:
            ULTIMO_ERRO[url.split("/api/")[0]] = f"{type(e).__name__}: {str(e)[:150]}"
            espera = 2 ** tentativa + random.random()
        time.sleep(espera)
    return None


# ---------------------------------------------------------------- Parsing do CV
IGNORAR = {"id", "autores", "palavrasChave", "areasDoConhecimento", "setoresDeAtividade",
           "informacoesAdicionais", "informacoesAdicionaisCursos",
           "informacoesAdicionaisInstituicoes", "participanteBanca",
           "participanteDeEventosCongressos"}


# Itens contados como 1 unidade mesmo sem "sequenciaProducao" (não descer dentro deles)
FOLHAS = {"participacaoEmProjeto", "vinculos", "ensino", "direcaoEAdministracao",
          "conselhoComissaoEConsultoria", "extensaoUniversitaria", "estagio",
          "servicoTecnicoEspecializado", "treinamentoMinistrado", "pesquisaEDesenvolvimento",
          "outraAtividadeTecnicoCientifica"}


def _contar(node, prefixo: str, out: dict, prof: int = 0) -> None:
    if prof > 10:
        return
    ultimo = prefixo.rsplit(".", 1)[-1]
    if isinstance(node, dict):
        for k, v in node.items():
            if k in IGNORAR or k.endswith("ByDate"):
                continue
            if isinstance(v, (list, dict)):
                _contar(v, f"{prefixo}.{k}" if prefixo else k, out, prof + 1)
    elif isinstance(node, list):
        out.setdefault(prefixo, 0)
        if not node:
            return
        dicts = [d for d in node if isinstance(d, dict)]
        wrapper = ultimo not in FOLHAS and dicts and all(
            "sequenciaProducao" not in d and any(isinstance(v, (list, dict)) and k not in IGNORAR
                                                 for k, v in d.items())
            for d in dicts
        )
        if wrapper:
            for d in dicts:
                _contar(d, prefixo, out, prof + 1)
        else:
            out[prefixo] += len(node)


ORDEM_TITULACAO = [  # (chave no Lattes, rótulo, nível)
    ("posDoutorado", "Pós-doutorado", 5),
    ("doutorado", "Doutorado", 4),
    ("mestrados", "Mestrado", 3),
    ("mestradoProfissionalizante", "Mestrado", 3),
    ("especializacoes", "Especialização", 2),
    ("aperfeicoamento", "Aperfeiçoamento", 1),
    ("graduacoes", "Graduação", 1),
]


def _ano(x):
    try:
        a = int(str(x)[:4])
        return a if 1940 < a <= datetime.now().year + 1 else None
    except (TypeError, ValueError):
        return None


def extrair_features_cv(cv: dict) -> dict:
    dg = cv.get("dadosGerais") or {}
    form = dg.get("formacaoAcademicaTitulacao") or {}
    f: dict = {}

    # Titulação máxima (concluída, quando há ano de conclusão/obtenção)
    f["titulacao"], f["nivel_titulacao"] = "Sem informação", 0
    for chave, rotulo, nivel in ORDEM_TITULACAO:
        itens = form.get(chave) or []
        concl = [i for i in itens if isinstance(i, dict) and
                 (_ano(i.get("anoDeConclusao")) or _ano(i.get("anoDeObtencaoDoTitulo"))
                  or chave == "posDoutorado")]
        if concl:
            f["titulacao"], f["nivel_titulacao"] = rotulo, nivel
            break
    for chave in ("graduacoes", "mestrados", "doutorado"):
        anos = [_ano(i.get("anoDeConclusao")) for i in (form.get(chave) or []) if isinstance(i, dict)]
        anos = [a for a in anos if a]
        f[f"ano_{chave}"] = min(anos) if anos else None
    f["n_formacoes"] = sum(len(v) for v in form.values() if isinstance(v, list))

    areas = ((dg.get("areasDeAtuacao") or {}).get("areaDeAtuacao") or [])
    if areas:
        def _seq(a):
            try:
                return int(a.get("sequenciaAreaDeAtuacao") or 99)
            except (TypeError, ValueError):
                return 99
        a0 = sorted([a for a in areas if isinstance(a, dict)] or [{}], key=_seq)[0]
        f["grande_area"] = (a0.get("nomeGrandeAreaDoConhecimento") or "").replace("_", " ").title()
        f["area"] = a0.get("nomeDaAreaDoConhecimento")
    f["orcid"] = bool(dg.get("orcidId") or dg.get("orcid_id"))
    f["situacao"] = cv.get("situacao")
    f["tipo"] = cv.get("tipo")
    f["data_atualizacao_lattes"] = cv.get("dataAtualizacao")

    cont: dict = {}
    for secao, apelido in (("producaoBibliografica", "pb"), ("producaoTecnica", "pt"),
                           ("outraProducao", "op"), ("dadosComplementares", "dc")):
        _contar(cv.get(secao) or {}, apelido, cont)
    _contar((dg.get("atuacoesProfissionais") or {}), "atu", cont)
    f["contagens"] = {k: v for k, v in cont.items() if v}
    return f


# ---------------------------------------------------------------- Extração
def testar_instituicao(sess, sigla) -> int | None:
    dados = get_json(sess, f"{C.base_url(sigla)}/api/portfolio//pessoa/data?start=0&length=1")
    try:
        return int(dados[0]["total"])
    except (TypeError, KeyError, IndexError, ValueError):
        return None


def listar_pessoas(sess, sigla, total, limite=None):
    pessoas, start, falhas = [], 0, 0
    alvo = min(total, limite) if limite else total
    while start < alvo and falhas < 8:
        url = f"{C.base_url(sigla)}/api/portfolio//pessoa/data?start={start}&length={C.PAGE_SIZE}"
        dados = get_json(sess, url)
        pagina = dados[1] if isinstance(dados, list) and len(dados) > 1 else None
        if not pagina:
            falhas += 1                      # página instável: pula e segue
            start += C.PAGE_SIZE // 2
            continue
        falhas = 0
        pessoas.extend(pagina)
        start += len(pagina)   # a API pode devolver menos que PAGE_SIZE por página
    # remove duplicatas (paginação instável)
    vistos, unicos = set(), []
    for p in pessoas:
        if p.get("slug") not in vistos:
            vistos.add(p.get("slug")); unicos.append(p)
    if len(unicos) < alvo:
        log(f"{sigla}: listagem parcial ({len(unicos)} de {alvo}) — rode de novo depois para completar")
    return unicos[:alvo]


def extrair_pessoa(sess, sigla, p, sem_cv=False) -> dict:
    base = C.base_url(sigla)
    reg = {
        "sigla": sigla,
        "pessoa_uid": f"{sigla}::{p.get('lattesId')}",
        "slug": p.get("slug"),
        "nome": p.get("nome"),
        "campus": p.get("campusNome"),
        "campus_id": p.get("campusId"),
        "cargo": p.get("cargo"),
        "id_integra": p.get("lattesId"),
        "numero_lattes": p.get("numeroIdentificador"),
        "atualizado_em": p.get("atualizadoEm"),
        "extraido_em": datetime.now().isoformat(timespec="seconds"),
    }
    ind = get_json(sess, f"{base}/api/portfolio/indicador/pessoa/{p.get('lattesId')}/-1/-1")
    if isinstance(ind, list) and len(ind) >= 2:
        reg["indicadores"] = {str(k): int(v) for k, v in zip(ind[0], ind[1])}
    else:
        reg["indicadores"] = None
    if not sem_cv and p.get("slug"):
        cv = get_json(sess, f"{base}/api/portfolio/pessoa/s/{p['slug']}")
        try:
            reg["cv"] = extrair_features_cv(cv) if isinstance(cv, dict) else None
        except Exception as e:  # noqa: BLE001 — um CV malformado não derruba a pessoa
            reg["cv"], reg["erro_cv"] = None, str(e)
    return reg


def ja_extraidos(sigla) -> set:
    arq = C.RAW_DIR / f"{sigla}.jsonl"
    feitos = set()
    if arq.exists():
        with arq.open(encoding="utf-8") as fh:
            for linha in fh:
                try:
                    feitos.add(json.loads(linha)["slug"])
                except (ValueError, KeyError):
                    pass
    return feitos


def extrair_instituicao(sigla, limite=None, sem_cv=False) -> dict:
    sess = nova_sessao()
    total = testar_instituicao(sess, sigla)
    if total is None:
        log(f"{sigla}: portal Integra indisponível — pulando ({ULTIMO_ERRO.get(C.base_url(sigla), 'sem resposta')})")
        return {"sigla": sigla, "status": "indisponivel"}
    log(f"{sigla}: {total} pessoas no portal")
    pessoas = listar_pessoas(sess, sigla, total, limite)
    feitos = ja_extraidos(sigla)
    pendentes = [p for p in pessoas if p.get("slug") not in feitos]
    log(f"{sigla}: {len(feitos)} já extraídas, {len(pendentes)} pendentes")

    arq = C.RAW_DIR / f"{sigla}.jsonl"
    if arq.exists() and arq.stat().st_size and not arq.read_bytes().endswith(b"\n"):
        with arq.open("a", encoding="utf-8") as fh:   # linha cortada por interrupção
            fh.write("\n")
    lock = threading.Lock()
    n_ok = 0
    with arq.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=C.WORKERS_POR_INST) as ex:
        futs = [ex.submit(extrair_pessoa, sess, sigla, p, sem_cv) for p in pendentes]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                reg = fut.result()
            except Exception as e:  # noqa: BLE001
                log(f"{sigla}: erro numa pessoa ({e})")
                continue
            with lock:
                fh.write(json.dumps(reg, ensure_ascii=False) + "\n")
                fh.flush()
            n_ok += 1
            if i % 100 == 0 or i == len(pendentes):
                log(f"{sigla}: {i}/{len(pendentes)}")
    return {"sigla": sigla, "status": "ok", "total_portal": total,
            "extraidos_agora": n_ok, "ja_existentes": len(feitos)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inst", nargs="*", help="siglas (padrão: todas)")
    ap.add_argument("--limite", type=int, help="máx. de pessoas por instituição (teste)")
    ap.add_argument("--sem-cv", action="store_true", help="não baixa o currículo completo")
    args = ap.parse_args()

    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    siglas = [s.upper() for s in args.inst] if args.inst else list(C.INSTITUICOES)
    desconhecidas = [s for s in siglas if s not in C.INSTITUICOES]
    if desconhecidas:
        sys.exit(f"Siglas desconhecidas: {desconhecidas}")

    t0 = time.time()
    log(f"Iniciando extração de {len(siglas)} instituições")
    resumo = []
    with ThreadPoolExecutor(max_workers=C.INSTS_PARALELAS) as ex:
        futs = {ex.submit(extrair_instituicao, s, args.limite, args.sem_cv): s for s in siglas}
        for fut in as_completed(futs):
            try:
                resumo.append(fut.result())
            except Exception as e:  # noqa: BLE001
                resumo.append({"sigla": futs[fut], "status": f"erro: {e}"})
                log(f"{futs[fut]}: erro geral ({e})")

    resumo.sort(key=lambda r: r["sigla"])
    (C.RAW_DIR / "_resumo_extracao.json").write_text(
        json.dumps({"finalizado_em": datetime.now().isoformat(timespec="seconds"),
                    "duracao_min": round((time.time() - t0) / 60, 1),
                    "instituicoes": resumo}, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = [r for r in resumo if r["status"] == "ok"]
    log(f"FIM em {(time.time() - t0) / 60:.1f} min — {len(ok)} instituições com dados, "
        f"{sum(r['extraidos_agora'] for r in ok)} pessoas extraídas nesta rodada")


if __name__ == "__main__":
    main()
