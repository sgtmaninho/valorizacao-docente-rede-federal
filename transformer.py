"""Transforma data/raw/*.jsonl em tabelas analíticas (data/processed/).

Saídas:
  pessoas_completo.csv   uma linha por pessoa, todas as colunas (indicadores + contagens do CV)
  docentes.csv           só docentes, com as variáveis agregadas usadas na PCA/clusterização
  instituicoes.csv       agregados por instituição (médias, % doutores, volume)
"""
from __future__ import annotations

import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

import config as C

# Agrupamento dos rótulos do endpoint de indicadores (Integra) em dimensões
GRUPOS_INDICADORES = [
    ("art_qualis_A", re.compile(r"artigo.*-\s*A\d", re.I)),
    ("art_qualis_B", re.compile(r"artigo.*-\s*B\d", re.I)),
    ("art_outros", re.compile(r"artigo", re.I)),
    ("livros_capitulos", re.compile(r"livro|cap[ií]tulo", re.I)),
    ("trab_eventos", re.compile(r"evento|congresso|resumo", re.I)),
    ("prod_tecnica", re.compile(r"patente|software|produto|processo|t[eé]cnic|registro|marca|cultivar|desenho", re.I)),
]

VARIAVEIS_PCA = ["nivel_titulacao", "anos_carreira", "art_qualis_A", "art_qualis_B", "art_outros",
                 "livros_capitulos", "trab_eventos", "orientacoes", "bancas", "prod_tecnica",
                 "projetos", "extensao", "gestao"]


def grupo_do_rotulo(rotulo: str) -> str:
    for nome, rx in GRUPOS_INDICADORES:
        if rx.search(rotulo):
            return nome
    return "outros_indicadores"


def carregar_raw() -> list[dict]:
    regs = []
    for arq in sorted(C.RAW_DIR.glob("*.jsonl")):
        with arq.open(encoding="utf-8") as fh:
            for linha in fh:
                try:
                    regs.append(json.loads(linha))
                except ValueError:
                    pass
    return regs


def _soma(cont: dict, padrao: str) -> int:
    rx = re.compile(padrao, re.I)
    return int(sum(v for k, v in cont.items() if rx.search(k)))


def achatar(r: dict) -> dict:
    nome, uf, _ = C.INSTITUICOES.get(r["sigla"], (r["sigla"], "", ""))
    linha = {k: r.get(k) for k in ("sigla", "pessoa_uid", "slug", "nome", "campus", "cargo",
                                   "numero_lattes", "atualizado_em", "extraido_em")}
    linha["instituicao"], linha["uf"] = nome, uf
    linha["campus_completo"] = f"{r['sigla']} – {r.get('campus') or 'Sem campus'}"

    ind = r.get("indicadores") or {}
    for g, _ in GRUPOS_INDICADORES:
        linha[g] = 0
    linha["outros_indicadores"] = 0
    for rot, v in ind.items():
        linha[grupo_do_rotulo(rot)] += v
        linha[f"ind: {rot}"] = v
    linha["tem_indicadores"] = bool(ind)

    cv = r.get("cv") or {}
    cont = cv.get("contagens") or {}
    for k in ("titulacao", "nivel_titulacao", "grande_area", "area", "orcid", "situacao", "tipo",
              "ano_graduacoes", "ano_mestrados", "ano_doutorado", "data_atualizacao_lattes"):
        linha[k] = cv.get(k)
    linha["tem_cv"] = bool(cv)
    # Dimensões que o endpoint de indicadores não cobre: vêm do currículo
    linha["orientacoes"] = _soma(cont, r"^op\.orientacoesConcluidas")
    linha["bancas"] = _soma(cont, r"participacaoEmBanca")
    linha["projetos"] = _soma(cont, r"atividadesDeParticipacaoEmProjeto")
    linha["extensao"] = _soma(cont, r"atividadesDeExtensao")
    linha["gestao"] = _soma(cont, r"atividadesDeDirecao|atividadesDeConselho")
    linha["ensino_atividades"] = _soma(cont, r"atividadesDeEnsino")
    linha["orient_andamento"] = _soma(cont, r"orientacoesEmAndamento")
    linha["eventos_participacao"] = _soma(cont, r"participacaoEmEventos")
    # Se o endpoint de indicadores não trouxe algo, completa com o CV
    if not ind and cont:
        linha["art_outros"] = _soma(cont, r"^pb\.artigosPublicados")
        linha["livros_capitulos"] = _soma(cont, r"^pb\.livrosECapitulos")
        linha["trab_eventos"] = _soma(cont, r"^pb\.trabalhosEmEventos")
        linha["prod_tecnica"] = _soma(cont, r"^pt\.(?!demais)")
    for k, v in cont.items():
        linha[f"cv: {k}"] = v
    ano_ini = cv.get("ano_graduacoes")
    linha["anos_carreira"] = (datetime.now().year - ano_ini) if ano_ini else np.nan
    return linha


def eh_docente(df: pd.DataFrame) -> pd.Series:
    texto = (df["cargo"].fillna("") + " " + df["tipo"].fillna("")).str.lower()
    return texto.str.contains("docente|professor")


def main():
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    regs = carregar_raw()
    if not regs:
        raise SystemExit("Nenhum dado em data/raw — rode primeiro o extractor.py")
    df = pd.DataFrame([achatar(r) for r in regs]).drop_duplicates("pessoa_uid", keep="last")
    df.to_csv(C.PROCESSED_DIR / "pessoas_completo.csv", index=False)

    doc = df[eh_docente(df)].copy()
    doc["anos_carreira"] = doc["anos_carreira"].fillna(doc["anos_carreira"].median())
    doc["nivel_titulacao"] = doc["nivel_titulacao"].fillna(0)
    doc["producao_total"] = doc[["art_qualis_A", "art_qualis_B", "art_outros", "livros_capitulos",
                                 "trab_eventos", "prod_tecnica"]].sum(axis=1)
    base_cols = ["sigla", "instituicao", "uf", "campus", "campus_completo", "pessoa_uid", "nome",
                 "numero_lattes", "cargo", "titulacao", "grande_area", "area", "orcid",
                 "producao_total", "orient_andamento", "ensino_atividades", "eventos_participacao"]
    doc[base_cols + VARIAVEIS_PCA].to_csv(C.PROCESSED_DIR / "docentes.csv", index=False)

    inst = doc.groupby(["sigla", "instituicao", "uf"]).agg(
        docentes=("pessoa_uid", "count"),
        campi=("campus", "nunique"),
        pct_doutores=("nivel_titulacao", lambda s: 100 * (s >= 4).mean()),
        **{f"media_{v}": (v, "mean") for v in VARIAVEIS_PCA if v != "nivel_titulacao"},
    ).reset_index().round(2)
    inst.to_csv(C.PROCESSED_DIR / "instituicoes.csv", index=False)

    print(f"Pessoas: {len(df)} | Docentes: {len(doc)} | Instituições: {doc['sigla'].nunique()} "
          f"| Campi: {doc['campus_completo'].nunique()}")
    print(f"Arquivos gerados em {C.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
