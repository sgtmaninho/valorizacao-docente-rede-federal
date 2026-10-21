"""Ranking institucional: agrega os docentes efetivos por região, instituição ou campus.

Fórmula do Score Institucional (SI) da unidade j:

    s_i   = Σ_k w_k · P_ik                  (escore do docente i: pilares em percentil 0–100, pesos w_k, Σ w_k = 1)
    SI_j  = (1 / n_j) · Σ_{i ∈ j} s_i       (média dos docentes efetivos da unidade)
    EP_j  = dp(s_i) / √n_j                  (erro-padrão)       IC95%_j = SI_j ± 1,96 · EP_j

Faixas: percorrendo o ranking, uma unidade fica na faixa do líder da faixa enquanto o limite superior do
seu IC95% alcança o limite inferior do IC95% do líder; senão abre uma nova faixa. Unidades da mesma faixa
não diferem de forma estatisticamente clara.

Com pesos iguais (w_k = 1/4), SI_j é também a média dos quatro pilares institucionais
PI_jk = média de P_ik na unidade, porque a média é linear.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PILARES = {"F1": "Produção científica", "F2": "Difusão e extensão", "F3": "Gestão e ensino",
           "F4": "Pós-graduação e inovação"}
PERFIS = ["Pesquisador-formador", "Pesquisador acadêmico", "Gestor-formador", "Extensionista-difusor", "Baixo registro"]
NIVEIS = {"Instituição": ["sigla"], "Campus": ["sigla", "campus"], "Região": ["regiao"]}
PESOS_IGUAIS = {k: 0.25 for k in PILARES}


def agregar(d: pd.DataFrame, nivel: str = "Instituição", pesos: dict | None = None, min_n: int = 1) -> pd.DataFrame:
    pesos = pesos or PESOS_IGUAIS
    tot = sum(pesos.values()) or 1
    w = {k: v / tot for k, v in pesos.items()}
    chaves = NIVEIS[nivel]
    x = d.copy()
    if nivel == "Campus":
        x = x[~x["localizacao"].isin(["Reitoria", "Sem campus"]) & x["campus"].notna()]
    x["_s"] = sum(x[f"P_{k}"] * w[k] for k in PILARES)
    g = x.groupby(chaves, dropna=True)
    a = g.agg(docentes=("_s", "size"), SI=("_s", "mean"), _dp=("_s", "std"),
              SP_medio=("SP", "mean"), IPA_T_medio=("IPA_T", "mean"), IPA_R_medio=("IPA_R", "mean"),
              IM_medio=("IM", "mean"),
              pct_doutores=("nivel_titulacao", lambda s: 100 * (s >= 4).mean()),
              qualisA_por_docente=("art_qualis_A", "mean"), artigos_por_docente=("artigos_total", "mean"),
              orient_mestrado_por_docente=("orient_mestrado", "mean"), extensao_por_docente=("extensao", "mean"),
              gestao_por_docente=("gestao", "mean"), software_por_docente=("software_patentes", "mean"),
              **{f"PI_{k}": (f"P_{k}", "mean") for k in PILARES})
    if nivel != "Região":
        a = a.join(g["regiao"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else None).rename("regiao_"))
        a = a.join(g["uf"].agg(lambda s: s.mode().iat[0]).rename("uf"))
    if nivel == "Campus":
        a = a.join(g["localizacao"].agg(lambda s: s.mode().iat[0]).rename("localizacao"))
    comp = (100 * pd.crosstab([x[c] for c in chaves], x["perfil"], normalize="index")).reindex(columns=PERFIS, fill_value=0)
    comp.columns = [f"pct_{p}" for p in comp.columns]
    a = a.join(comp)
    a["perfil_predominante"] = comp.idxmax(axis=1).str.replace("pct_", "", regex=False)
    a = a[a["docentes"] >= min_n].copy()
    a["EP"] = (a["_dp"] / np.sqrt(a["docentes"])).fillna(0)
    a["IC_inf"], a["IC_sup"] = a["SI"] - 1.96 * a["EP"], a["SI"] + 1.96 * a["EP"]
    pis = a[[f"PI_{k}" for k in PILARES]]
    a["equilibrio"] = 100 - pis.std(axis=1, ddof=0)          # 100 = pilares iguais
    a["pilar_forte"] = pis.idxmax(axis=1).str[3:].map(PILARES)
    for c in ["SI", "SP_medio", "IPA_T_medio", "IPA_R_medio", "IM_medio"] + [f"PI_{k}" for k in PILARES]:
        a[f"pos_{c}"] = a[c].rank(ascending=False, method="min").astype(int)
    a = a.sort_values("SI", ascending=False)
    # faixas de desempenho: a unidade entra na faixa do líder enquanto seu IC95% alcança o IC do líder
    faixa, lider_inf, f = [], None, 0
    for inf, sup in zip(a["IC_inf"], a["IC_sup"]):
        if lider_inf is None or sup < lider_inf:
            f += 1; lider_inf = inf
        faixa.append(f)
    a["faixa"] = faixa
    a = a.drop(columns="_dp").reset_index()
    if "regiao_" in a:
        a = a.rename(columns={"regiao_": "regiao"})
    return a.round(2)


def unidade_nome(a: pd.DataFrame, nivel: str) -> pd.Series:
    if nivel == "Campus":
        return a["sigla"] + " – " + a["campus"].astype(str)
    return a[NIVEIS[nivel][0]].astype(str)
