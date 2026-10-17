"""Séries temporais da atuação docente na Rede Federal.

Entrada : data/raw_series/*.jsonl (extrair_series.py) + data/analise/base_docentes.csv (analise_exploratoria.py)
Saída   : data/analise/series/*.csv + resumo_series.json

  painel_docente_ano.csv      docente x ano x variável (contagens por ano do item)
  serie_rede.csv              rede inteira: total, por 100 docentes "em carreira" e % de docentes ativos, por ano
  serie_regiao.csv            idem, por região
  serie_titulacao.csv         evolução da titulação do quadro atual (doutorados concluídos por ano)
  serie_perfil.csv            produção por perfil docente, por ano
  serie_instituicao.csv       Qualis A e índice multidimensional médio por instituto e triênio
  janelas_indices.csv         IPA-T, IPA-R e IM por docente em triênios (2014-16 ... 2023-25)
  janelas_concordancia.csv    concordância entre índices e estabilidade do ranking entre triênios

Observação: atividades de ensino, projetos, extensão e gestão entram pelo ANO DE INÍCIO do registro
(fluxo de novas atividades). A série vai até 2025; 2026 é parcial e fica de fora.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from analise_exploratoria import PESOS_IPA, indice_pesos_fixos, concordancia, escala100

SER = C.BASE_DIR / "data" / "raw_series"
ANA = C.BASE_DIR / "data" / "analise"
OUT = ANA / "series"
ANO_INI, ANO_FIM = 2000, 2025
JANELAS = [(2014, 2016), (2017, 2019), (2020, 2022), (2023, 2025)]

# rótulo do indicador (Qualis) -> variável
MAP_IND = [
    ("art_qualis_A", r"- A[1-4]$"), ("art_qualis_B", r"- B\d$"),
    ("art_qualis_C_sem", r"^Artigo.*(- C|sem Qualis|ISSN)$"), ("livros", r"^Livro"),
    ("capitulos", r"^Cap[ií]t"), ("trab_completos_eventos", r"completos em eventos"),
    ("resumos_eventos", r"resumo"), ("trab_tecnicos", r"Trab\. t[eé]cnicos"),
    ("software_patentes", r"software|Patente|cultivar|desenho"),
]
# chave do currículo -> variável
MAP_CV = [
    ("orient_IC_grad_outras", r"^op\.orientacoesConcluidas\.outras"),
    ("orient_mestrado", r"^op\.orientacoesConcluidas\.orientacoesConcluidasParaMestrado"),
    ("orient_em_andamento", r"^dc\.orientacoesEmAndamento"),
    ("bancas_trabalhos", r"participacaoEmBancaTrabalhosConclusao"),
    ("bancas_julgadoras", r"participacaoEmBancaJulgadora"),
    ("atividades_ensino", r"atividadesDeEnsino"),
    ("projetos", r"atividadesDeParticipacaoEmProjeto"),
    ("extensao", r"atividadesDeExtensao|formacaoComplementarDeExtensao"),
    ("gestao", r"atividadesDeDirecao|atividadesDeConselho"),
    ("material_didatico_cursos", r"desenvolvimentoDeMaterialDidatico|cursoDeCurtaDuracaoMinistrado"),
    ("difusao_eventos", r"apresentacaoDeTrabalho|organizacaoDeEvento|midiaSocial|programaDeRadio"),
    ("participacao_eventos", r"participacaoEmEventos"),
]
# produção pelo currículo (sem estrato Qualis): artigos para todos; demais só quando o portal não data os itens
MAP_CV_ART = [("artigos_cv", r"^pb\.artigosPublicados")]
MAP_CV_FALLBACK = [
    ("livros", r"^pb\.livrosECapitulos\.livrosPublicados"), ("capitulos", r"^pb\.livrosECapitulos\.capitulos"),
    ("trab_tecnicos", r"^pt\.trabalhoTecnico"), ("software_patentes", r"^pt\.(software|patente|cultivar|desenho)"),
]
MAP_IND = [(v, re.compile(p)) for v, p in MAP_IND]
MAP_CV = [(v, re.compile(p)) for v, p in MAP_CV]
MAP_CV_ART = [(v, re.compile(p)) for v, p in MAP_CV_ART]
MAP_CV_FALLBACK = [(v, re.compile(p)) for v, p in MAP_CV_FALLBACK]
VARS = [v for v, _ in MAP_IND] + [v for v, _ in MAP_CV] + ["artigos_cv"]
# variáveis que só existem com o estrato Qualis datado pelo portal (nem todos os portais datam os itens)
QVARS = ["art_qualis_A", "art_qualis_B", "art_qualis_C_sem", "trab_completos_eventos", "resumos_eventos"]
# variáveis-chave de cada dimensão (cargas >= 0,5 na análise fatorial) para o IM por janela
DIMENSOES = {
    "F1": ["art_qualis_A", "art_qualis_B", "art_qualis_C_sem", "resumos_eventos"],
    "F2": ["livros", "capitulos", "material_didatico_cursos", "difusao_eventos", "trab_tecnicos"],
    "F3": ["gestao", "atividades_ensino", "orient_IC_grad_outras", "bancas_julgadoras"],
    "F4": ["orient_mestrado", "software_patentes", "orient_em_andamento"],
}


def _var(chave, mapa):
    for v, rx in mapa:
        if rx.search(chave):
            return v
    return None


def carregar_painel(uids: set) -> pd.DataFrame:
    acc = defaultdict(int)
    for arq in sorted(SER.glob("*.jsonl")):
        for linha in arq.open(encoding="utf-8"):
            try:
                r = json.loads(linha)
            except ValueError:
                continue
            uid = r["pessoa_uid"]
            if uid not in uids:
                continue
            qa = r.get("qualis_ano") or {}
            for rot, anos in qa.items():
                v = _var(rot, MAP_IND)
                if v:
                    for a, n in anos.items():
                        acc[(uid, int(a), v)] += n
            mapas = MAP_CV + MAP_CV_ART + ([] if qa else MAP_CV_FALLBACK)
            for chave, anos in (r.get("cv_ano") or {}).items():
                v = _var(chave, mapas)
                if v:
                    for a, n in anos.items():
                        acc[(uid, int(a), v)] += n
    df = pd.DataFrame([(u, a, v, n) for (u, a, v), n in acc.items()], columns=["pessoa_uid", "ano", "variavel", "n"])
    return df[(df.ano >= 1980) & (df.ano <= ANO_FIM + 1)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(ANA / "base_docentes.csv", low_memory=False, encoding="utf-8-sig")
    base = base[base["efetivo"] & base["na_amostra_analitica"]].copy()
    feitos, com_qualis = set(), defaultdict(lambda: [0, 0])
    for arq in SER.glob("*.jsonl"):
        for linha in arq.open(encoding="utf-8"):
            try:
                r = json.loads(linha)
            except ValueError:
                continue
            if not (r.get("ok_ind") or r.get("ok_cv")):      # tentativa que falhou (portal fora do ar): não conta
                continue
            feitos.add(r["pessoa_uid"])
            com_qualis[r["sigla"]][0] += 1
            com_qualis[r["sigla"]][1] += bool(r.get("qualis_ano"))
    QIF = sorted(s_ for s_, (n, q) in com_qualis.items() if n and q / n >= 0.3)
    b = base[base["pessoa_uid"].isin(feitos)].copy()
    resumo = {"docentes_base": int(len(base)), "docentes_com_serie": int(len(b)),
              "cobertura_series_pct": round(100 * len(b) / len(base), 1),
              "instituicoes": int(b["sigla"].nunique()),
              "instituicoes_com_qualis_datado": QIF,
              "docentes_com_qualis_datado": int(b["sigla"].isin(QIF).sum())}
    cob = (b.groupby("sigla").size() / base.groupby("sigla").size()).fillna(0).round(3)
    cob.rename("cobertura_series").to_csv(OUT / "cobertura_series.csv")

    painel = carregar_painel(set(b["pessoa_uid"]))
    painel.to_csv(OUT / "painel_docente_ano.csv", index=False)
    anos = list(range(ANO_INI, ANO_FIM + 1))
    wide = painel[painel.ano.between(ANO_INI, ANO_FIM)].pivot_table(
        index=["pessoa_uid", "ano"], columns="variavel", values="n", aggfunc="sum", fill_value=0)
    wide = wide.reindex(columns=VARS, fill_value=0).reset_index()

    # exposição: docente "em carreira" no ano t = graduado até t-1 (ou sem informação -> conta)
    grad = b.set_index("pessoa_uid")["ano_graduacao"]
    attrs = b.set_index("pessoa_uid")[["sigla", "regiao", "titulacao", "perfil", "campus_completo"]]
    # pós-estratificação por instituição: cada docente com série "representa" 1/cobertura docentes do seu IF
    wgt = b.set_index("pessoa_uid")["sigla"].map(1 / cob.replace(0, np.nan)).fillna(1.0)
    resumo["ponderacao"] = "pós-estratificação por instituição (peso = 1 / cobertura da série no IF)"

    def serie(grupo_col=None):
        linhas = []
        grupos = [(None, b)] if grupo_col is None else list(b.groupby(grupo_col))
        for g, sub in grupos:
            uids = set(sub["pessoa_uid"])
            w = wide[wide.pessoa_uid.isin(uids)]
            gs = grad.reindex(list(uids)); ws = wgt.reindex(list(uids))
            uq = set(sub.loc[sub["sigla"].isin(QIF), "pessoa_uid"])
            qmask = pd.Series([u in uq for u in gs.index], index=gs.index)
            for t in anos:
                exp_mask = (gs.isna()) | (gs < t)
                expostos = float(ws[exp_mask].sum())
                wt = w[w.ano == t]
                pw = wt["pessoa_uid"].map(wgt).values
                row = {"ano": t, "docentes_em_carreira": int(exp_mask.sum()), "docentes_ponderados": round(expostos)}
                if grupo_col:
                    row[grupo_col] = g
                qm = np.array([u in uq for u in wt["pessoa_uid"]], dtype=bool)
                exp_q = float(ws[exp_mask & qmask].sum())
                for v in VARS:
                    vals, pwv, den = (wt[v].values[qm], pw[qm], exp_q) if v in QVARS else (wt[v].values, pw, expostos)
                    row[f"{v}_total"] = float(vals.sum())
                    tot = float((vals * pwv).sum())
                    row[f"{v}_por100"] = round(100 * tot / den, 2) if den else np.nan
                    row[f"{v}_pct_ativos"] = round(100 * float(((vals > 0) * pwv).sum()) / den, 2) if den else np.nan
                linhas.append(row)
        return pd.DataFrame(linhas)

    rede = serie()
    rede.to_csv(OUT / "serie_rede.csv", index=False)
    serie("regiao").to_csv(OUT / "serie_regiao.csv", index=False)
    serie("perfil").to_csv(OUT / "serie_perfil.csv", index=False)

    # titulação do quadro atual ao longo do tempo
    tit = []
    for t in range(1990, ANO_FIM + 1):
        em = b[(b["ano_graduacao"].isna()) | (b["ano_graduacao"] < t)]
        tit.append({"ano": t, "doutorados_concluidos": int((b["ano_doutorado"] == t).sum()),
                    "pct_doutores_no_quadro_atual": round(100 * float((b["ano_doutorado"] <= t).mean()), 2),
                    "docentes_em_carreira": int(len(em))})
    pd.DataFrame(tit).to_csv(OUT / "serie_titulacao.csv", index=False)

    # janelas (triênios): índices e estabilidade do ranking — só institutos com Qualis datado
    b_all, b = b, b[b["sigla"].isin(QIF)].copy()
    feats = []
    for (a0, a1) in JANELAS:
        w = wide[wide.ano.between(a0, a1)].groupby("pessoa_uid")[VARS].sum()
        w = w.reindex(b["pessoa_uid"]).fillna(0)
        w["nivel_titulacao"] = b.set_index("pessoa_uid")["nivel_titulacao"].reindex(w.index).fillna(0).values
        # titulação na época: doutorado concluído após a janela não conta
        anod = b.set_index("pessoa_uid")["ano_doutorado"].reindex(w.index)
        w.loc[(anod > a1) & (w["nivel_titulacao"] >= 4), "nivel_titulacao"] = 3
        out = pd.DataFrame(index=w.index)
        out["janela"] = f"{a0}-{a1}"
        for nome, pesos in PESOS_IPA.items():
            out[nome] = indice_pesos_fixos(w, pesos).values
        L = np.log1p(w[VARS])
        Z = (L - L.mean()) / L.std(ddof=0).replace(0, 1)
        dims = pd.DataFrame({f: Z[cols].mean(axis=1) for f, cols in DIMENSOES.items()})
        out[list(DIMENSOES)] = dims.values
        out["IM"] = escala100(dims.mean(axis=1)).values
        feats.append(out.join(attrs, how="left"))
    jan = pd.concat(feats).reset_index()
    jan.to_csv(OUT / "janelas_indices.csv", index=False)

    conc = []
    for j, sub in jan.groupby("janela"):
        for x, y in [("IPA_T", "IM"), ("IPA_R", "IM"), ("IPA_T", "IPA_R")]:
            conc.append({"janela": j, "comparacao": f"{x} x {y}", **concordancia(sub[x].reset_index(drop=True),
                                                                               sub[y].reset_index(drop=True))})
    nomes_j = [f"{a0}-{a1}" for a0, a1 in JANELAS]
    for c in ["IPA_T", "IPA_R", "IM"]:
        piv = jan.pivot_table(index="pessoa_uid", columns="janela", values=c)
        for j0, j1 in zip(nomes_j[:-1], nomes_j[1:]):
            conc.append({"janela": f"{j0} -> {j1}", "comparacao": f"estabilidade {c}",
                         **concordancia(piv[j0].reset_index(drop=True), piv[j1].reset_index(drop=True))})
    conc = pd.DataFrame(conc).round(3)
    conc.to_csv(OUT / "janelas_concordancia.csv", index=False)

    # composição do decil superior por perfil, em cada janela e índice
    comp = []
    for j, sub in jan.groupby("janela"):
        n10 = len(sub) // 10
        for c in ["IPA_T", "IPA_R", "IM"]:
            vc = 100 * sub.nlargest(n10, c)["perfil"].value_counts(normalize=True)
            comp.append({"janela": j, "indice": c, **vc.round(1).to_dict()})
    pd.DataFrame(comp).to_csv(OUT / "janelas_topo_perfil.csv", index=False)

    b = b_all
    # institutos: Qualis A por 100 docentes por ano e IM médio por janela (posição)
    ins = serie("sigla")[["sigla", "ano", "docentes_em_carreira", "art_qualis_A_por100", "extensao_por100",
                          "projetos_por100", "gestao_por100"]]
    ins.to_csv(OUT / "serie_instituicao_anual.csv", index=False)
    ij = jan.groupby(["janela", "sigla"])[["IM", "IPA_T", "IPA_R"]].mean().reset_index()
    for c in ["IM", "IPA_T", "IPA_R"]:
        ij[f"pos_{c}"] = ij.groupby("janela")[c].rank(ascending=False, method="min").astype(int)
    ij.round(3).to_csv(OUT / "serie_instituicao.csv", index=False)

    # resumo numérico
    r = rede.set_index("ano")
    def v(col, a):
        return float(r.loc[a, col])
    resumo["artigos_cv_por100"] = {a: v("artigos_cv_por100", a) for a in (2005, 2010, 2015, 2019, 2020, 2022, 2025)}
    resumo["gestao_por100"] = {a: v("gestao_por100", a) for a in (2010, 2015, 2017, 2019, 2022, 2025)}
    resumo["qualisA_por100"] = {a: v("art_qualis_A_por100", a) for a in (2005, 2010, 2015, 2019, 2020, 2022, 2025)}
    resumo["pct_ativos_qualisA"] = {a: v("art_qualis_A_pct_ativos", a) for a in (2010, 2015, 2019, 2020, 2025)}
    resumo["participacao_eventos_por100"] = {a: v("participacao_eventos_por100", a) for a in (2018, 2019, 2020, 2021, 2023)}
    resumo["extensao_por100"] = {a: v("extensao_por100", a) for a in (2010, 2015, 2019, 2020, 2025)}
    resumo["projetos_por100"] = {a: v("projetos_por100", a) for a in (2010, 2015, 2019, 2020, 2025)}
    t = pd.DataFrame(tit).set_index("ano")
    resumo["pct_doutores_quadro_atual"] = {a: float(t.loc[a, "pct_doutores_no_quadro_atual"]) for a in (2000, 2008, 2015, 2020, 2025)}
    resumo["janelas_concordancia"] = conc.to_dict(orient="records")
    (OUT / "resumo_series.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: resumo[k] for k in resumo if k != "janelas_concordancia"}, ensure_ascii=False, indent=1))
    print(conc.to_string(index=False))


if __name__ == "__main__":
    main()
