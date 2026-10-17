"""Módulo de ranking (foco do TCC) — roda depois de analise_exploratoria.py.

Gera, em data/analise/ranking/:
  funil_selecao.csv           fluxo de seleção (tipo PRISMA): Rede Federal -> amostra analítica
  funil_producao.csv          docentes -> com CV -> com produção -> com artigo -> com Qualis A
  titulacao_amostra.csv       quantos pós-doutores, doutores, mestres...
  cobertura_instituicoes.csv  IFs com mais / menos dados
  localizacao_capital_interior.csv, localizacao_teste.json
  areas_grande_area.csv, areas_detalhe.csv, areas_clusters.csv (agrupamento de áreas)
  titulacao_indicadores.csv
  regiao_padronizada.csv      desigualdade regional bruta x padronizada por titulação
  clusters_k4_k6.csv, clusters_associacao.csv (V de Cramér)
  score_perfil_pesos.csv      pesos 80/20 por perfil K-Means (k = 5)
  ranking_score_perfil.csv    ranking geral e dentro do perfil, pilares, pontos IPA-T por item
  subrankings_pilares.csv     top 100 de cada pilar
  ganha_perde.csv             quem ganha e quem perde ao trocar o índice
  ranking_campus_sp.csv, ranking_instituicoes_sp.csv
  insights.json               os três achados centrais + números de apoio
  fig/*.png
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analise_exploratoria import PESOS_IPA, concordancia, escala100

BASE = Path(__file__).parent
AN = BASE / "data" / "analise"
OUT = AN / "ranking"
FIG = OUT / "fig"
AZUL, LARANJA, CINZA = "#2a78d6", "#eb6834", "#8a8f98"
PILARES = {"F1": "Produção científica", "F2": "Difusão e extensão",
           "F3": "Gestão e ensino", "F4": "Pós-graduação e inovação"}
P_LIM = 0.25          # centróide acima disso = pilar dominante do perfil
PESO_PERFIL = 0.80    # 80% resultado do perfil, 20% complementar

CAPITAIS = {"AC": "rio branco", "AL": "maceio", "AM": "manaus", "AP": "macapa", "BA": "salvador",
            "CE": "fortaleza", "DF": "brasilia", "ES": "vitoria", "GO": "goiania", "MA": "sao luis",
            "MG": "belo horizonte", "MS": "campo grande", "MT": "cuiaba", "PA": "belem",
            "PB": "joao pessoa", "PE": "recife", "PI": "teresina", "PR": "curitiba",
            "RJ": "rio de janeiro", "RN": "natal", "RO": "porto velho", "RR": "boa vista",
            "RS": "porto alegre", "SC": "florianopolis", "SE": "aracaju", "SP": "sao paulo",
            "TO": "palmas"}
# bairros/unidades que ficam na capital mas não trazem o nome da cidade
NA_CAPITAL_EXTRA = {"pirituba", "sao miguel paulista", "ceilandia", "gama", "planaltina", "samambaia",
                    "sao sebastiao", "taguatinga", "riacho fundo", "recanto das emas", "estrutural",
                    "brasilia", "candangolandia", "itapoa", "lago norte", "sobradinho", "brazlandia",
                    "monte castelo", "maracana", "realengo", "duque de caxias"}  # DF inteiro é capital


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()


def cramer_v(x, y) -> float:
    t = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    n = t.values.sum()
    return float(np.sqrt(chi2 / (n * (min(t.shape) - 1))))


def salvar(df, nome, index=False):
    df.to_csv(OUT / nome, index=index, encoding="utf-8-sig")


# ------------------------------------------------------------------ localização
def classificar_localizacao(b: pd.DataFrame) -> pd.Series:
    campi = json.loads((BASE / "data" / "campi_integra.json").read_text(encoding="utf-8"))
    cidade = {}
    for sigla, _id, nome, end in campi:
        m = re.search(r"\|\s*([^|]+?)\s*-\s*([A-Z]{2})\s*\|", end or "")
        if m:
            cidade[(sigla, norm(nome))] = (norm(m.group(1)), m.group(2))

    def um(r):
        nc = norm(r["campus"])
        if not nc or nc == "nan":
            return "Sem campus"
        if "reitoria" in nc:
            return "Reitoria"
        uf = r["uf"]
        cap = CAPITAIS.get(uf, "#")
        if uf == "DF":
            return "Capital"
        c = cidade.get((r["sigla"], nc))
        if c:
            return "Capital" if c[0] == cap else "Interior"
        nome = nc.replace("campus ", "")
        if nome == cap or nome.startswith(cap + " ") or cap in nome or nome in NA_CAPITAL_EXTRA and uf in ("SP", "RJ"):
            return "Capital"
        return "Interior"
    return b.apply(um, axis=1)


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    res = json.loads((AN / "resumo_achados.json").read_text(encoding="utf-8"))
    b = pd.read_csv(AN / "base_docentes.csv", encoding="utf-8-sig", low_memory=False)
    cob = pd.read_csv(AN / "cobertura_extracao.csv", encoding="utf-8-sig")
    pes = pd.read_csv(BASE / "data" / "processed" / "pessoas_completo.csv", low_memory=False,
                      usecols=["sigla", "pessoa_uid", "tem_cv"])
    import config as CFG
    a = b[b["F1"].notna()].copy()          # amostra analítica (docentes efetivos, IFs incluídos)
    incl = sorted(a["sigla"].unique())
    a["localizacao"] = classificar_localizacao(a)
    b["localizacao"] = classificar_localizacao(b)

    # ============================================================ 1. FUNIL
    n_rede = 41  # 38 IFs + 2 CEFETs + UTFPR (Lei 11.892/2008, art. 1º) — Colégio Pedro II à parte
    n_config = len(CFG.INSTITUICOES)
    resp = cob[cob["extraidas"] > 0]
    funil = [
        ("Instituições da Rede Federal (IFs, CEFETs e UTFPR)", n_rede, "Lei 11.892/2008"),
        ("Instituições com endereço de portal Integra testado", n_config, "config.py"),
        ("Portais que responderam com dados de pessoas", int(len(resp)), "cobertura_extracao.csv"),
        ("Instituições incluídas (cobertura ≥ 50% e ≥ 100 servidores)", len(incl), "critério de inclusão"),
        ("Servidores listados nos portais das incluídas", int(cob[cob.sigla.isin(incl)]["total_portal"].sum()), "API pessoa/data"),
        ("Servidores extraídos", int(pes[pes.sigla.isin(incl)]["pessoa_uid"].nunique()), "data/raw"),
        ("Docentes (cargo de professor)", int(b[b.sigla.isin(incl)].shape[0]), "cargo/tipo"),
        ("Docentes efetivos (amostra analítica)", len(a), "situação funcional"),
    ]
    funil = pd.DataFrame(funil, columns=["etapa", "n", "fonte"])
    salvar(funil, "funil_selecao.csv")
    excl = cob[~cob.sigla.isin(incl)][["sigla", "extraidas", "total_portal"]].copy()
    motivos = {"IFPI": "portal passou a listar 0 pessoas", "IFMT": "portal com 9 pessoas (< 100)",
               "CEFETRJ": "portal sem resposta (erro 503)", "CEFETMG": "portal sem resposta",
               "IFES": "sem portal Integra ativo", "IFSUL": "portal vazio", "IFBAIANO": "portal vazio"}
    nao_resp = [s for s in CFG.INSTITUICOES if s not in set(cob.sigla)]
    excl = pd.concat([excl, pd.DataFrame({"sigla": nao_resp})], ignore_index=True)
    excl["motivo"] = excl["sigla"].map(motivos).fillna("portal sem dados disponíveis")
    excl["instituicao"] = excl["sigla"].map(lambda s: CFG.INSTITUICOES.get(s, (s,))[0])
    salvar(excl, "instituicoes_excluidas.csv")

    tem_cv = a["pessoa_uid"].isin(pes.loc[pes["tem_cv"] == True, "pessoa_uid"])  # noqa: E712
    prod_cols = ["artigos_total", "livros", "capitulos", "trab_completos_eventos", "resumos_eventos",
                 "trab_tecnicos", "software_patentes"]
    fp = [("Docentes efetivos", len(a)), ("Com currículo Lattes lido", int(tem_cv.sum())),
          ("Com alguma produção registrada", int((a[prod_cols].sum(axis=1) > 0).sum())),
          ("Com ao menos 1 artigo", int((a["artigos_total"] > 0).sum())),
          ("Com ao menos 1 artigo Qualis A", int((a["art_qualis_A"] > 0).sum())),
          ("Com orientação de mestrado concluída", int((a["orient_mestrado"] > 0).sum()))]
    fp = pd.DataFrame(fp, columns=["etapa", "docentes"])
    fp["pct"] = (100 * fp["docentes"] / len(a)).round(1)
    salvar(fp, "funil_producao.csv")
    totais = {"artigos": int(a["artigos_total"].sum()), "artigos_qualis_A": int(a["art_qualis_A"].sum()),
              "artigos_qualis_B": int(a["art_qualis_B"].sum()), "livros": int(a["livros"].sum()),
              "capitulos": int(a["capitulos"].sum()), "software_patentes": int(a["software_patentes"].sum()),
              "orient_mestrado": int(a["orient_mestrado"].sum()), "orient_doutorado": int(a["orient_doutorado"].sum())}
    tit = a["titulacao"].value_counts().rename_axis("titulacao").reset_index(name="docentes")
    tit["pct"] = (100 * tit["docentes"] / len(a)).round(1)
    salvar(tit, "titulacao_amostra.csv")

    # ============================================================ 2. COBERTURA POR IF
    g = a.groupby("sigla")
    ci = pd.DataFrame({
        "instituicao": g["instituicao"].first(), "uf": g["uf"].first(), "regiao": g["regiao"].first(),
        "servidores_portal": cob.set_index("sigla")["total_portal"],
        "docentes_efetivos": g.size(), "campi": g["campus"].nunique(),
        "pct_doutores": 100 * g["doutor"].mean(),
        "artigos_total": g["artigos_total"].sum(), "qualis_A_total": g["art_qualis_A"].sum(),
        "artigos_por_docente": g["artigos_total"].mean(),
        "pct_sem_producao": 100 * g.apply(lambda d: (d[prod_cols].sum(axis=1) == 0).mean()),
        "pct_lattes_mais_2anos": 100 * g["anos_sem_atualizar_lattes"].apply(lambda s: (s > 2).mean()),
        "pct_baixo_registro": 100 * g["perfil"].apply(lambda s: (s == "Baixo registro").mean()),
    }).dropna(subset=["docentes_efetivos"]).round(2)
    ci["registros_producao"] = g.apply(lambda d: d[prod_cols].sum().sum())
    ci = ci.sort_values("registros_producao", ascending=False)
    ci["posicao_volume_dados"] = range(1, len(ci) + 1)
    salvar(ci.reset_index(), "cobertura_instituicoes.csv")

    # ============================================================ 3. LOCALIZAÇÃO
    al = a[a["localizacao"].isin(["Capital", "Interior"])]
    vars_loc = ["doutor", "art_qualis_A", "artigos_total", "orient_mestrado", "extensao", "gestao", "IM", "IPA_T"]
    loc = al.groupby("localizacao")[vars_loc].mean().T.round(3)
    loc["razao_capital_interior"] = (loc["Capital"] / loc["Interior"]).round(2)
    loc.loc["docentes"] = [int((al.localizacao == "Capital").sum()), int((al.localizacao == "Interior").sum()), np.nan]
    # comparação controlando titulação (mesma titulação, capital x interior)
    loc_tit = al.groupby(["titulacao", "localizacao"])["art_qualis_A"].mean().unstack().round(2)
    loc_tit["n"] = al.groupby("titulacao").size()
    salvar(loc.reset_index(names="indicador"), "localizacao_capital_interior.csv")
    salvar(loc_tit.reset_index(), "localizacao_por_titulacao.csv")
    mw = stats.mannwhitneyu(al.loc[al.localizacao == "Capital", "IM"], al.loc[al.localizacao == "Interior", "IM"])
    pct_interior = 100 * (al.localizacao == "Interior").mean()

    # ============================================================ 4. ÁREAS
    vars_area = ["doutor", "art_qualis_A", "artigos_total", "livros", "orient_mestrado", "extensao",
                 "gestao", "software_patentes", "F1", "F2", "F3", "F4", "IM", "IPA_T"]
    ga = a.groupby("grande_area")[vars_area].mean()
    ga.insert(0, "docentes", a.groupby("grande_area").size())
    # Qualis A esperado pela titulação (padronização indireta) -> quem produz menos "de fato"
    esp_tit = a.groupby("titulacao")["art_qualis_A"].mean()
    a["qa_esperado_tit"] = a["titulacao"].map(esp_tit)
    ga["qualisA_obs_esp"] = a.groupby("grande_area").apply(
        lambda d: d["art_qualis_A"].sum() / d["qa_esperado_tit"].sum())
    ga = ga.sort_values("art_qualis_A").round(3)
    salvar(ga.reset_index(), "areas_grande_area.csv")
    ad = a.groupby(["grande_area", "area"])[vars_area].mean()
    ad.insert(0, "docentes", a.groupby(["grande_area", "area"]).size())
    ad = ad[ad["docentes"] >= 100].copy()
    ad["qualisA_obs_esp"] = a.groupby(["grande_area", "area"]).apply(
        lambda d: d["art_qualis_A"].sum() / d["qa_esperado_tit"].sum()).reindex(ad.index)
    Xa = ad[["F1", "F2", "F3", "F4"]].values
    melhor = None
    for k in range(3, 6):
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Xa)
        s = silhouette_score(Xa, km.labels_)
        if melhor is None or s > melhor[0]:
            melhor = (s, k, km)
    s_area, k_area, km_area = melhor
    ad["grupo_area"] = km_area.labels_
    cen = pd.DataFrame(km_area.cluster_centers_, columns=["F1", "F2", "F3", "F4"])

    def nome_grupo(r):
        dom = r.idxmax()
        if r.max() < 0.05:
            return "Áreas de menor registro"
        return {"F1": "Áreas de produção científica", "F2": "Áreas de difusão e extensão",
                "F3": "Áreas de gestão e ensino", "F4": "Áreas de pós-graduação e inovação"}[dom]
    nomes_g = cen.apply(nome_grupo, axis=1)
    if nomes_g.duplicated().any():
        nomes_g = [f"{n} ({i + 1})" if nomes_g.duplicated(keep=False)[i] else n for i, n in enumerate(nomes_g)]
    ad["grupo_area_nome"] = ad["grupo_area"].map(dict(enumerate(nomes_g)))
    ad = ad.round(3).sort_values(["grupo_area", "art_qualis_A"])
    salvar(ad.reset_index(), "areas_detalhe.csv")
    ac = ad.groupby("grupo_area_nome").agg(areas=("docentes", "size"), docentes=("docentes", "sum"),
                                           **{v: (v, "mean") for v in ["F1", "F2", "F3", "F4", "art_qualis_A", "doutor"]}).round(3)
    ac["exemplos"] = ad.reset_index().groupby("grupo_area_nome").apply(
        lambda d: "; ".join(d.sort_values("docentes", ascending=False)["area"].head(5)))
    salvar(ac.reset_index(), "areas_clusters.csv")

    # ============================================================ 5. TITULAÇÃO / REGIÃO
    ordem_tit = ["Graduação", "Aperfeiçoamento", "Especialização", "Mestrado", "Doutorado", "Pós-doutorado", "Sem informação"]
    tt = a.groupby("titulacao")[vars_area[1:] + ["anos_carreira"]].mean()
    tt.insert(0, "docentes", a.groupby("titulacao").size())
    tt = tt.reindex([t for t in ordem_tit if t in tt.index]).round(3)
    salvar(tt.reset_index(), "titulacao_indicadores.csv")
    # padronização direta por titulação: quanto da diferença regional some quando todos têm a mesma composição
    pesos_tit = a["titulacao"].value_counts(normalize=True)
    rg = []
    for r, d in a.groupby("regiao"):
        m = d.groupby("titulacao")["art_qualis_A"].mean()
        padron = (m.reindex(pesos_tit.index).fillna(d["art_qualis_A"].mean()) * pesos_tit).sum()
        rg.append({"regiao": r, "docentes": len(d), "pct_doutores": 100 * d["doutor"].mean(),
                   "qualisA_bruto": d["art_qualis_A"].mean(), "qualisA_padronizado_titulacao": padron,
                   "IM_medio": d["IM"].mean()})
    rg = pd.DataFrame(rg).round(3).sort_values("qualisA_bruto", ascending=False)
    amp_bruta = rg["qualisA_bruto"].max() - rg["qualisA_bruto"].min()
    amp_pad = rg["qualisA_padronizado_titulacao"].max() - rg["qualisA_padronizado_titulacao"].min()
    salvar(rg, "regiao_padronizada.csv")

    # ============================================================ 6. CLUSTERS k = 4..6
    F = a[["F1", "F2", "F3", "F4"]].values
    amostra = np.random.default_rng(1).choice(len(F), min(len(F), 10000), replace=False)
    labs, ks = {}, []
    for k in (4, 5, 6):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(F)
        labs[k] = km.labels_
        ks.append({"k": k, "silhouette": round(float(silhouette_score(F[amostra], km.labels_[amostra])), 3),
                   "inercia": round(float(km.inertia_), 1),
                   "tamanhos_pct": " / ".join(f"{x:.1f}" for x in sorted(100 * np.bincount(km.labels_) / len(F), reverse=True))})
    ks = pd.DataFrame(ks)
    ks["ARI_vs_k5"] = [round(adjusted_rand_score(labs[5], labs[k]), 3) for k in (4, 5, 6)]
    salvar(ks, "clusters_k4_k6.csv")
    assoc = []
    for v in ["regiao", "grande_area", "titulacao", "localizacao", "sigla"]:
        d = a[a[v].notna()]
        assoc.append({"variavel": v, "V_de_Cramer": round(cramer_v(d["perfil"], d[v]), 3)})
    assoc = pd.DataFrame(assoc).sort_values("V_de_Cramer", ascending=False)
    salvar(assoc, "clusters_associacao.csv")
    for v, nome in [("regiao", "perfil_x_regiao.csv"), ("titulacao", "perfil_x_titulacao.csv"),
                    ("grande_area", "perfil_x_area.csv"), ("localizacao", "perfil_x_localizacao.csv")]:
        salvar((100 * pd.crosstab(a[v], a["perfil"], normalize="index")).round(1).reset_index(), nome)

    # ============================================================ 7. SCORE POR PERFIL (80/20)
    # pilares = escalas somadas das variáveis salientes de cada fator (carga >= 0,40; Hair et al., 2010):
    # mais transparentes que o escore de regressão e zeram quem não tem registro
    cargas = pd.read_csv(AN / "fatorial_cargas.csv", encoding="utf-8-sig", index_col=0)[list(PILARES)]
    cargas = cargas.drop(index=["nivel_titulacao"], errors="ignore")
    dom = cargas.idxmax(axis=1); vmax = cargas.max(axis=1)
    vars_pilar = {f: [v for v in cargas.index if dom[v] == f and vmax[v] >= 0.40] for f in PILARES}
    salvar(pd.DataFrame([{"pilar": f, "nome": PILARES[f], "variaveis": ", ".join(v),
                          "cargas": ", ".join(f"{cargas.loc[x, f]:.2f}" for x in v)} for f, v in vars_pilar.items()]),
           "score_perfil_pilares_vars.csv")
    for f, vs in vars_pilar.items():
        Zp = np.log1p(a[vs].fillna(0).astype(float)); Zp = (Zp - Zp.mean()) / Zp.std()
        a[f"E_{f}"] = Zp.mean(axis=1)
        a[f"E_{f}"] = (a[f"E_{f}"] - a[f"E_{f}"].mean()) / a[f"E_{f}"].std()
        a[f"P_{f}"] = (100 * a[f"E_{f}"].rank(pct=True)).round(2)    # pilar 0–100 (percentil)
    cent = a.groupby("perfil")[[f"E_{f}" for f in PILARES]].mean()
    cent.columns = list(PILARES)
    pesos = []
    for perf, m in cent.iterrows():
        pos = m.clip(lower=0).where(m > P_LIM, 0)
        if pos.sum() == 0:                                           # perfil sem pilar dominante
            w = pd.Series(1 / 4, index=m.index)
            regra = "sem pilar dominante: pesos iguais"
        else:
            w = PESO_PERFIL * pos / pos.sum() + (1 - PESO_PERFIL) / 4
            regra = "80% pilares dominantes (proporcional ao centróide) + 20% igual entre os 4"
        pesos.append({"perfil": perf, **{f"peso_{k}": round(float(v), 4) for k, v in w.items()},
                      **{f"centroide_{k}": round(float(v), 3) for k, v in m.items()}, "regra": regra,
                      "pilares_dominantes": ", ".join(PILARES[k] for k in pos.index if pos[k] > 0) or "—"})
    pesos = pd.DataFrame(pesos)
    salvar(pesos, "score_perfil_pesos.csv")
    W = pesos.set_index("perfil")[[f"peso_{k}" for k in PILARES]]
    a["SP"] = sum(a[f"P_{k}"] * a["perfil"].map(W[f"peso_{k}"]) for k in PILARES).round(2)
    a["rank_SP"] = a["SP"].rank(ascending=False, method="min").astype(int)
    a["rank_SP_no_perfil"] = a.groupby("perfil")["SP"].rank(ascending=False, method="min").astype(int)
    a["n_perfil"] = a.groupby("perfil")["SP"].transform("size")
    for k in PILARES:
        a[f"rank_P_{k}"] = a[f"P_{k}"].rank(ascending=False, method="min").astype(int)
    a["desloc_IPAT_SP"] = a["rank_IPA_T"] - a["rank_SP"]
    a["desloc_IM_SP"] = a["rank_IM"] - a["rank_SP"]
    # pontos IPA-T por item (o "quanto vale" cada atividade do docente)
    pt = PESOS_IPA["IPA_T"]
    for v, w in pt.items():
        a[f"pts_{v}"] = (w * (a[v] if v == "nivel_titulacao" else np.log1p(a[v]))).round(3)
    pts_cols = [f"pts_{v}" for v in pt]
    a["pts_IPA_T_total"] = a[pts_cols].sum(axis=1).round(3)

    conc_sp = {f"SP x {n}": concordancia(a["SP"], a[c]) for c, n in
               [("IPA_T", "IPA-T"), ("IPA_R", "IPA-R"), ("IM", "IM"), ("indice_PC1", "PC1")]}
    cols_rank = (["rank_SP", "rank_SP_no_perfil", "n_perfil", "SP", "perfil", "nome", "sigla", "campus", "uf",
                  "regiao", "localizacao", "titulacao", "grande_area", "area", "anos_carreira",
                  "P_F1", "P_F2", "P_F3", "P_F4", "rank_P_F1", "rank_P_F2", "rank_P_F3", "rank_P_F4",
                  "IPA_T", "IPA_R", "IM", "rank_IPA_T", "rank_IPA_R", "rank_IM", "desloc_IPAT_SP", "desloc_IM_SP",
                  "art_qualis_A", "art_qualis_B", "artigos_total", "livros", "capitulos", "orient_mestrado",
                  "orient_doutorado", "software_patentes", "projetos", "extensao", "gestao", "atividades_ensino",
                  "pts_IPA_T_total"] + pts_cols + ["pessoa_uid", "numero_lattes"])
    rk = a.sort_values("rank_SP")[cols_rank]
    rk.to_csv(OUT / "ranking_score_perfil.csv", index=False, encoding="utf-8-sig")
    sub = []
    for k, nome in PILARES.items():
        t = a.nsmallest(100, f"rank_P_{k}")[["nome", "sigla", "campus", "perfil", "titulacao", f"P_{k}", k, "SP", "rank_SP"]]
        t.insert(0, "pilar", nome); t.insert(1, "posicao", range(1, len(t) + 1))
        sub.append(t.rename(columns={f"P_{k}": "pilar_0_100", k: "escore_fatorial"}))
    salvar(pd.concat(sub), "subrankings_pilares.csv")

    # ============================================================ 8. QUEM GANHA E QUEM PERDE
    gp = []
    for grupo in ["perfil", "regiao", "titulacao", "localizacao", "grande_area"]:
        for val, d in a.groupby(grupo):
            if len(d) < 50:
                continue
            top = lambda c: 100 * (d[c] <= len(a) * 0.10).mean()
            gp.append({"dimensao": grupo, "grupo": val, "docentes": len(d),
                       "desloc_mediano_IPAT_para_SP": float(d["desloc_IPAT_SP"].median()),
                       "desloc_mediano_IPAT_para_IM": float((d["rank_IPA_T"] - d["rank_IM"]).median()),
                       "pct_no_top10_IPA_T": round(top("rank_IPA_T"), 1), "pct_no_top10_IM": round(top("rank_IM"), 1),
                       "pct_no_top10_SP": round(top("rank_SP"), 1),
                       "pct_sobe_mais_10pct": round(100 * (d["desloc_IPAT_SP"] > 0.1 * len(a)).mean(), 1),
                       "pct_cai_mais_10pct": round(100 * (d["desloc_IPAT_SP"] < -0.1 * len(a)).mean(), 1)})
    gp = pd.DataFrame(gp)
    salvar(gp, "ganha_perde.csv")
    top10 = a[a["rank_SP"] <= 0.1 * len(a)]
    comp_top = pd.DataFrame({idx: 100 * a.loc[a[f"rank_{c}"] <= 0.1 * len(a), "perfil"].value_counts(normalize=True)
                             for idx, c in [("IPA-T", "IPA_T"), ("IPA-R", "IPA_R"), ("IM", "IM"), ("Score por Perfil", "SP")]}).round(1)
    salvar(comp_top.reset_index(names="perfil"), "top10_composicao_perfil.csv")

    # ============================================================ 9. CAMPUS / IF pelo SP
    am = a[~a["localizacao"].isin(["Reitoria", "Sem campus"])]
    rc = am.groupby(["sigla", "campus"]).agg(docentes=("SP", "size"), SP_medio=("SP", "mean"), IM_medio=("IM", "mean"),
                                             IPA_T_medio=("IPA_T", "mean"), pct_doutores=("doutor", "mean"),
                                             localizacao=("localizacao", "first"), uf=("uf", "first"),
                                             perfil_predominante=("perfil", lambda s: s.mode().iat[0]))
    rc = rc[rc["docentes"] >= 20].copy()
    for c in ["SP_medio", "IM_medio", "IPA_T_medio"]:
        rc[f"pos_{c}"] = rc[c].rank(ascending=False, method="min").astype(int)
    rc["pct_doutores"] *= 100
    rc = rc.round(2).sort_values("pos_SP_medio")
    salvar(rc.reset_index(), "ranking_campus_sp.csv")
    ri = a.groupby("sigla").agg(instituicao=("instituicao", "first"), docentes=("SP", "size"), SP_medio=("SP", "mean"),
                                IM_medio=("IM", "mean"), IPA_T_medio=("IPA_T", "mean"), IPA_R_medio=("IPA_R", "mean"),
                                pct_doutores=("doutor", "mean"))
    for c in ["SP_medio", "IM_medio", "IPA_T_medio", "IPA_R_medio"]:
        ri[f"pos_{c}"] = ri[c].rank(ascending=False, method="min").astype(int)
    ri["pct_doutores"] *= 100
    ri = ri.round(2).sort_values("pos_SP_medio")
    salvar(ri.reset_index(), "ranking_instituicoes_sp.csv")

    # ============================================================ 10. INSIGHTS
    rho_car = stats.spearmanr(a["anos_carreira"], a["art_qualis_A"], nan_policy="omit")[0]
    a["faixa_carreira"] = pd.cut(a["anos_carreira"], [0, 10, 20, 30, 80], labels=["até 10", "11–20", "21–30", "> 30"])
    car = a.groupby("faixa_carreira", observed=True).agg(docentes=("IM", "size"), qualisA_medio=("art_qualis_A", "mean"),
                                                        qualisA_mediana=("art_qualis_A", "median"),
                                                        pct_doutor=("doutor", "mean"), gestao=("gestao", "mean"),
                                                        IM=("IM", "mean"), SP=("SP", "mean")).round(3)
    car["pct_doutor"] *= 100
    salvar(car.reset_index(), "carreira_faixas.csv")
    sem_prod = 100 * (a[prod_cols].sum(axis=1) == 0).mean()
    ins = {
        "amostra": {"docentes": len(a), "instituicoes": len(incl), "campi": int(a["campus_completo"].nunique()),
                    "ufs": int(a["uf"].nunique()), "pct_interior": round(pct_interior, 1)},
        "totais_producao": totais,
        "insight_1_desigualdade_regional": {
            "texto": "A diferença entre regiões em artigos Qualis A cai quando a composição de titulação é igualada: "
                     "a desigualdade regional é, em boa parte, desigualdade de acesso ao doutorado.",
            "amplitude_bruta": round(amp_bruta, 3), "amplitude_padronizada": round(amp_pad, 3),
            "reducao_pct": round(100 * (1 - amp_pad / amp_bruta), 1),
            "regioes": rg.to_dict(orient="records"),
            "capital_x_interior": {k: v for k, v in loc["razao_capital_interior"].dropna().items()},
            "mann_whitney_IM_p": float(mw.pvalue)},
        "insight_2_carreira_longa": {
            "texto": "Tempo de carreira não se converte em mais produção Qualis A: a relação é fraca e, controlando "
                     "titulação, negativa (IRR do log dos anos de carreira < 1 na regressão).",
            "spearman_anos_carreira_qualisA": round(float(rho_car), 3),
            "IRR_log_anos_carreira": res.get("regressao", {}).get("IRR", {}).get("np.log(anos_carreira)"),
            "faixas": car.reset_index().astype({"faixa_carreira": str}).to_dict(orient="records")},
        "insight_3_um_terco_baixo_registro": {
            "texto": "Cerca de um terço dos docentes efetivos está no perfil de baixo registro: pouca ou nenhuma "
                     "atividade lançada no Lattes, o que é sub-registro, não necessariamente baixa atuação.",
            "pct_baixo_registro": round(100 * (a["perfil"] == "Baixo registro").mean(), 1),
            "pct_sem_producao": round(sem_prod, 1),
            "pct_lattes_mais_2anos": round(100 * (a["anos_sem_atualizar_lattes"] > 2).mean(), 1),
            "pct_baixo_registro_lattes_desatualizado": round(100 * (a.loc[a.perfil == "Baixo registro", "anos_sem_atualizar_lattes"] > 2).mean(), 1)},
        "areas": {"menor_qualisA": ga.index[0], "maior_qualisA": ga.index[-1],
                  "menor_obs_esp": ga["qualisA_obs_esp"].idxmin(), "maior_obs_esp": ga["qualisA_obs_esp"].idxmax(),
                  "grupos_areas": int(k_area), "silhouette_areas": round(float(s_area), 3)},
        "clusters": {"k_comparacao": ks.to_dict(orient="records"), "associacao": assoc.to_dict(orient="records")},
        "score_perfil": {"pesos": pesos.to_dict(orient="records"), "concordancia": conc_sp,
                         "top10_composicao": comp_top.to_dict()},
    }
    (OUT / "insights.json").write_text(json.dumps(ins, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # ============================================================ 11. FIGURAS
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    fu = pd.concat([funil.iloc[4:][["etapa", "n"]], fp.iloc[1:].rename(columns={"docentes": "n"})[["etapa", "n"]]])
    ax.barh(range(len(fu))[::-1], fu["n"], color=[AZUL] * 4 + [LARANJA] * (len(fu) - 4))
    for i, (e, n) in enumerate(zip(fu["etapa"], fu["n"])):
        ax.text(n + 500, len(fu) - 1 - i, f"{n:,}".replace(",", "."), va="center", fontsize=8)
    ax.set_yticks(range(len(fu))[::-1]); ax.set_yticklabels(fu["etapa"], fontsize=8)
    ax.set_xlabel("pessoas"); ax.set_xlim(0, fu["n"].max() * 1.15)
    ax.set_title("Funil de seleção e de registro da produção (Rede Federal, Portal Integra)", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "funil.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    c2 = ci.sort_values("docentes_efetivos")
    ax.barh(c2.index, c2["docentes_efetivos"], color=AZUL, label="docentes efetivos")
    ax2 = ax.twiny(); ax2.plot(c2["pct_doutores"], c2.index, "o", color=LARANJA, ms=3.5, label="% doutores")
    ax2.set_xlabel("% doutores", color=LARANJA); ax.set_xlabel("docentes efetivos")
    ax.tick_params(axis="y", labelsize=7); ax.set_title("Volume de dados por instituição", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "cobertura_instituicoes.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3))
    x = np.arange(len(rg))
    ax.bar(x - 0.2, rg["qualisA_bruto"], 0.4, color=AZUL, label="observado")
    ax.bar(x + 0.2, rg["qualisA_padronizado_titulacao"], 0.4, color=LARANJA, label="padronizado pela titulação")
    ax.set_xticks(x); ax.set_xticklabels(rg["regiao"]); ax.set_ylabel("artigos Qualis A por docente")
    ax.legend(frameon=False, fontsize=8); ax.set_title("Desigualdade regional antes e depois de igualar a titulação", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "regiao_padronizada.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    gq = ga.sort_values("qualisA_obs_esp")
    ax.barh(gq.index, gq["qualisA_obs_esp"], color=[LARANJA if v < 1 else AZUL for v in gq["qualisA_obs_esp"]])
    ax.axvline(1, color=CINZA, lw=0.8, ls="--"); ax.set_xlabel("Qualis A observado ÷ esperado pela titulação")
    ax.set_title("Grandes áreas: produção Qualis A ajustada pela titulação", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "areas_obs_esp.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    perfis = list(W.index); x = np.arange(len(perfis)); cores = [AZUL, LARANJA, "#4fa36b", "#8e5bc4"]
    base_ = np.zeros(len(perfis))
    for i, k in enumerate(PILARES):
        v = W[f"peso_{k}"].values * 100
        ax.bar(x, v, bottom=base_, color=cores[i], label=PILARES[k]); base_ += v
    ax.set_xticks(x); ax.set_xticklabels([p.replace(" ", "\n", 1) for p in perfis], fontsize=8)
    ax.set_ylabel("% do Score por Perfil"); ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title("Pesos do Score por Perfil (80% pilares dominantes + 20% complementar)", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "pesos_score_perfil.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ct = comp_top.reindex(perfis).fillna(0)
    x = np.arange(len(perfis)); w_ = 0.2
    for i, c in enumerate(ct.columns):
        ax.bar(x + (i - 1.5) * w_, ct[c], w_, label=c, color=[CINZA, "#b9c3cf", AZUL, LARANJA][i])
    ax.set_xticks(x); ax.set_xticklabels([p.replace(" ", "\n", 1) for p in perfis], fontsize=8)
    ax.set_ylabel("% do top 10%"); ax.legend(frameon=False, fontsize=7, ncol=4)
    ax.set_title("Quem ocupa o topo do ranking em cada índice", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "top10_por_indice.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    gpp = gp[gp.dimensao == "perfil"].set_index("grupo").reindex(perfis)
    x = np.arange(len(perfis))
    ax.bar(x - 0.2, gpp["desloc_mediano_IPAT_para_IM"], 0.4, color=AZUL, label="IPA-T → IM")
    ax.bar(x + 0.2, gpp["desloc_mediano_IPAT_para_SP"], 0.4, color=LARANJA, label="IPA-T → Score por Perfil")
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("posições (mediana; + = sobe)")
    ax.set_xticks(x); ax.set_xticklabels([p.replace(" ", "\n", 1) for p in perfis], fontsize=8)
    ax.legend(frameon=False, fontsize=8); ax.set_title("Quem ganha e quem perde ao trocar o índice", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "ganha_perde.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    smp = a.sample(min(len(a), 6000), random_state=1)
    ax.scatter(smp["rank_IPA_T"], smp["rank_SP"], s=2, alpha=0.25, color=AZUL)
    ax.plot([0, len(a)], [0, len(a)], color=LARANJA, lw=0.8)
    ax.set_xlabel("posição no IPA-T"); ax.set_ylabel("posição no Score por Perfil")
    ax.set_title(f"IPA-T × Score por Perfil (ρ = {conc_sp['SP x IPA-T']['spearman']:.2f})", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "dispersao_ipat_sp.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.plot(ks["k"], ks["silhouette"], "o-", color=AZUL)
    for _, r in ks.iterrows():
        ax.annotate(f"{r.silhouette:.3f}", (r.k, r.silhouette), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xticks([4, 5, 6]); ax.set_xlabel("número de perfis (k)"); ax.set_ylabel("silhouette")
    ax.set_title("Escolha do número de perfis (K-Means)", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "silhouette_k.png"); plt.close(fig)

    # base enriquecida para o dashboard
    extra = a[["pessoa_uid", "localizacao", "P_F1", "P_F2", "P_F3", "P_F4", "SP", "rank_SP", "rank_SP_no_perfil",
               "n_perfil", "desloc_IPAT_SP", "desloc_IM_SP", "pts_IPA_T_total"] + pts_cols]
    extra.to_csv(OUT / "score_perfil_docentes.csv", index=False, encoding="utf-8-sig")
    # base enxuta para o painel (dashboard.py)
    DASH = BASE / "data" / "dashboard"; DASH.mkdir(parents=True, exist_ok=True)
    keep = ["pessoa_uid", "nome", "numero_lattes", "sigla", "instituicao", "uf", "regiao", "campus", "localizacao",
            "titulacao", "nivel_titulacao", "grande_area", "area", "anos_carreira", "anos_sem_atualizar_lattes",
            "art_qualis_A", "art_A1", "art_A2", "art_A3", "art_A4", "art_qualis_B", "art_qualis_C_sem", "artigos_total",
            "livros", "capitulos", "trab_completos_eventos", "resumos_eventos", "trab_tecnicos", "software_patentes",
            "orient_IC_grad_outras", "orient_mestrado", "orient_doutorado", "orient_em_andamento", "bancas_trabalhos",
            "bancas_julgadoras", "atividades_ensino", "projetos", "extensao", "gestao", "material_didatico_cursos",
            "difusao_eventos", "participacao_eventos", "F1", "F2", "F3", "F4", "E_F1", "E_F2", "E_F3", "E_F4",
            "P_F1", "P_F2", "P_F3", "P_F4", "rank_P_F1", "rank_P_F2", "rank_P_F3", "rank_P_F4", "cluster", "perfil",
            "IPA_T", "IPA_R", "indice_PC1", "IM", "SP", "rank_IPA_T", "rank_IPA_R", "rank_indice_PC1", "rank_IM",
            "rank_SP", "rank_SP_no_perfil", "n_perfil", "pts_IPA_T_total"] + pts_cols
    a[keep].to_csv(DASH / "docentes.csv.gz", index=False, compression="gzip")
    print(json.dumps({k: ins[k] for k in ["amostra", "totais_producao"]}, ensure_ascii=False))
    print(funil.to_string(index=False)); print(fp.to_string(index=False)); print(tit.to_string(index=False))
    print(ci.head(5)[["docentes_efetivos", "registros_producao", "pct_doutores"]]); print(ci.tail(5)[["docentes_efetivos", "registros_producao", "pct_doutores"]])
    print(loc); print(loc_tit); print(rg.to_string(index=False)); print("redução", ins["insight_1_desigualdade_regional"]["reducao_pct"])
    print(ga[["docentes", "doutor", "art_qualis_A", "qualisA_obs_esp", "IM"]]); print(ac)
    print(ks); print(assoc); print(vars_pilar); print(pesos.to_string(index=False)); print(json.dumps(conc_sp, indent=0))
    print(comp_top); print(gp[gp.dimensao.isin(["perfil", "localizacao", "regiao"])].to_string(index=False)); print(car)
    print(json.dumps(ins["insight_3_um_terco_baixo_registro"], ensure_ascii=False)); print(ri.head(5)); print(rc.head(5))


if __name__ == "__main__":
    main()
