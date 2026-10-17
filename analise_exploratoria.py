"""Análise exploratória e multivariada da valorização docente na Rede Federal.

Entrada : data/processed/pessoas_completo.csv  (gerado pelo transformer.py)
Saída   : data/analise/*.csv  + resumo_achados.json

  base_docentes.csv            1 linha por docente, todas as variáveis analíticas + escores
  dicionario_dados.csv         descrição de cada coluna da base
  descritivas.csv              média, mediana, desvio, % zeros, Gini, participação do top 10%
  correlacao_spearman.csv      matriz de correlação (Spearman) entre as variáveis
  correlacoes_destaque.csv     pares mais correlacionados
  fatorial_cargas.csv          análise fatorial (varimax) — cargas
  fatorial_variancia.csv       variância explicada por fator
  clusters_perfil.csv          perfil médio de cada cluster (K-Means sobre os fatores)
  testes_grupos.csv            Kruskal-Wallis por titulação / região / grande área / instituição
  medias_por_grupo.csv         medianas e médias por grupo
  regressao_qualisA.csv        binomial negativa: fatores associados a artigos Qualis A
  instituicoes.csv             indicadores agregados por instituição
  ranking_ipa_vs_fatores.csv   quanto o ranking muda entre pesos fixos e pesos dos dados
"""
from __future__ import annotations

import json
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import config as CFG

warnings.filterwarnings("ignore")
OUT = CFG.BASE_DIR / "data" / "analise"
ANO = datetime.now().year
REGIAO = {**dict.fromkeys(["AC", "AP", "AM", "PA", "RO", "RR", "TO"], "Norte"),
          **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Nordeste"),
          **dict.fromkeys(["DF", "GO", "MT", "MS"], "Centro-Oeste"),
          **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Sudeste"),
          **dict.fromkeys(["PR", "RS", "SC"], "Sul")}
ORDEM_TIT = ["Graduação", "Aperfeiçoamento", "Especialização", "Mestrado", "Doutorado", "Pós-doutorado"]

# ------------------------------------------------------------------ variáveis
DIC: dict[str, str] = {}


def col(df, padrao, fonte="cv"):
    """Soma das colunas do CV/indicadores cujo nome casa com o padrão (regex)."""
    pref = "cv: " if fonte == "cv" else "ind: "
    rx = re.compile(padrao)
    cols = [c for c in df.columns if c.startswith(pref) and rx.search(c[len(pref):])]
    return df[cols].fillna(0).sum(axis=1) if cols else pd.Series(0, index=df.index)


def construir_base(p: pd.DataFrame) -> pd.DataFrame:
    doc = p[p["cargo"].fillna("").str.contains("Docente|Professor", case=False)].copy()
    d = pd.DataFrame(index=doc.index)
    ident = ["pessoa_uid", "nome", "numero_lattes", "sigla", "instituicao", "uf", "campus",
             "campus_completo", "cargo", "situacao", "titulacao", "grande_area", "area"]
    for c in ident:
        d[c] = doc[c]
    d["regiao"] = d["uf"].map(REGIAO)
    sit = doc["situacao"].fillna("").str.upper().str.replace("_", " ").str.strip()
    # formatos variam entre portais: "ATIVO PERMANENTE", "ativo_permanente", "Ativo" (IFAC só usa "Ativo")
    d["efetivo"] = sit.str.contains("PERMANENTE") | (sit == "ATIVO") | \
        ((sit == "") & (doc["cargo"].fillna("").str.strip() == "Docente"))   # portal sem situação (IFSertãoPE)
    d["nivel_titulacao"] = doc["nivel_titulacao"].fillna(0)
    d["doutor"] = (d["nivel_titulacao"] >= 4).astype(int)
    d["orcid"] = doc["orcid"].fillna(False).astype(bool).astype(int)
    d["ano_graduacao"] = doc["ano_graduacoes"]
    d["ano_doutorado"] = doc["ano_doutorado"]
    d["anos_carreira"] = (ANO - doc["ano_graduacoes"]).clip(lower=1)
    d["anos_pos_doutorado"] = (ANO - doc["ano_doutorado"]).clip(lower=0)
    atual = pd.to_datetime(doc["data_atualizacao_lattes"], errors="coerce")
    d["anos_sem_atualizar_lattes"] = ((pd.Timestamp.now() - atual).dt.days / 365.25).round(2)

    # Pesquisa (endpoint de indicadores, com Qualis)
    d["art_A1"] = col(doc, r"- A1$", "ind"); d["art_A2"] = col(doc, r"- A2$", "ind")
    d["art_A3"] = col(doc, r"- A3$", "ind"); d["art_A4"] = col(doc, r"- A4$", "ind")
    d["art_qualis_A"] = d[["art_A1", "art_A2", "art_A3", "art_A4"]].sum(axis=1)
    d["art_qualis_B"] = col(doc, r"- B\d$", "ind")
    d["art_qualis_C_sem"] = col(doc, r"^Artigo.*(- C|sem Qualis|ISSN)$", "ind")
    d["artigos_total"] = d["art_qualis_A"] + d["art_qualis_B"] + d["art_qualis_C_sem"]
    d["livros"] = col(doc, r"^Livro", "ind")
    d["capitulos"] = col(doc, r"^Cap[ií]t", "ind")
    d["trab_completos_eventos"] = col(doc, r"completos em eventos", "ind")
    d["resumos_eventos"] = col(doc, r"resumo", "ind")
    # Inovação / técnica
    d["trab_tecnicos"] = col(doc, r"Trab\. t[eé]cnicos", "ind")
    d["software_patentes"] = col(doc, r"software|Patente|cultivar|desenho", "ind")
    d["prod_artistica"] = col(doc, r"Artes|art[ií]stica", "ind")
    # Formação de pessoas (CV)
    d["orient_IC_grad_outras"] = col(doc, r"^op\.orientacoesConcluidas\.outras")
    d["orient_mestrado"] = col(doc, r"^op\.orientacoesConcluidas\.orientacoesConcluidasParaMestrado")
    d["orient_doutorado"] = col(doc, r"^op\.orientacoesConcluidas\.orientacoesConcluidasParaDoutorado")
    d["orientacoes_total"] = d[["orient_IC_grad_outras", "orient_mestrado", "orient_doutorado"]].sum(axis=1)
    d["orient_em_andamento"] = col(doc, r"^dc\.orientacoesEmAndamento")
    d["bancas_trabalhos"] = col(doc, r"participacaoEmBancaTrabalhosConclusao")
    d["bancas_julgadoras"] = col(doc, r"participacaoEmBancaJulgadora")
    # Ensino, extensão, gestão, projetos (CV — atuação profissional)
    d["atividades_ensino"] = col(doc, r"atividadesDeEnsino")
    d["projetos"] = col(doc, r"atividadesDeParticipacaoEmProjeto")
    d["extensao"] = col(doc, r"atividadesDeExtensao|formacaoComplementarDeExtensao")
    d["gestao"] = col(doc, r"atividadesDeDirecao|atividadesDeConselho")
    d["material_didatico_cursos"] = col(doc, r"desenvolvimentoDeMaterialDidatico|cursoDeCurtaDuracaoMinistrado")
    d["difusao_eventos"] = col(doc, r"apresentacaoDeTrabalho|organizacaoDeEvento|midiaSocial|programaDeRadio")
    d["participacao_eventos"] = col(doc, r"participacaoEmEventos")
    d["formacao_complementar"] = col(doc, r"^dc\.formacaoComplementar")

    d["producao_bibliografica"] = d[["artigos_total", "livros", "capitulos", "trab_completos_eventos"]].sum(axis=1)
    d["artigos_por_ano"] = (d["artigos_total"] / d["anos_carreira"]).round(3)
    d["qualisA_por_ano"] = (d["art_qualis_A"] / d["anos_carreira"]).round(3)
    d["tem_producao"] = (d["producao_bibliografica"] > 0).astype(int)
    return d


VARS_MULTI = ["nivel_titulacao", "art_qualis_A", "art_qualis_B", "art_qualis_C_sem", "livros", "capitulos",
              "trab_completos_eventos", "resumos_eventos", "trab_tecnicos", "software_patentes",
              "orient_IC_grad_outras", "orient_mestrado", "orient_em_andamento", "bancas_trabalhos",
              "bancas_julgadoras", "atividades_ensino", "projetos", "extensao", "gestao",
              "material_didatico_cursos", "difusao_eventos", "participacao_eventos"]
CONTAGENS = [v for v in VARS_MULTI if v != "nivel_titulacao"]


# ------------------------------------------------------------------ utilidades estatísticas
def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def top10_share(x):
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    return float(x[: max(1, len(x) // 10)].sum() / x.sum()) if x.sum() else 0.0


def kmo(corr):
    inv = np.linalg.pinv(corr)
    parc = -inv / np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    np.fill_diagonal(parc, 0)
    r = corr.copy(); np.fill_diagonal(r, 0)
    return float((r ** 2).sum() / ((r ** 2).sum() + (parc ** 2).sum()))


def bartlett(corr, n):
    p = corr.shape[0]
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(corr))
    gl = p * (p - 1) / 2
    return float(chi2), float(gl), float(stats.chi2.sf(chi2, gl))


def parallel_analysis(Z, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    n, p = Z.shape
    real = np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False))[::-1]
    sims = np.array([np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)), rowvar=False))[::-1]
                     for _ in range(iters)])
    lim = np.percentile(sims, 95, axis=0)
    return int(max(2, (real > lim).sum())), real


def nomear_perfis(a, fatores):
    """Nomeia os clusters pelo padrão dos escores médios (F1 produção, F2 difusão/extensão,
    F3 gestão/ensino, F4 pós-graduação/inovação)."""
    m = a.groupby("cluster")[fatores].mean()
    nomes, usados = {}, set()
    ordem = [("Baixo registro", lambda r: -r.sum()),
             ("Pesquisador-formador", lambda r: r[fatores[3]]),
             ("Gestor-formador", lambda r: r[fatores[2]]),
             ("Extensionista-difusor", lambda r: r[fatores[1]]),
             ("Pesquisador acadêmico", lambda r: r[fatores[0]])]
    for nome, chave in ordem:
        cand = [c for c in m.index if c not in usados]
        if not cand:
            break
        c = max(cand, key=lambda c: chave(m.loc[c]))
        nomes[c] = nome; usados.add(c)
    for c in m.index:
        nomes.setdefault(c, f"Perfil {c}")
    return nomes


def varimax(L, gamma=1.0, iters=100, tol=1e-6):
    p, k = L.shape
    Rm, d = np.eye(k), 0
    for _ in range(iters):
        Lr = L @ Rm
        u, s_, vh = np.linalg.svd(L.T @ (Lr ** 3 - (gamma / p) * Lr @ np.diag((Lr ** 2).sum(axis=0))))
        Rm = u @ vh
        d_old, d = d, s_.sum()
        if d_old and d / d_old < 1 + tol:
            break
    return L @ Rm


# ---------------------------------------------------------------- índices de pesos fixos
# Pesos por unidade, normalizados para "artigo Qualis A / artigo indexado" = 4 (contagens em log1p).
# IPA-T: adaptado da Tabela de Pontuação de Atividades Acadêmicas EBTT da UTFPR (promoção a Titular),
#        que diferencia estratos Qualis e privilegia produção científica.
# IPA-R: adaptado dos regulamentos de RSC do IFMG (Res. 3/2014) e do IFPE (Res. 76/2014): média dos
#        pontos normalizados das duas tabelas; não diferenciam Qualis e valorizam ensino, gestão e material.
# Titulação (escala 1-5) recebe peso 3 nos dois, refletindo a retribuição por titulação (Lei 12.772/2012).
PESOS_IPA = {
    "IPA_T": {"nivel_titulacao": 3, "art_qualis_A": 4.0, "art_qualis_B": 1.4, "art_qualis_C_sem": 0.53,
              "livros": 2.67, "capitulos": 1.33, "trab_completos_eventos": 0.6, "resumos_eventos": 0.33,
              "orient_IC_grad_outras": 0.13, "orient_mestrado": 0.8, "bancas_trabalhos": 0.2,
              "software_patentes": 1.33, "projetos": 0.67, "extensao": 0.4, "gestao": 2.0,
              "material_didatico_cursos": 1.33, "difusao_eventos": 0.5, "atividades_ensino": 1.07},
    "IPA_R": {"nivel_titulacao": 3, "art_qualis_A": 4.0, "art_qualis_B": 4.0, "art_qualis_C_sem": 2.6,
              "livros": 10.0, "capitulos": 3.0, "trab_completos_eventos": 3.0, "resumos_eventos": 2.36,
              "orient_IC_grad_outras": 2.0, "orient_mestrado": 2.0, "bancas_trabalhos": 0.6,
              "software_patentes": 3.3, "projetos": 0.84, "extensao": 1.3, "gestao": 1.7,
              "material_didatico_cursos": 4.1, "difusao_eventos": 0.8, "atividades_ensino": 1.5},
}
INDICES = ["IPA_T", "IPA_R", "indice_PC1", "IM"]
ROTULO_IDX = {"IPA_T": "IPA-T", "IPA_R": "IPA-R", "indice_PC1": "PC1", "IM": "IM"}


def escala100(s):
    return (100 * (s - s.min()) / (s.max() - s.min())).round(2)


def indice_pesos_fixos(a, pesos):
    X = a[list(pesos)].fillna(0).astype(float)
    cont = [c for c in pesos if c != "nivel_titulacao"]
    X[cont] = np.log1p(X[cont])
    return escala100(sum(X[c] * w for c, w in pesos.items()))


def concordancia(x, y):
    """Correlação (Spearman) x concordância (topo, decis, deslocamento) entre dois índices."""
    n = len(x)
    rx, ry = x.rank(ascending=False), y.rank(ascending=False)
    sh = (rx - ry).abs()
    n10 = n // 10
    dx = np.ceil(10 * rx / n).clip(1, 10); dy = np.ceil(10 * ry / n).clip(1, 10)
    tx, ty = set(x.nlargest(n10).index), set(y.nlargest(n10).index)
    return {"spearman": float(stats.spearmanr(x, y)[0]),
            "top10_coincide_pct": 100 * len(tx & ty) / n10,
            "mesmo_decil_pct": 100 * float((dx == dy).mean()),
            "muda_2_ou_mais_decis_pct": 100 * float(((dx - dy).abs() >= 2).mean()),
            "desloc_mediano": float(sh.median()), "desloc_p90": float(sh.quantile(.9)),
            "pct_move_mais_10pct_do_ranking": 100 * float((sh > n / 10).mean())}


def epsilon2(h, n):
    return float(h / ((n ** 2 - 1) / (n + 1)))


MIN_COBERTURA = 0.5   # só entra na análise a instituição com >= 50% das pessoas extraídas
MIN_SERVIDORES = 100  # ... e com portal minimamente povoado (exclui portais quase vazios)


def cobertura(p: pd.DataFrame) -> pd.DataFrame:
    """Compara pessoas extraídas com o total do portal (lido do log da extração)."""
    totais = {}
    log = CFG.BASE_DIR / "data" / "extracao.log"
    if log.exists():
        for m in re.finditer(r"\] (\S+): (\d+) pessoas no portal", log.read_text(encoding="utf-8")):
            totais[m.group(1)] = int(m.group(2))
    ext = p.groupby("sigla").size()
    cob = pd.DataFrame({"sigla": ext.index, "extraidas": ext.values})
    cob["total_portal"] = cob["sigla"].map(totais)
    cob["cobertura"] = (cob["extraidas"] / cob["total_portal"]).round(3)
    cob["incluida_na_analise"] = (cob["cobertura"].fillna(1) >= MIN_COBERTURA) & (cob["extraidas"] >= MIN_SERVIDORES)
    return cob.sort_values("cobertura", ascending=False)


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(CFG.PROCESSED_DIR / "pessoas_completo.csv", low_memory=False)
    d = construir_base(p)
    cob = cobertura(p)
    cob.to_csv(OUT / "cobertura_extracao.csv", index=False)
    incluidas = set(cob.loc[cob["incluida_na_analise"], "sigla"])
    d["na_amostra_analitica"] = d["sigla"].isin(incluidas)
    resumo = {"gerado_em": datetime.now().isoformat(timespec="minutes"),
              "pessoas_total": int(len(p)), "docentes": int(len(d)),
              "docentes_efetivos": int(d["efetivo"].sum()),
              "instituicoes": int(d["sigla"].nunique()), "campi": int(d["campus_completo"].nunique()),
              "ufs": int(d["uf"].nunique())}
    resumo["instituicoes_na_analise"] = sorted(incluidas)
    resumo["criterio_amostra"] = f"docentes efetivos de instituições com cobertura >= {MIN_COBERTURA:.0%} e >= {MIN_SERVIDORES} servidores no portal"
    a = d[d["efetivo"] & d["na_amostra_analitica"]].copy()
    resumo["docentes_analise"] = int(len(a))

    # 1. Descritivas e concentração
    desc = []
    for v in CONTAGENS + ["producao_bibliografica", "artigos_por_ano", "anos_carreira", "anos_sem_atualizar_lattes"]:
        x = a[v].dropna()
        desc.append({"variavel": v, "media": x.mean(), "mediana": x.median(), "desvio": x.std(),
                     "p90": x.quantile(.9), "max": x.max(), "pct_zero": 100 * (x == 0).mean(),
                     "gini": gini(x), "top10_pct_do_total": 100 * top10_share(x)})
    desc = pd.DataFrame(desc).round(3)
    desc.to_csv(OUT / "descritivas.csv", index=False)
    resumo["pct_sem_producao_bibliografica"] = round(100 * (a["producao_bibliografica"] == 0).mean(), 1)
    resumo["pct_sem_artigo_qualisA"] = round(100 * (a["art_qualis_A"] == 0).mean(), 1)
    resumo["gini_qualisA"] = round(gini(a["art_qualis_A"]), 3)
    resumo["top10_share_qualisA"] = round(100 * top10_share(a["art_qualis_A"]), 1)
    resumo["pct_doutores"] = round(100 * a["doutor"].mean(), 1)
    resumo["mediana_anos_sem_atualizar_lattes"] = round(float(a["anos_sem_atualizar_lattes"].median()), 2)
    resumo["pct_lattes_mais_2anos"] = round(100 * (a["anos_sem_atualizar_lattes"] > 2).mean(), 1)

    # 2. Correlações (Spearman)
    corr = a[VARS_MULTI + ["anos_carreira"]].corr(method="spearman")
    corr.round(3).to_csv(OUT / "correlacao_spearman.csv")
    pares = (corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().rename("rho").reset_index()
             .rename(columns={"level_0": "var1", "level_1": "var2"}))
    pares["abs"] = pares["rho"].abs()
    pares.sort_values("abs", ascending=False).drop(columns="abs").round(3).to_csv(
        OUT / "correlacoes_destaque.csv", index=False)
    resumo["corr_titulacao_qualisA"] = round(float(corr.loc["nivel_titulacao", "art_qualis_A"]), 3)
    resumo["corr_gestao_qualisA"] = round(float(corr.loc["gestao", "art_qualis_A"]), 3)
    resumo["corr_extensao_qualisA"] = round(float(corr.loc["extensao", "art_qualis_A"]), 3)
    resumo["corr_ensino_qualisA"] = round(float(corr.loc["atividades_ensino", "art_qualis_A"]), 3)
    resumo["corr_carreira_qualisA"] = round(float(corr.loc["anos_carreira", "art_qualis_A"]), 3)

    # 3. Análise fatorial exploratória
    X = a[VARS_MULTI].fillna(0).astype(float).copy()
    X[CONTAGENS] = np.log1p(X[CONTAGENS])
    X = X.loc[:, X.std() > 0]
    Z = StandardScaler().fit_transform(X)
    R = np.corrcoef(Z, rowvar=False)
    resumo["KMO"] = round(kmo(R), 3)
    chi2, gl, pv = bartlett(R, len(Z))
    resumo["bartlett"] = {"chi2": round(chi2, 1), "gl": gl, "p": pv}
    n_fat, eig = parallel_analysis(Z)
    resumo["fatores_retidos_analise_paralela"] = n_fat
    resumo["autovalores"] = [round(float(e), 3) for e in eig[:8]]
    # Extração por componentes principais + rotação varimax (implementação própria)
    ev, evec = np.linalg.eigh(R)
    ordem = np.argsort(ev)[::-1][:n_fat]
    L = evec[:, ordem] * np.sqrt(ev[ordem])
    L = varimax(L)
    L = L * np.where(L.sum(axis=0) < 0, -1, 1)          # sinal: cargas majoritariamente positivas
    cargas = pd.DataFrame(L, index=X.columns, columns=[f"F{i+1}" for i in range(n_fat)])
    var_f = (L ** 2).sum(axis=0) / L.shape[0]
    ordem_f = np.argsort(var_f)[::-1]
    cargas = cargas.iloc[:, ordem_f]; cargas.columns = [f"F{i+1}" for i in range(n_fat)]
    var_f = var_f[ordem_f]; L = cargas.values
    # ordem canônica dos pilares: F1 produção científica, F2 difusão/extensão, F3 gestão/ensino, F4 pós-grad./inovação
    from itertools import permutations
    ancoras = ["art_qualis_A", "livros", "gestao", "orient_mestrado"]
    if n_fat == 4 and all(v in cargas.index for v in ancoras):
        M = cargas.loc[ancoras].values          # linhas: âncora, colunas: fator
        perm = max(permutations(range(4)), key=lambda p_: sum(M[i, p_[i]] for i in range(4)))
        cargas = cargas.iloc[:, list(perm)]; cargas.columns = [f"F{i+1}" for i in range(4)]
        var_f = var_f[list(perm)]; L = cargas.values
    pd.DataFrame({"fator": cargas.columns, "variancia": var_f, "acumulada": np.cumsum(var_f)}).round(3).to_csv(
        OUT / "fatorial_variancia.csv", index=False)
    escores = Z @ np.linalg.pinv(R) @ L                  # escores por regressão (Thurstone)
    cargas["comunalidade"] = (L ** 2).sum(axis=1)
    # nomeia cada fator pela variável de maior carga
    nomes = {}
    for f in [c for c in cargas.columns if c.startswith("F")]:
        top = cargas[f].abs().sort_values(ascending=False).index[:3].tolist()
        nomes[f] = " / ".join(top)
    cargas.round(3).to_csv(OUT / "fatorial_cargas.csv")
    resumo["fatores"] = nomes
    resumo["variancia_fatores"] = [round(float(v), 4) for v in var_f]
    for i, f in enumerate(nomes):
        a[f] = escores[:, i]

    pca = PCA().fit(Z)
    resumo["pca_var_PC1"] = round(float(pca.explained_variance_ratio_[0]), 3)
    a["PC1"] = pca.transform(Z)[:, 0] * (1 if pca.components_[0].sum() > 0 else -1)

    # 4. Clusters sobre os fatores
    F = a[list(nomes)].values
    amostra = np.random.default_rng(1).choice(len(F), min(len(F), 10000), replace=False)
    melhor = None
    sil = {}
    for k in range(4, 7):                       # 4 a 6 perfis (recomendação da orientação)
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(F)
        s = silhouette_score(F[amostra], km.labels_[amostra])
        sil[k] = round(float(s), 3)
        if melhor is None or s > melhor[0]:
            melhor = (s, k, km)
    resumo["silhouette_por_k"] = sil
    s, k, km = melhor
    a["cluster"] = km.labels_
    a["perfil"] = a["cluster"].map(nomear_perfis(a, list(nomes)))
    resumo["clusters"] = {"k": k, "silhouette": round(float(s), 3)}
    perfil = a.groupby("cluster")[VARS_MULTI + ["anos_carreira", "doutor"] + list(nomes)].mean()
    perfil.insert(0, "pct_docentes", 100 * a["cluster"].value_counts(normalize=True).sort_index())
    perfil.insert(0, "docentes", a["cluster"].value_counts().sort_index())
    perfil.round(2).to_csv(OUT / "clusters_perfil.csv")

    # 5. Testes por grupo (Kruskal-Wallis + tamanho de efeito ε²)
    alvo = ["art_qualis_A", "artigos_por_ano", "producao_bibliografica", "orientacoes_total",
            "extensao", "gestao", "projetos", "PC1"] + list(nomes)
    testes, medias = [], []
    for g in ["titulacao", "regiao", "grande_area", "sigla"]:
        sub = a.dropna(subset=[g])
        cont = sub[g].value_counts()
        sub = sub[sub[g].isin(cont[cont >= 30].index)]
        for v in alvo:
            grupos = [x[v].dropna().values for _, x in sub.groupby(g)]
            if len(grupos) < 2:
                continue
            h, pv = stats.kruskal(*grupos)
            testes.append({"agrupamento": g, "variavel": v, "H": h, "p": pv,
                           "epsilon2": epsilon2(h, len(sub)), "grupos": len(grupos)})
        m = sub.groupby(g)[alvo].agg(["median", "mean"]).round(3)
        m.columns = [f"{a_}_{b}" for a_, b in m.columns]
        m.insert(0, "docentes", sub[g].value_counts())
        m.insert(0, "agrupamento", g)
        medias.append(m.reset_index().rename(columns={g: "grupo"}))
    pd.DataFrame(testes).round(5).to_csv(OUT / "testes_grupos.csv", index=False)
    pd.concat(medias).to_csv(OUT / "medias_por_grupo.csv", index=False)

    # 6. Regressão binomial negativa: artigos Qualis A
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
        r = a.dropna(subset=["anos_carreira", "grande_area", "regiao"]).copy()
        r["titulacao_cat"] = pd.Categorical(r["titulacao"].where(r["titulacao"].isin(ORDEM_TIT), "Graduação"),
                                            categories=ORDEM_TIT)
        r["tit"] = r["titulacao_cat"].astype(str).replace({"Aperfeiçoamento": "Graduação"})
        for v in ["gestao", "extensao", "projetos", "atividades_ensino"]:
            r[f"{v}_log"] = np.log1p(r[v])
        formula = ("art_qualis_A ~ C(tit, Treatment('Mestrado')) + np.log(anos_carreira) + gestao_log + "
                   "extensao_log + projetos_log + atividades_ensino_log + orcid + "
                   "C(regiao, Treatment('Sudeste')) + C(grande_area)")
        m = smf.glm(formula, data=r, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
        tab = pd.DataFrame({"coef": m.params, "IRR": np.exp(m.params), "p": m.pvalues,
                            "IC95_inf": np.exp(m.conf_int()[0]), "IC95_sup": np.exp(m.conf_int()[1])})
        tab.round(4).to_csv(OUT / "regressao_qualisA.csv")
        resumo["regressao"] = {"n": int(m.nobs), "pseudo_R2_McFadden": round(1 - m.llf / m.llnull, 3),
                               "IRR": {k_: round(float(v), 3) for k_, v in np.exp(m.params).items()
                                       if m.pvalues[k_] < 0.05}}
    except Exception as e:  # noqa: BLE001
        resumo["regressao"] = f"não rodou: {e}"

    # 7. Nível institucional
    inst = a.groupby(["sigla", "uf", "regiao"]).agg(
        docentes=("pessoa_uid", "count"), campi=("campus", "nunique"),
        pct_doutores=("doutor", lambda s: 100 * s.mean()),
        qualisA_media=("art_qualis_A", "mean"), qualisA_mediana=("art_qualis_A", "median"),
        artigos_por_ano_media=("artigos_por_ano", "mean"),
        pct_sem_producao=("tem_producao", lambda s: 100 * (1 - s.mean())),
        gini_qualisA=("art_qualis_A", gini), extensao_media=("extensao", "mean"),
        gestao_media=("gestao", "mean"), projetos_media=("projetos", "mean"),
        PC1_medio=("PC1", "mean"), lattes_anos_sem_atualizar=("anos_sem_atualizar_lattes", "median"),
    ).round(3).reset_index()
    inst.to_csv(OUT / "instituicoes.csv", index=False)
    if len(inst) >= 5:
        rho, pv = stats.spearmanr(inst["pct_doutores"], inst["qualisA_media"])
        resumo["inst_corr_pctdoutores_qualisA"] = {"rho": round(float(rho), 3), "p": round(float(pv), 4)}

    # 8. Pesos fixos (IPA-T, IPA-R) x pesos dos dados (PC1, IM)
    a["indice_PC1"] = escala100(a["PC1"])
    F_ = a[list(nomes)].mean(axis=1)
    a["IM"] = escala100(F_)
    for nome_ipa, pesos in PESOS_IPA.items():
        a[nome_ipa] = indice_pesos_fixos(a, pesos)
    for c in INDICES:
        a["rank_" + c] = a[c].rank(ascending=False, method="min")
    a["desloc_IPAT_IM"] = a["rank_IPA_T"] - a["rank_IM"]
    a["desloc_IPAR_IM"] = a["rank_IPA_R"] - a["rank_IM"]
    a["desloc_IPAT_IPAR"] = a["rank_IPA_T"] - a["rank_IPA_R"]
    n10 = len(a) // 10
    res_idx = []
    from itertools import combinations
    for x, y in combinations(INDICES, 2):
        res_idx.append({"comparacao": f"{ROTULO_IDX[x]} x {ROTULO_IDX[y]}", **concordancia(a[x], a[y])})
    conc = pd.DataFrame(res_idx).round(3)
    conc.to_csv(OUT / "indices_concordancia.csv", index=False)
    resumo["indices_concordancia"] = conc.set_index("comparacao").to_dict(orient="index")
    perf = a.groupby("perfil").agg(docentes=("IM", "size"),
                                   **{f"rank_{c}": (f"rank_{c}", "median") for c in INDICES},
                                   desloc_mediano_IPAT_IM=("desloc_IPAT_IM", "median"),
                                   desloc_mediano_IPAR_IM=("desloc_IPAR_IM", "median"),
                                   desloc_mediano_IPAT_IPAR=("desloc_IPAT_IPAR", "median"))
    for c in INDICES:
        perf[f"top10_{c}_pct"] = 100 * a.nlargest(n10, c)["perfil"].value_counts().reindex(perf.index).fillna(0) / n10
    perf.round(1).to_csv(OUT / "indices_por_perfil.csv")
    resumo["indices_por_perfil"] = perf.round(1).to_dict(orient="index")
    tit = {c: (100 * a.nlargest(n10, c)["titulacao"].value_counts(normalize=True)).round(1).to_dict() for c in INDICES}
    resumo["top10_por_titulacao"] = tit
    # Sensibilidade: mesma lógica (IPA-T), pesos perturbados em +-50%
    rng = np.random.default_rng(7)
    base_t = a["IPA_T"]
    sens = []
    for _ in range(200):
        pw = {k: v * rng.uniform(.5, 1.5) for k, v in PESOS_IPA["IPA_T"].items()}
        sens.append(concordancia(base_t, indice_pesos_fixos(a, pw)))
    sens = pd.DataFrame(sens)
    resumo["sensibilidade_IPA_T_pm50"] = {"spearman_mediana": round(float(sens.spearman.median()), 3),
                                          "spearman_min": round(float(sens.spearman.min()), 3),
                                          "top10_mediana": round(float(sens.top10_coincide_pct.median()), 1),
                                          "top10_min": round(float(sens.top10_coincide_pct.min()), 1),
                                          "desloc_mediano_mediana": float(sens.desloc_mediano.median())}
    a[["pessoa_uid", "sigla", "titulacao", "perfil"] + INDICES + ["rank_" + c for c in INDICES]].to_csv(
        OUT / "ranking_indices_docentes.csv", index=False)

    # 8b. Ranking por campus (campi com >= MIN_DOC_CAMPUS docentes efetivos)
    MIN_DOC_CAMPUS = 20
    cp = a.groupby(["sigla", "uf", "regiao", "campus_completo"]).agg(
        docentes=("pessoa_uid", "count"), pct_doutores=("doutor", lambda s: 100 * s.mean()),
        qualisA_media=("art_qualis_A", "mean"),
        **{f"{c}_medio": (c, "mean") for c in INDICES},
        **{f"pct_{p_.lower().replace(' ', '_').replace('-', '_')}": ("perfil", lambda s, p_=p_: 100 * (s == p_).mean())
           for p_ in sorted(a["perfil"].unique())}).reset_index()
    cp = cp[(cp["docentes"] >= MIN_DOC_CAMPUS) & ~cp["campus_completo"].str.contains("Reitoria|Sem campus", case=False)].copy()
    for c in INDICES:
        cp["pos_" + c] = cp[f"{c}_medio"].rank(ascending=False, method="min").astype(int)
    cp["desloc_IPAT_IM"] = cp["pos_IPA_T"] - cp["pos_IM"]
    cp["desloc_IPAR_IM"] = cp["pos_IPA_R"] - cp["pos_IM"]
    cp["desloc_IPAT_IPAR"] = cp["pos_IPA_T"] - cp["pos_IPA_R"]
    cp = cp.sort_values("pos_IM").round(2)
    cp.to_csv(OUT / "ranking_campus.csv", index=False)
    top = lambda col: set(cp.nsmallest(max(1, len(cp) // 10), col)["campus_completo"])
    resumo["campus"] = {"campi_no_ranking": int(len(cp)), "min_docentes": MIN_DOC_CAMPUS}
    for x, y, lab_ in [("IPA_T", "IM", "IPAT_IM"), ("IPA_R", "IM", "IPAR_IM"), ("IPA_T", "IPA_R", "IPAT_IPAR")]:
        resumo["campus"][lab_] = {"spearman": round(float(stats.spearmanr(cp[f"{x}_medio"], cp[f"{y}_medio"])[0]), 3),
                                  "desloc_mediano": float(cp[f"desloc_{lab_}"].abs().median()),
                                  "desloc_max": int(cp[f"desloc_{lab_}"].abs().max()),
                                  "pct_muda_30_ou_mais": round(100 * float((cp[f"desloc_{lab_}"].abs() >= 30).mean()), 1),
                                  "top10_coincide_pct": round(100 * len(top(f"pos_{x}") & top(f"pos_{y}")) / max(1, len(top(f"pos_{y}"))), 1)}
    ins2 = a.groupby("sigla")[INDICES].mean()
    for c in INDICES:
        ins2["pos_" + c] = ins2[c].rank(ascending=False, method="min").astype(int)
    ins2.round(2).sort_values("pos_IM").to_csv(OUT / "ranking_instituicoes.csv")
    resumo["inst_spearman"] = {f"{x}_{y}": round(float(stats.spearmanr(ins2[x], ins2[y])[0]), 3)
                               for x, y in [("IPA_T", "IM"), ("IPA_R", "IM"), ("IPA_T", "IPA_R")]}

    # 9. Base final + dicionário
    base = d.join(a[list(nomes) + ["PC1", "cluster", "perfil"] + INDICES + ["rank_" + c for c in INDICES]
                    + ["desloc_IPAT_IM", "desloc_IPAR_IM", "desloc_IPAT_IPAR"]])
    base.to_csv(OUT / "base_docentes.csv", index=False)
    descr = {
        "efetivo": "Situação ATIVO PERMANENTE (análises multivariadas usam só efetivos)",
        "nivel_titulacao": "1=graduação/aperfeiçoamento 2=especialização 3=mestrado 4=doutorado 5=pós-doc",
        "anos_carreira": f"{ANO} − ano de conclusão da 1ª graduação",
        "anos_sem_atualizar_lattes": "anos desde a última atualização do Lattes",
        "art_qualis_A": "artigos Qualis A1–A4 (toda a carreira; endpoint de indicadores do Integra)",
        "art_qualis_C_sem": "artigos Qualis C, sem Qualis ou sem estrato informado",
        "orient_IC_grad_outras": "orientações concluídas de IC, TCC, especialização e outras",
        "atividades_ensino": "nº de registros de ensino na atuação profissional do Lattes",
        "projetos": "participações em projetos (pesquisa, extensão, desenvolvimento)",
        "extensao": "atividades de extensão + formação complementar de extensão",
        "gestao": "direção/administração + conselhos/comissões/consultoria",
        "difusao_eventos": "apresentações, organização de eventos, mídia e rádio/TV",
        "artigos_por_ano": "artigos_total / anos_carreira",
        "PC1": "1º componente principal (pesos derivados dos dados)",
        "IPA_T": "índice de pesos fixos (0–100) adaptado da tabela de promoção a Titular EBTT da UTFPR",
        "IPA_R": "índice de pesos fixos (0–100) adaptado dos regulamentos de RSC do IFMG (2014) e do IFPE (2014)",
        "indice_PC1": "PC1 reescalado 0–100",
        "cluster": "grupo K-Means sobre os escores fatoriais",
    }
    descr.update({f: f"escore do fator {f} ({n})" for f, n in nomes.items()})
    pd.DataFrame({"coluna": base.columns,
                  "descricao": [descr.get(c, "") for c in base.columns],
                  "tipo": [str(t) for t in base.dtypes]}).to_csv(OUT / "dicionario_dados.csv", index=False)

    # CSVs com BOM UTF-8 para o Excel reconhecer acentos
    for arq in OUT.glob("*.csv"):
        txt = arq.read_text(encoding="utf-8")
        if not txt.startswith("\ufeff"):
            arq.write_text("\ufeff" + txt, encoding="utf-8")
    (OUT / "resumo_achados.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=float),
                                             encoding="utf-8")
    print(json.dumps(resumo, ensure_ascii=False, indent=2, default=float))
    print(f"\nCSVs em {OUT}")


if __name__ == "__main__":
    main()
