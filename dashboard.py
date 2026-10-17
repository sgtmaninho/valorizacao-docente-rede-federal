"""Painel do TCC — Valorização docente na Rede Federal (Portal Integra).

Mantém a estrutura do protótipo original do IFB (Classificação por Área, Clusters, Publicações,
Ranking, Dados Detalhados, Sobre) e acrescenta: Score por Perfil, Comparação de professores,
Insights, Funil e Cobertura, Séries temporais e Instituições/Campi.

Rodar:  streamlit run dashboard.py      (precisa de analise_exploratoria.py + analise_ranking.py)
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
DASH = BASE / "data" / "dashboard"
RK = BASE / "data" / "analise" / "ranking"
SER = BASE / "data" / "analise" / "series"
AZUL, LARANJA, VERDE, ROXO, CINZA = "#2a78d6", "#eb6834", "#4fa36b", "#8e5bc4", "#8a8f98"
PILARES = {"F1": "Produção científica", "F2": "Difusão e extensão", "F3": "Gestão e ensino",
           "F4": "Pós-graduação e inovação"}
COR_PERFIL = {"Pesquisador-formador": ROXO, "Pesquisador acadêmico": AZUL, "Gestor-formador": VERDE,
              "Extensionista-difusor": LARANJA, "Baixo registro": CINZA}
INDICES = {"SP": "Score por Perfil", "IPA_T": "IPA-T (tabela UTFPR adaptada)", "IPA_R": "IPA-R (tabelas RSC IFMG/IFPE)",
           "IM": "IM (índice multidimensional)", "indice_PC1": "PC1 (pesos dos dados)"}
ROT_ITEM = {"nivel_titulacao": "Titulação", "art_qualis_A": "Artigos Qualis A", "art_qualis_B": "Artigos Qualis B",
            "art_qualis_C_sem": "Artigos C/sem Qualis", "livros": "Livros", "capitulos": "Capítulos",
            "trab_completos_eventos": "Trab. completos em eventos", "resumos_eventos": "Resumos em eventos",
            "orient_IC_grad_outras": "Orientações IC/TCC", "orient_mestrado": "Orientações de mestrado",
            "bancas_trabalhos": "Bancas", "software_patentes": "Software/patentes", "projetos": "Projetos",
            "extensao": "Extensão", "gestao": "Gestão", "material_didatico_cursos": "Material didático/cursos",
            "difusao_eventos": "Organização/difusão", "atividades_ensino": "Atividades de ensino"}

st.set_page_config(page_title="Valorização Docente – Rede Federal", page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")


# ------------------------------------------------------------------ dados
@st.cache_data
def carregar():
    d = pd.read_csv(DASH / "docentes.csv.gz", low_memory=False)
    ler = lambda n: pd.read_csv(RK / n, encoding="utf-8-sig") if (RK / n).exists() else pd.DataFrame()
    t = {n.replace(".csv", ""): ler(n) for n in [
        "score_perfil_pesos.csv", "score_perfil_pilares_vars.csv", "funil_selecao.csv", "funil_producao.csv",
        "titulacao_amostra.csv", "cobertura_instituicoes.csv", "instituicoes_excluidas.csv",
        "localizacao_capital_interior.csv", "localizacao_por_titulacao.csv", "areas_grande_area.csv",
        "areas_detalhe.csv", "areas_clusters.csv", "titulacao_indicadores.csv", "regiao_padronizada.csv",
        "clusters_k4_k6.csv", "clusters_associacao.csv", "ganha_perde.csv", "top10_composicao_perfil.csv",
        "ranking_campus_sp.csv", "ranking_instituicoes_sp.csv", "carreira_faixas.csv", "perfil_x_titulacao.csv",
        "perfil_x_regiao.csv", "perfil_x_area.csv"]}
    ins = json.loads((RK / "insights.json").read_text(encoding="utf-8")) if (RK / "insights.json").exists() else {}
    ser = {}
    for n in ["serie_rede", "serie_regiao", "serie_perfil", "serie_titulacao", "janelas_concordancia", "serie_instituicao"]:
        f = SER / f"{n}.csv"
        if f.exists():
            ser[n] = pd.read_csv(f, encoding="utf-8-sig")
    return d, t, ins, ser


d0, T, INS, SERIES = carregar()
N_REDE = len(d0)
PTS = [c for c in d0.columns if c.startswith("pts_") and c != "pts_IPA_T_total"]


def score_perfil(d: pd.DataFrame, peso_perfil: float, complemento: str) -> pd.Series:
    """Recalcula o Score por Perfil com os parâmetros escolhidos na barra lateral."""
    pes = T["score_perfil_pesos"].set_index("perfil")
    out = pd.Series(0.0, index=d.index)
    for perf, r in pes.iterrows():
        cen = pd.Series({k: r[f"centroide_{k}"] for k in PILARES})
        pos = cen.clip(lower=0).where(cen > 0.25, 0)
        if pos.sum() == 0:
            w = pd.Series(0.25, index=cen.index)
        else:
            comp = (pd.Series(0.25, index=cen.index) if complemento == "Equilibrado (4 pilares)"
                    else pd.Series({"F1": 0, "F2": 0, "F3": 0, "F4": 1.0}))
            w = peso_perfil * pos / pos.sum() + (1 - peso_perfil) * comp
        m = d["perfil"] == perf
        out[m] = sum(d.loc[m, f"P_{k}"] * w[k] for k in PILARES)
    return out.round(2)


# ------------------------------------------------------------------ barra lateral
st.sidebar.header("🔍 Filtros")
sel = lambda rot, col: st.sidebar.multiselect(rot, sorted(d0[col].dropna().unique()))
f_if = sel("Instituição", "sigla")
f_reg = sel("Região", "regiao")
base_campus = d0[d0["sigla"].isin(f_if)] if f_if else d0
f_campus = st.sidebar.multiselect("Campus", sorted(base_campus["campus"].dropna().unique()))
f_loc = sel("Localização", "localizacao")
f_area = sel("Grande área CNPq", "grande_area")
f_tit = sel("Titulação", "titulacao")
f_perf = sel("Perfil (K-Means, k = 5)", "perfil")
f_min = st.sidebar.slider("Mínimo de artigos", 0, 50, 0)
f_nome = st.sidebar.text_input("Buscar por nome")
st.sidebar.markdown("---")
st.sidebar.header("⚖️ Score por Perfil")
peso = st.sidebar.slider("Peso dos pilares do perfil (%)", 50, 100, 80, 5) / 100
compl = st.sidebar.radio("Complemento (restante)", ["Equilibrado (4 pilares)", "Pós-graduação e inovação (software/patentes)"])
reposicionar = st.sidebar.checkbox("Reposicionar dentro do filtro", value=False,
                                   help="Desmarcado: mostra a posição na Rede Federal inteira.")

anon = st.sidebar.checkbox("Ocultar nomes (modo apresentação)", value=st.query_params.get("anon") == "1")
d0 = d0.copy()
if anon:
    d0["nome"] = "Docente " + d0["sigla"] + "-" + (d0["pessoa_uid"].astype(str).map(lambda u: zlib.crc32(u.encode())) % 100000).astype(str).str.zfill(5)
if peso != 0.80 or compl != "Equilibrado (4 pilares)":
    d0["SP"] = score_perfil(d0, peso, compl)
    d0["rank_SP"] = d0["SP"].rank(ascending=False, method="min").astype(int)
    d0["rank_SP_no_perfil"] = d0.groupby("perfil")["SP"].rank(ascending=False, method="min").astype(int)

d = d0
for col, v in [("sigla", f_if), ("regiao", f_reg), ("campus", f_campus), ("localizacao", f_loc),
               ("grande_area", f_area), ("titulacao", f_tit), ("perfil", f_perf)]:
    if v:
        d = d[d[col].isin(v)]
d = d[d["artigos_total"] >= f_min]
if f_nome:
    d = d[d["nome"].fillna("").str.contains(f_nome, case=False)]
if reposicionar:
    d = d.copy()
    for c in INDICES:
        d[f"rank_{c}"] = d[c].rank(ascending=False, method="min").astype(int)

# ------------------------------------------------------------------ cabeçalho
st.title("🎓 Valorização Docente na Rede Federal")
st.caption("Dados públicos do Portal Integra (currículos Lattes) · docentes efetivos de "
           f"{d0['sigla'].nunique()} instituições · ranking multidimensional com Score por Perfil")
c = st.columns(5)
c[0].metric("Docentes no filtro", f"{len(d):,}".replace(",", "."))
c[1].metric("Instituições", d["sigla"].nunique())
c[2].metric("Campi", d[["sigla", "campus"]].dropna().drop_duplicates().shape[0])
c[3].metric("Artigos registrados", f"{int(d['artigos_total'].sum()):,}".replace(",", "."))
c[4].metric("% doutores", f"{100 * (d['nivel_titulacao'] >= 4).mean():.1f}%" if len(d) else "—")
if d.empty:
    st.warning("Nenhum docente com esses filtros.")
    st.stop()

abas = st.tabs(["🏆 Ranking", "🔀 Comparar professores", "🧩 Perfis (Clusters)", "📊 Classificação por Área",
                "📚 Publicações", "💡 Insights", "🔻 Funil e Cobertura", "📈 Séries temporais",
                "🏫 Instituições e Campi", "📋 Dados Detalhados", "ℹ️ Sobre"])


def radar(linhas: list[tuple[str, list[float], str]], titulo=""):
    fig = go.Figure()
    cats = list(PILARES.values())
    for nome, vals, cor in linhas:
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], name=nome, line_color=cor,
                                      fill="toself", opacity=0.55))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=380, title=titulo,
                      margin=dict(l=40, r=40, t=40, b=20), legend=dict(orientation="h"))
    return fig


# ================================================================== 1. RANKING
with abas[0]:
    st.subheader("🏆 Ranking de docentes")
    c1, c2, c3 = st.columns([2, 2, 1])
    idx = c1.selectbox("Índice", list(INDICES), format_func=INDICES.get)
    escopo = c2.radio("Ranking", ["Geral", "Dentro do perfil", "Por pilar"], horizontal=True)
    n_top = c3.number_input("Mostrar", 10, 500, 50, 10)
    if escopo == "Por pilar":
        pil = st.selectbox("Pilar", list(PILARES), format_func=PILARES.get)
        r = d.sort_values(f"rank_P_{pil}").head(n_top)
        cols = [f"rank_P_{pil}", "nome", "sigla", "campus", "perfil", "titulacao", f"P_{pil}", "SP", "rank_SP"]
        st.dataframe(r[cols].rename(columns={f"rank_P_{pil}": "posição no pilar", f"P_{pil}": "pilar (0–100)"}),
                     hide_index=True, use_container_width=True)
    else:
        if escopo == "Dentro do perfil" and idx == "SP":
            r = d.sort_values(["perfil", "rank_SP_no_perfil"]).groupby("perfil").head(n_top)
            pos_col = "rank_SP_no_perfil"
        else:
            r = d.sort_values(f"rank_{idx}").head(n_top); pos_col = f"rank_{idx}"
        top3 = r.head(3)
        mc = st.columns(3)
        for i, (_, x) in enumerate(top3.iterrows()):
            with mc[i]:
                st.markdown(f"### {['🥇', '🥈', '🥉'][i]} {x['nome']}")
                st.caption(f"{x['sigla']} · {x['campus']} · {x['perfil']} · {x['titulacao']}")
                st.metric(INDICES[idx].split(" (")[0], f"{x[idx]:.1f}")
        cols = [pos_col, "nome", "sigla", "campus", "perfil", "titulacao", idx, "SP", "IPA_T", "IM",
                "rank_SP", "rank_IPA_T", "rank_IM", "art_qualis_A", "orient_mestrado", "gestao", "extensao"]
        cols = list(dict.fromkeys(cols))
        st.dataframe(r[cols], hide_index=True, use_container_width=True,
                     column_config={idx: st.column_config.ProgressColumn(INDICES[idx], min_value=0, max_value=100, format="%.1f")})
        fig = px.bar(r.head(20).iloc[::-1], x=idx, y="nome", color="perfil", color_discrete_map=COR_PERFIL,
                     orientation="h", height=520, title=f"Top 20 — {INDICES[idx]}")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔎 Ficha do docente — pontos e score")
    nome = st.selectbox("Docente", d.sort_values("rank_SP")["nome"].fillna("?").head(5000).unique())
    x = d[d["nome"] == nome].iloc[0]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"**{x['nome']}** — {x['sigla']} · {x['campus']} ({x['localizacao']})  \n"
                    f"{x['titulacao']} · {x['grande_area']} / {x['area']} · perfil **{x['perfil']}**")
        pos = pd.DataFrame({"Índice": [INDICES[k] for k in INDICES],
                            "Valor (0–100)": [round(x[k], 1) for k in INDICES],
                            "Posição na Rede": [int(x[f"rank_{k}"]) for k in INDICES]})
        st.dataframe(pos, hide_index=True, use_container_width=True)
        st.markdown(f"Posição dentro do perfil: **{int(x['rank_SP_no_perfil'])}º de {int(x['n_perfil'])}**")
        pes = T["score_perfil_pesos"].set_index("perfil").loc[x["perfil"]]
        comp = pd.DataFrame({"Pilar": list(PILARES.values()),
                             "Pilar (0–100)": [x[f"P_{k}"] for k in PILARES],
                             "Peso no perfil": [pes[f"peso_{k}"] for k in PILARES]})
        comp["Contribuição ao score"] = (comp["Pilar (0–100)"] * comp["Peso no perfil"]).round(2)
        st.dataframe(comp, hide_index=True, use_container_width=True)
    with c2:
        med = d0[d0["perfil"] == x["perfil"]][[f"P_{k}" for k in PILARES]].median().tolist()
        st.plotly_chart(radar([(x["nome"].split()[0], [x[f"P_{k}"] for k in PILARES], LARANJA),
                               (f"mediana do perfil", med, AZUL)], "Pilares (percentil na Rede)"), use_container_width=True)
    pts = pd.DataFrame({"Item": [ROT_ITEM.get(p[4:], p[4:]) for p in PTS],
                        "Quantidade": [x[p[4:]] if p[4:] in x else np.nan for p in PTS],
                        "Pontos IPA-T": [x[p] for p in PTS]}).sort_values("Pontos IPA-T", ascending=False)
    fig = px.bar(pts[pts["Pontos IPA-T"] > 0], x="Pontos IPA-T", y="Item", orientation="h", height=420,
                 title=f"De onde vêm os pontos (IPA-T bruto = {x['pts_IPA_T_total']:.1f})", color_discrete_sequence=[AZUL])
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig, use_container_width=True)

# ================================================================== 2. COMPARAR
with abas[1]:
    st.subheader("🔀 Comparação de professores")
    nomes = st.multiselect("Escolha de 2 a 4 docentes", d.sort_values("rank_SP")["nome"].dropna().unique()[:5000],
                           max_selections=4)
    if len(nomes) >= 2:
        sel_d = d[d["nome"].isin(nomes)].drop_duplicates("nome")
        cores = [AZUL, LARANJA, VERDE, ROXO]
        st.plotly_chart(radar([(r["nome"], [r[f"P_{k}"] for k in PILARES], cores[i])
                               for i, (_, r) in enumerate(sel_d.iterrows())], "Pilares"), use_container_width=True)
        linhas = (["sigla", "campus", "perfil", "titulacao", "grande_area"] + list(INDICES) +
                  [f"rank_{k}" for k in INDICES] + ["art_qualis_A", "art_qualis_B", "livros", "capitulos",
                  "orient_mestrado", "orient_doutorado", "software_patentes", "projetos", "extensao", "gestao",
                  "atividades_ensino", "pts_IPA_T_total"])
        st.dataframe(sel_d.set_index("nome")[linhas].T, use_container_width=True)
    else:
        st.info("Selecione pelo menos dois docentes.")

# ================================================================== 3. PERFIS
with abas[2]:
    st.subheader("🧩 Perfis de atuação (K-Means sobre os 4 fatores)")
    c1, c2 = st.columns([3, 2])
    with c1:
        smp = d.sample(min(len(d), 6000), random_state=1)
        eixo = st.radio("Eixos", ["F1 × F3", "F1 × F2", "F3 × F4"], horizontal=True)
        ex, ey = eixo.split(" × ")
        fig = px.scatter(smp, x=ex, y=ey, color="perfil", color_discrete_map=COR_PERFIL, opacity=0.45, height=460,
                         labels={ex: PILARES[ex], ey: PILARES[ey]}, hover_name="nome")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        cnt = d["perfil"].value_counts().reset_index()
        st.plotly_chart(px.pie(cnt, names="perfil", values="count", color="perfil", color_discrete_map=COR_PERFIL,
                               hole=0.45, height=300), use_container_width=True)
        st.dataframe(T["clusters_k4_k6"], hide_index=True, use_container_width=True)
        st.caption("k = 5 tem o maior silhouette entre 4 e 6 perfis; ARI mede a semelhança com a solução k = 5.")
    pes = T["score_perfil_pesos"]
    st.plotly_chart(radar([(r["perfil"], [d0.loc[d0.perfil == r["perfil"], f"P_{k}"].median() for k in PILARES],
                            COR_PERFIL.get(r["perfil"], CINZA)) for _, r in pes.iterrows()],
                          "Perfil típico (mediana dos pilares)"), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.markdown("**Associação do perfil com outras variáveis (V de Cramér)**")
    c1.dataframe(T["clusters_associacao"], hide_index=True, use_container_width=True)
    c2.markdown("**Perfil por titulação (% da linha)**")
    c2.dataframe(T["perfil_x_titulacao"], hide_index=True, use_container_width=True)

# ================================================================== 4. ÁREAS
with abas[3]:
    st.subheader("📊 Classificação por área do conhecimento")
    ga = d.groupby("grande_area").agg(docentes=("SP", "size"), qualis_A=("art_qualis_A", "mean"),
                                      pct_doutores=("nivel_titulacao", lambda s: 100 * (s >= 4).mean()),
                                      SP=("SP", "mean"), IM=("IM", "mean")).reset_index()
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(ga.sort_values("docentes"), x="docentes", y="grande_area", orientation="h", height=380,
                           title="Docentes por grande área", color_discrete_sequence=[AZUL]), use_container_width=True)
    g2 = T["areas_grande_area"].sort_values("qualisA_obs_esp")
    fig = px.bar(g2, x="qualisA_obs_esp", y="grande_area", orientation="h", height=380,
                 title="Qualis A observado ÷ esperado pela titulação (Rede)",
                 color=g2["qualisA_obs_esp"] < 1, color_discrete_map={True: LARANJA, False: AZUL})
    fig.add_vline(x=1, line_dash="dash"); fig.update_layout(showlegend=False)
    c2.plotly_chart(fig, use_container_width=True)
    st.markdown("**Agrupamento das áreas (K-Means sobre o perfil médio de cada área com ≥ 100 docentes)**")
    st.dataframe(T["areas_clusters"], hide_index=True, use_container_width=True)
    ad = T["areas_detalhe"]
    fig = px.scatter(ad, x="F1", y="F3", size="docentes", color="grupo_area_nome", hover_name="area", height=480,
                     labels={"F1": PILARES["F1"], "F3": PILARES["F3"]})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(ga.round(2), hide_index=True, use_container_width=True)

# ================================================================== 5. PUBLICAÇÕES
with abas[4]:
    st.subheader("📚 Produção registrada")
    tot = {"Artigos": "artigos_total", "Qualis A": "art_qualis_A", "Qualis B": "art_qualis_B", "Livros": "livros",
           "Capítulos": "capitulos", "Software/patentes": "software_patentes", "Orient. mestrado": "orient_mestrado"}
    mc = st.columns(len(tot))
    for i, (k, v) in enumerate(tot.items()):
        mc[i].metric(k, f"{int(d[v].sum()):,}".replace(",", "."))
    estr = d[["art_A1", "art_A2", "art_A3", "art_A4", "art_qualis_B", "art_qualis_C_sem"]].sum()
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(x=estr.index, y=estr.values, labels={"x": "estrato", "y": "artigos"}, height=340,
                           title="Artigos por estrato Qualis", color_discrete_sequence=[AZUL]), use_container_width=True)
    c2.plotly_chart(px.histogram(d, x=np.log1p(d["art_qualis_A"]), nbins=40, height=340,
                                 title="Distribuição de log(1 + Qualis A)", color_discrete_sequence=[LARANJA]),
                    use_container_width=True)
    porif = d.groupby("sigla").agg(qualis_A_por_docente=("art_qualis_A", "mean"), docentes=("SP", "size")).reset_index()
    st.plotly_chart(px.bar(porif.sort_values("qualis_A_por_docente"), x="qualis_A_por_docente", y="sigla",
                           orientation="h", height=650, title="Qualis A por docente — instituição",
                           color_discrete_sequence=[AZUL]), use_container_width=True)

# ================================================================== 6. INSIGHTS
with abas[5]:
    st.subheader("💡 Insights")
    if INS:
        i1, i2, i3 = (INS["insight_1_desigualdade_regional"], INS["insight_2_carreira_longa"],
                      INS["insight_3_um_terco_baixo_registro"])
        c = st.columns(3)
        c[0].metric("1 · Desigualdade regional", f"−{i1['reducao_pct']:.0f}%",
                    "da amplitude entre regiões some ao igualar a titulação", delta_color="off")
        c[1].metric("2 · Carreira longa", f"ρ = {i2['spearman_anos_carreira_qualisA']:.2f}",
                    "anos de carreira × Qualis A", delta_color="off")
        c[2].metric("3 · Um terço não registra", f"{i3['pct_baixo_registro']:.1f}%",
                    "dos efetivos no perfil de baixo registro", delta_color="off")
        st.markdown(f"**1.** {i1['texto']}")
        rg = T["regiao_padronizada"].rename(columns={"qualisA_bruto": "observado", "qualisA_padronizado_titulacao": "padronizado pela titulação"}).melt(
            id_vars="regiao", value_vars=["observado", "padronizado pela titulação"], var_name="medida")
        st.plotly_chart(px.bar(rg, x="regiao", y="value", color="medida", barmode="group", height=330,
                               color_discrete_sequence=[AZUL, LARANJA], labels={"value": "Qualis A por docente"}),
                        use_container_width=True)
        st.markdown(f"**2.** {i2['texto']}")
        st.dataframe(T["carreira_faixas"], hide_index=True, use_container_width=True)
        st.markdown(f"**3.** {i3['texto']} {i3['pct_sem_producao']:.1f}% não têm nenhuma produção lançada e "
                    f"{i3['pct_baixo_registro_lattes_desatualizado']:.1f}% dos docentes de baixo registro não atualizam o "
                    "Lattes há mais de 2 anos.")
        st.markdown("---")
        st.markdown("**Quem ganha e quem perde ao trocar o índice** (posições medianas; + = sobe)")
        gp = T["ganha_perde"]
        dim = st.radio("Dimensão", gp["dimensao"].unique(), horizontal=True)
        g = gp[gp.dimensao == dim].rename(columns={"desloc_mediano_IPAT_para_IM": "IPA-T → IM", "desloc_mediano_IPAT_para_SP": "IPA-T → Score por Perfil"}).melt(
            id_vars="grupo", value_vars=["IPA-T → IM", "IPA-T → Score por Perfil"], var_name="troca", value_name="posições")
        st.plotly_chart(px.bar(g, x="grupo", y="posições", color="troca", barmode="group", height=360,
                               color_discrete_sequence=[AZUL, LARANJA]), use_container_width=True)
        st.markdown("**Composição do top 10% em cada índice (% por perfil)**")
        st.dataframe(T["top10_composicao_perfil"], hide_index=True, use_container_width=True)
        st.markdown("**Capital × interior**")
        st.dataframe(T["localizacao_capital_interior"], hide_index=True, use_container_width=True)

# ================================================================== 7. FUNIL
with abas[6]:
    st.subheader("🔻 Funil de seleção (tipo PRISMA) e cobertura")
    c1, c2 = st.columns(2)
    fs = T["funil_selecao"]
    c1.plotly_chart(go.Figure(go.Funnel(y=fs["etapa"], x=fs["n"], marker_color=AZUL)).update_layout(height=440,
                    margin=dict(l=10, r=10, t=30, b=10), title="Instituições e pessoas"), use_container_width=True)
    fp = T["funil_producao"]
    c2.plotly_chart(go.Figure(go.Funnel(y=fp["etapa"], x=fp["docentes"], marker_color=LARANJA,
                                        textinfo="value+percent initial")).update_layout(height=440,
                    margin=dict(l=10, r=10, t=30, b=10), title="Registro da produção"), use_container_width=True)
    st.markdown("**Por que 33 instituições e não 41?** A Rede Federal tem 38 IFs, 2 CEFETs e a UTFPR; "
                "só entram as que publicam dados no Portal Integra com cobertura suficiente.")
    st.dataframe(T["instituicoes_excluidas"], hide_index=True, use_container_width=True)
    ci = T["cobertura_instituicoes"]
    c1, c2 = st.columns(2)
    c1.markdown("**IFs com mais dados**"); c1.dataframe(ci.head(8)[["sigla", "docentes_efetivos", "registros_producao", "pct_doutores"]], hide_index=True)
    c2.markdown("**IFs com menos dados**"); c2.dataframe(ci.tail(8)[["sigla", "docentes_efetivos", "registros_producao", "pct_doutores"]], hide_index=True)
    st.dataframe(ci, hide_index=True, use_container_width=True)
    st.markdown("**Titulação da amostra**")
    st.dataframe(T["titulacao_amostra"], hide_index=True)

# ================================================================== 8. SÉRIES
with abas[7]:
    st.subheader("📈 Séries temporais (produção datada, docentes do quadro atual)")
    if "serie_rede" in SERIES:
        sr = SERIES["serie_rede"]; sr = sr[sr["ano"].between(2005, 2025)]
        vars_ = sorted({c.rsplit("_", 1)[0].replace("_pct", "") for c in sr.columns if c.endswith("_por100")})
        v = st.selectbox("Indicador", vars_, index=vars_.index("artigos_cv") if "artigos_cv" in vars_ else 0,
                         help="artigos_cv = artigos datados pelo Lattes (todas as instituições); art_qualis_* = só as 12 instituições cujo portal data o estrato Qualis.")
        medida = st.radio("Medida", ["por100", "pct_ativos", "total"], horizontal=True,
                          format_func={"por100": "por 100 docentes", "pct_ativos": "% de docentes ativos", "total": "total"}.get)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.line(sr, x="ano", y=f"{v}_{medida}", markers=True, height=340, title="Rede",
                                color_discrete_sequence=[AZUL]), use_container_width=True)
        if "serie_regiao" in SERIES:
            rg = SERIES["serie_regiao"]; rg = rg[rg["ano"].between(2005, 2025)]
            c2.plotly_chart(px.line(rg, x="ano", y=f"{v}_{medida}", color="regiao", height=340, title="Por região"),
                            use_container_width=True)
        if "serie_titulacao" in SERIES:
            stt = SERIES["serie_titulacao"]; stt = stt[stt["ano"] >= 1995]
            st.plotly_chart(px.area(stt, x="ano", y="pct_doutores_no_quadro_atual", height=300,
                                    title="% de doutores no quadro atual, por ano", color_discrete_sequence=[LARANJA]),
                            use_container_width=True)
        if "janelas_concordancia" in SERIES:
            jc = SERIES["janelas_concordancia"]
            st.plotly_chart(px.line(jc[jc.comparacao.str.contains(" x ")], x="janela", y="spearman", color="comparacao",
                                    markers=True, height=320, title="Concordância entre índices por triênio"),
                            use_container_width=True)
        st.caption("Últimos anos sofrem atraso de registro no Lattes; leia 2024–2025 com cautela. Séries por estrato Qualis e "
                   "concordância por triênio usam apenas as 12 instituições cujo portal data os itens Qualis.")
    else:
        st.info("Rode analise_series.py para gerar as séries.")

# ================================================================== 9. INSTITUIÇÕES
with abas[8]:
    st.subheader("🏫 Instituições e campi")
    ri = T["ranking_instituicoes_sp"]
    fig = px.scatter(ri, x="pos_IPA_T_medio", y="pos_SP_medio", text="sigla", size="docentes", height=480,
                     labels={"pos_IPA_T_medio": "posição pelo IPA-T", "pos_SP_medio": "posição pelo Score por Perfil"})
    fig.add_shape(type="line", x0=1, y0=1, x1=len(ri), y1=len(ri), line=dict(dash="dash", color=CINZA))
    fig.update_traces(textposition="top center", marker_color=AZUL)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(ri, hide_index=True, use_container_width=True)
    rc = T["ranking_campus_sp"]
    if f_if:
        rc = rc[rc["sigla"].isin(f_if)]
    st.markdown("**Campi (≥ 20 docentes efetivos)**")
    st.dataframe(rc, hide_index=True, use_container_width=True)

# ================================================================== 10. DADOS
with abas[9]:
    st.subheader("📋 Tabela completa (filtro atual)")
    cols = ["rank_SP", "nome", "sigla", "campus", "localizacao", "regiao", "titulacao", "grande_area", "area", "perfil",
            "SP", "IPA_T", "IPA_R", "IM", "P_F1", "P_F2", "P_F3", "P_F4", "art_qualis_A", "art_qualis_B", "artigos_total",
            "livros", "capitulos", "orient_mestrado", "software_patentes", "projetos", "extensao", "gestao", "numero_lattes"]
    st.dataframe(d.sort_values("rank_SP")[cols], hide_index=True, use_container_width=True, height=560)
    st.download_button("⬇️ Baixar CSV", d[cols].to_csv(index=False).encode("utf-8-sig"), "docentes_filtrados.csv", "text/csv")

# ================================================================== 11. SOBRE
with abas[10]:
    st.subheader("ℹ️ Sobre o painel")
    pv = T["score_perfil_pilares_vars"]
    st.markdown("""
**Fonte.** Portal Integra (Rede Integra/IFRS): lista de servidores, indicadores com Qualis e currículo Lattes completo,
coletados por API pública. Amostra analítica: docentes efetivos das instituições com cobertura ≥ 50% e ≥ 100 servidores.

**Pilares.** Análise fatorial (PCA + varimax) em 22 indicadores (log1p + z) gerou 4 dimensões. Cada pilar é a escala
somada das variáveis com carga ≥ 0,40 no fator, convertida em percentil (0–100) na Rede.

**Perfis.** K-Means sobre os escores fatoriais; k entre 4 e 6 escolhido pelo silhouette (k = 5).

**Score por Perfil (SP).** Para cada perfil, os pilares em que o centróide passa de 0,25 desvio-padrão recebem
80% do peso (proporcional ao centróide); os 20% restantes são complementares (iguais entre os 4 pilares, ou
dirigidos à pós-graduação e inovação, conforme a barra lateral). Perfil sem pilar dominante: pesos iguais.
É uma adaptação da lógica *benefit of the doubt* (cada docente é avaliado pelo que mais faz).

**Índices de comparação.** IPA-T (tabela de promoção a Titular EBTT da UTFPR, normalizada; Qualis A = 4),
IPA-R (média das tabelas de RSC do IFMG e do IFPE), IM (média dos 4 fatores) e PC1 (pesos dos dados).
""")
    st.dataframe(pv, hide_index=True, use_container_width=True)
    st.caption("Autor: Edvaldo Pinto Maninho Júnior · TSI/IFB · Orientador: Prof. Gustavo Henrique Dornelas de Deus")
