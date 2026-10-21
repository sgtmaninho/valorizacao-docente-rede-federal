"""Painel do TCC — Valorização docente na Rede Federal (Portal Integra).

Dois segmentos (menu superior), conforme a orientação de 19/10/2026:
  🏛️ Instituição — ranking institucional (região, instituição, campus), comparação de instituições,
                   regiões e localização, funil e cobertura, séries temporais, dados agregados, sobre.
  👩‍🏫 Docente     — ranking de docentes (Score por Perfil e índices de comparação) com ficha, comparação
                   de professores, perfis, áreas, publicações, quem ganha e quem perde, dados, sobre.
Os filtros da barra lateral valem para os dois segmentos.

Rodar:  streamlit run dashboard.py
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

import importlib

import ranking_institucional as RI

RI = importlib.reload(RI)  # garante a versão atual do módulo mesmo com o servidor já aberto

BASE = Path(__file__).parent
DASH = BASE / "data" / "dashboard"
RK = BASE / "data" / "analise" / "ranking"
SER = BASE / "data" / "analise" / "series"
AZUL, LARANJA, VERDE, ROXO, CINZA = "#2a78d6", "#eb6834", "#4fa36b", "#8e5bc4", "#8a8f98"
PILARES = RI.PILARES
COR_PERFIL = {"Pesquisador-formador": ROXO, "Pesquisador acadêmico": AZUL, "Gestor-formador": VERDE,
              "Extensionista-difusor": LARANJA, "Baixo registro": CINZA}
INDICES = {"SP": "Score por Perfil", "IPA_T": "IPA-T (tabela UTFPR adaptada)", "IPA_R": "IPA-R (tabelas RSC IFMG/IFPE)",
           "IM": "IM (índice multidimensional)", "indice_PC1": "PC1 (pesos dos dados)"}
IND_INST = {"SI": "Score Institucional (SI)", "valor_agregado": "Valor agregado (SI − esperado)", "SP_medio": "Score por Perfil médio", "IPA_T_medio": "IPA-T médio",
            "IPA_R_medio": "IPA-R médio", "IM_medio": "IM médio"}
ROT_ITEM = {"nivel_titulacao": "Titulação", "art_qualis_A": "Artigos Qualis A", "art_qualis_B": "Artigos Qualis B",
            "art_qualis_C_sem": "Artigos C/sem Qualis", "livros": "Livros", "capitulos": "Capítulos",
            "trab_completos_eventos": "Trab. completos em eventos", "resumos_eventos": "Resumos em eventos",
            "orient_IC_grad_outras": "Orientações IC/TCC", "orient_mestrado": "Orientações de mestrado",
            "bancas_trabalhos": "Bancas", "software_patentes": "Software/patentes", "projetos": "Projetos",
            "extensao": "Extensão", "gestao": "Gestão", "material_didatico_cursos": "Material didático/cursos",
            "difusao_eventos": "Organização/difusão", "atividades_ensino": "Atividades de ensino"}
W = dict(use_container_width=True)

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
        "areas_grande_area.csv", "areas_detalhe.csv", "areas_clusters.csv", "regiao_padronizada.csv",
        "clusters_k4_k6.csv", "clusters_associacao.csv", "ganha_perde.csv", "top10_composicao_perfil.csv",
        "carreira_faixas.csv", "perfil_x_titulacao.csv"]}
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
fmt_int = lambda v: f"{int(v):,}".replace(",", ".")


def score_perfil(d: pd.DataFrame, peso_perfil: float, complemento: str) -> pd.Series:
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


def radar(linhas, titulo="", rng=(0, 100)):
    fig = go.Figure()
    cats = list(PILARES.values())
    for nome, vals, cor in linhas:
        vals = list(vals)
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], name=nome, line_color=cor,
                                      fill="toself", opacity=0.5))
    fig.update_layout(polar=dict(radialaxis=dict(range=list(rng))), height=400, title=titulo,
                      margin=dict(l=40, r=40, t=50, b=20), legend=dict(orientation="h"))
    return fig


def como_ler(texto):
    with st.expander("ℹ️ Como ler / como foi calculado"):
        st.markdown(texto)


# ------------------------------------------------------------------ barra lateral (filtros comuns)
st.sidebar.header("🔍 Filtros (valem para Instituição e Docente)")
sel = lambda rot, col: st.sidebar.multiselect(rot, sorted(d0[col].dropna().unique()))
f_reg = sel("Região", "regiao")
base_if = d0[d0["regiao"].isin(f_reg)] if f_reg else d0
f_if = st.sidebar.multiselect("Instituição", sorted(base_if["sigla"].dropna().unique()))
base_campus = base_if[base_if["sigla"].isin(f_if)] if f_if else base_if
f_campus = st.sidebar.multiselect("Campus", sorted(base_campus["campus"].dropna().unique()))
f_loc = sel("Localização do campus", "localizacao")
f_area = sel("Grande área CNPq", "grande_area")
f_tit = sel("Titulação", "titulacao")
f_perf = sel("Perfil do docente (K-Means)", "perfil")
f_min = st.sidebar.slider("Mínimo de artigos do docente", 0, 50, 0)
f_nome = st.sidebar.text_input("Buscar docente por nome")
with st.sidebar.expander("⚖️ Parâmetros do Score por Perfil"):
    peso = st.slider("Peso dos pilares dominantes (%)", 50, 100, 80, 5) / 100
    compl = st.radio("Complemento (restante)", ["Equilibrado (4 pilares)", "Pós-graduação e inovação (software/patentes)"])
    reposicionar = st.checkbox("Reposicionar docentes dentro do filtro", value=False,
                               help="Desmarcado: a posição mostrada é a da Rede Federal inteira.")
anon = st.sidebar.checkbox("Ocultar nomes (modo apresentação)", value=st.query_params.get("anon") == "1")

d0 = d0.copy()
if anon:
    d0["nome"] = "Docente " + d0["sigla"] + "-" + (d0["pessoa_uid"].astype(str).map(lambda u: zlib.crc32(u.encode())) % 100000).astype(str).str.zfill(5)
if peso != 0.80 or compl != "Equilibrado (4 pilares)":
    d0["SP"] = score_perfil(d0, peso, compl)
    d0["rank_SP"] = d0["SP"].rank(ascending=False, method="min").astype(int)
    d0["rank_SP_no_perfil"] = d0.groupby("perfil")["SP"].rank(ascending=False, method="min").astype(int)

d = d0
for col, v in [("regiao", f_reg), ("sigla", f_if), ("campus", f_campus), ("localizacao", f_loc),
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
filtrado = len(d) < N_REDE

# ------------------------------------------------------------------ cabeçalho e menu superior
st.title("🎓 Valorização Docente na Rede Federal")
st.caption("Dados públicos do Portal Integra (currículos Lattes) · docentes efetivos de "
           f"{d0['sigla'].nunique()} instituições · ranking multidimensional de instituições e docentes")
c = st.columns(5)
c[0].metric("Docentes no filtro", fmt_int(len(d)))
c[1].metric("Instituições", d["sigla"].nunique())
c[2].metric("Campi", d[["sigla", "campus"]].dropna().drop_duplicates().shape[0])
c[3].metric("Artigos registrados", fmt_int(d["artigos_total"].sum()))
c[4].metric("% doutores", f"{100 * (d['nivel_titulacao'] >= 4).mean():.1f}%" if len(d) else "—")
if d.empty:
    st.warning("Nenhum docente com esses filtros.")
    st.stop()
if filtrado:
    st.info("Filtros ativos: os rankings institucionais são recalculados só com os docentes filtrados "
            "(por exemplo, só doutores ou só uma área).")

opcoes = ["🏛️ Instituição", "👩‍🏫 Docente"]
if hasattr(st, "segmented_control"):
    seg = st.segmented_control("Segmento", opcoes, default=opcoes[0], label_visibility="collapsed") or opcoes[0]
else:
    seg = st.radio("Segmento", opcoes, horizontal=True, label_visibility="collapsed")


# ================================================================== INSTITUIÇÃO
def segmento_instituicao():
    abas = st.tabs(["🏆 Ranking institucional", "💡 Insights institucionais", "🔀 Comparar instituições", "🗺️ Regiões e localização",
                    "🔻 Funil e cobertura", "📈 Séries temporais", "📋 Dados agregados", "ℹ️ Sobre (Instituição)"])

    # ---------- ranking institucional
    with abas[0]:
        st.subheader("🏆 Ranking institucional")
        c1, c2, c3 = st.columns([1.2, 1.6, 1.2])
        nivel = c1.radio("Nível de agregação", ["Instituição", "Campus", "Região"], horizontal=True)
        idx = c2.selectbox("Ordenar por", list(IND_INST) + [f"PI_{k}" for k in PILARES],
                           format_func=lambda k: IND_INST.get(k, f"Pilar: {PILARES.get(k[3:], k)}"))
        min_n = c3.number_input("Mínimo de docentes na unidade", 1, 500, 20 if nivel == "Campus" else 1)
        with st.expander("⚖️ Pesos dos pilares no Score Institucional (padrão: 25% cada)"):
            cp = st.columns(4)
            pesos = {k: cp[i].slider(PILARES[k], 0, 100, 25, 5, key=f"w_{k}") for i, k in enumerate(PILARES)}
            if sum(pesos.values()) == 0:
                pesos = RI.PESOS_IGUAIS
        a = RI.agregar(d, nivel, pesos, min_n)
        if a.empty:
            st.warning("Nenhuma unidade com esse mínimo de docentes.")
            return
        a["unidade"] = RI.unidade_nome(a, nivel)
        a = a.sort_values(idx, ascending=False).reset_index(drop=True)
        a.insert(0, "posição", range(1, len(a) + 1))
        top = a.head(3)
        mc = st.columns(3)
        for i, (_, x) in enumerate(top.iterrows()):
            mc[i].markdown(f"### {['🥇', '🥈', '🥉'][i]} {x['unidade']}")
            mc[i].metric(IND_INST.get(idx, idx), f"{x[idx]:.1f}",
                         f"{fmt_int(x['docentes'])} docentes · faixa {int(x['faixa'])}", delta_color="off")
        n_show = min(len(a), 40)
        fig = go.Figure(go.Bar(
            x=a[idx].head(n_show)[::-1], y=a["unidade"].head(n_show)[::-1], orientation="h", marker_color=AZUL,
            error_x=dict(type="data", symmetric=False, array=(a["IC_sup"] - a["SI"]).head(n_show)[::-1],
                         arrayminus=(a["SI"] - a["IC_inf"]).head(n_show)[::-1]) if idx == "SI" else None))
        fig.update_layout(height=max(380, 22 * n_show), margin=dict(l=10, r=10, t=40, b=10),
                          title=f"{IND_INST.get(idx, idx)} — {nivel.lower()}" + (" (barras: IC 95%)" if idx == "SI" else ""))
        st.plotly_chart(fig, **W)
        cols = (["posição", "unidade", "docentes", "faixa", "SI", "IC_inf", "IC_sup", "SI_esperado", "valor_agregado", "pos_valor_agregado"] + [f"PI_{k}" for k in PILARES] +
                ["pilar_forte", "perfil_predominante", "SP_medio", "IPA_T_medio", "IM_medio", "pos_SI", "pos_SP_medio",
                 "pos_IPA_T_medio", "pos_IM_medio", "pct_doutores", "qualisA_por_docente", "orient_mestrado_por_docente",
                 "extensao_por_docente", "gestao_por_docente"])
        cols = [c for c in cols if c in a.columns]
        st.dataframe(a[cols].rename(columns={f"PI_{k}": f"pilar {PILARES[k].lower()}" for k in PILARES}), hide_index=True, **W,
                     column_config={"SI": st.column_config.ProgressColumn("SI", min_value=0, max_value=100, format="%.1f")})
        como_ler(f"""
**Score Institucional (SI).** Cada docente efetivo recebe um escore $s_i = \\sum_k w_k P_{{ik}}$, em que $P_{{ik}}$ é o seu
pilar *k* em percentil na Rede (0 a 100) e $w_k$ são os pesos acima (padrão: 25% para cada pilar). O SI da unidade é a média
de $s_i$ entre os seus docentes. Com pesos iguais, o SI é também a média dos quatro **pilares institucionais** (colunas "pilar …").

**Intervalo e faixa.** O IC 95% é $SI \\pm 1{{,}}96\\,EP$, com $EP = dp(s_i)/\\sqrt{{n}}$. Unidades na mesma **faixa** têm
intervalos que se sobrepõem ao do líder da faixa: a diferença entre elas não é estatisticamente clara, e a ordem dentro da faixa
deve ser lida com cautela.

**Valor agregado.** Para cada docente calcula-se o escore médio dos docentes com a mesma titulação e a mesma grande área
(o "esperado"); o valor agregado é o SI observado menos a média desses esperados. Positivo: a unidade rende acima do que a
formação e a área dos seus docentes fariam esperar.

**Coleta incompleta.** Docentes cujo currículo não pôde ser lido na coleta ficam fora do ranking institucional (hoje, 475 do IFSP).

**Por que média e não soma.** A média mede a intensidade da atuação por docente e não favorece instituições grandes.
Os totais estão na aba *Dados agregados*.

**Exemplo ({a.iloc[0]['unidade']}).** {fmt_int(a.iloc[0]['docentes'])} docentes; pilares institucionais
{', '.join(f"{PILARES[k].lower()} = {a.iloc[0][f'PI_{k}']:.1f}" for k in PILARES)}; SI = {a.iloc[0]['SI']:.1f}
(IC 95%: {a.iloc[0]['IC_inf']:.1f} a {a.iloc[0]['IC_sup']:.1f}).
""")
        if nivel == "Instituição":
            st.markdown("**Ranking pelo SI × ranking pela média do IPA-T (tabela de pesos fixos)**")
            fig = px.scatter(a, x="pos_IPA_T_medio", y="pos_SI", text="unidade", size="docentes", height=460,
                             labels={"pos_IPA_T_medio": "posição pelo IPA-T médio", "pos_SI": "posição pelo SI"})
            fig.add_shape(type="line", x0=1, y0=1, x1=len(a), y1=len(a), line=dict(dash="dash", color=CINZA))
            fig.update_traces(textposition="top center", marker_color=AZUL)
            fig.update_yaxes(autorange="reversed"); fig.update_xaxes(autorange="reversed")
            st.plotly_chart(fig, **W)
            st.caption("Acima da diagonal: a instituição sobe ao trocar a tabela de pesos fixos pelo SI multidimensional.")

    # ---------- insights institucionais
    with abas[1]:
        st.subheader("💡 Insights institucionais")
        ai = RI.agregar(d, "Instituição")
        if ai.empty:
            st.warning("Sem dados para os filtros atuais.")
        else:
            dc = d[d["coleta_cv"].astype(bool)] if "coleta_cv" in d else d
            # 1 valor agregado
            st.markdown("#### 1 · Quem faz mais com o que tem: valor agregado")
            va = ai.sort_values("valor_agregado")
            fig = go.Figure(go.Bar(x=va["valor_agregado"], y=va["sigla"], orientation="h",
                                   marker_color=[AZUL if v >= 0 else LARANJA for v in va["valor_agregado"]]))
            fig.add_vline(x=0, line_color="black", line_width=1)
            fig.update_layout(height=max(380, 20 * len(va)), margin=dict(l=10, r=10, t=30, b=10),
                              xaxis_title="SI observado − SI esperado pela titulação e área")
            c1, c2 = st.columns([1.1, 1])
            c1.plotly_chart(fig, **W)
            from scipy import stats as _st
            rho = _st.spearmanr(ai["SI"], ai["pct_doutores"])[0] if len(ai) > 2 else float("nan")
            fig2 = px.scatter(ai, x="pct_doutores", y="SI", text="sigla", size="docentes", trendline="ols" if len(ai) > 2 else None,
                              labels={"pct_doutores": "% de doutores", "SI": "Score Institucional"}, height=520,
                              title=f"SI × % de doutores (ρ = {rho:.2f})")
            fig2.update_traces(textposition="top center", marker_color=AZUL, selector=dict(mode="markers+text"))
            c2.plotly_chart(fig2, **W)
            top_va, bot_va = ai.nlargest(3, "valor_agregado"), ai.nsmallest(3, "valor_agregado")
            st.markdown(f"A titulação explica boa parte do ranking (ρ = {rho:.2f} entre SI e % de doutores). Descontando titulação e área, "
                        f"quem mais rende acima do esperado: **{', '.join(f'{r.sigla} (+{r.valor_agregado:.1f})' for r in top_va.itertuples())}**; "
                        f"abaixo do esperado: **{', '.join(f'{r.sigla} ({r.valor_agregado:.1f})' for r in bot_va.itertuples())}**.")
            # 2 concentração
            st.markdown("#### 2 · Dependência de poucos pesquisadores")
            cc = RI.concentracao(dc).sort_values("pct_art_qualis_A_top5")
            fig = px.bar(cc, x="pct_art_qualis_A_top5", y="sigla", orientation="h", height=max(380, 20 * len(cc)),
                         labels={"pct_art_qualis_A_top5": "% dos artigos Qualis A feitos pelos 5% mais produtivos"},
                         color="pct_art_qualis_A_top5", color_continuous_scale=[[0, AZUL], [1, LARANJA]])
            fig.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, **W)
            st.markdown(f"Quanto mais à direita, mais a produção qualificada depende de poucos docentes: se eles saírem, a instituição cai. "
                        f"Menos concentradas: {', '.join(cc.head(3)['sigla'])}; mais concentradas: {', '.join(cc.tail(3)['sigla'][::-1])}.")
            # 3 doutores invisíveis
            st.markdown("#### 3 · Doutores invisíveis")
            dt = dc[dc["nivel_titulacao"] >= 4]
            inv = dt.groupby("sigla").agg(doutores=("SP", "size"),
                                          no_baixo_registro=("perfil", lambda s: (s == "Baixo registro").sum()),
                                          sem_qualisA=("art_qualis_A", lambda s: (s == 0).sum())).reset_index()
            inv["% no baixo registro"] = (100 * inv["no_baixo_registro"] / inv["doutores"]).round(1)
            inv["% sem Qualis A"] = (100 * inv["sem_qualisA"] / inv["doutores"]).round(1)
            c1, c2, c3 = st.columns(3)
            c1.metric("Doutores no filtro", fmt_int(len(dt)))
            c2.metric("No perfil de baixo registro", f"{fmt_int((dt['perfil'] == 'Baixo registro').sum())} ({100 * (dt['perfil'] == 'Baixo registro').mean():.1f}%)")
            c3.metric("Sem nenhum artigo Qualis A", f"{100 * (dt['art_qualis_A'] == 0).mean():.1f}%")
            st.dataframe(inv.sort_values("% no baixo registro", ascending=False), hide_index=True, **W)
            # 4 início de carreira
            st.markdown("#### 4 · Início de carreira")
            fx = pd.cut(dc["anos_carreira"], [0, 10, 20, 30, 80], labels=["até 10 anos", "11–20", "21–30", "> 30"])
            pc = (100 * pd.crosstab(fx, dc["perfil"], normalize="index")).reindex(columns=RI.PERFIS).reset_index().melt(
                id_vars="anos_carreira", var_name="perfil", value_name="%")
            st.plotly_chart(px.bar(pc, x="anos_carreira", y="%", color="perfil", color_discrete_map=COR_PERFIL, height=360,
                                   labels={"anos_carreira": "anos desde a graduação"}, title="Perfil por tempo de carreira"), **W)
            st.markdown("Metade dos docentes com até 10 anos de carreira está no perfil de baixo registro: sinal para políticas de entrada "
                        "(mentoria, incentivo ao registro e à pesquisa) e não apenas de premiação de quem já produz.")
            # 5 qualidade do registro
            st.markdown("#### 5 · Qualidade do registro e da coleta")
            ql = d.groupby("sigla").agg(docentes=("SP", "size"),
                                        coleta_incompleta=("coleta_cv", lambda s: int((~s.astype(bool)).sum())) if "coleta_cv" in d else ("SP", lambda s: 0),
                                        pct_lattes_mais_2anos=("anos_sem_atualizar_lattes", lambda s: round(100 * (s > 2).mean(), 1)),
                                        pct_baixo_registro=("perfil", lambda s: round(100 * (s == "Baixo registro").mean(), 1))).reset_index()
            st.dataframe(ql.sort_values("pct_baixo_registro", ascending=False), hide_index=True, **W)
            st.warning("Cuidados: 475 docentes do IFSP não tiveram o currículo lido na coleta (ficam fora do ranking institucional "
                       "até a nova coleta). A data de atualização do Lattes parece incorreta no portal do IFRN (99,7% \"desatualizados\") "
                       "e ausente no do IFMS; por isso essa coluna não deve ser comparada entre instituições.")

    # ---------- comparar instituições
    with abas[2]:
        st.subheader("🔀 Comparar instituições")
        nivel_c = st.radio("Comparar", ["Instituição", "Campus", "Região"], horizontal=True, key="niv_cmp")
        ac = RI.agregar(d, nivel_c, None, 20 if nivel_c == "Campus" else 1)
        ac["unidade"] = RI.unidade_nome(ac, nivel_c)
        padrao = list(ac["unidade"].head(3))
        escolha = st.multiselect("Escolha de 2 a 4 unidades", list(ac["unidade"]), default=padrao, max_selections=4)
        if len(escolha) < 2:
            st.info("Selecione pelo menos duas unidades.")
        else:
            cores = [AZUL, LARANJA, VERDE, ROXO]
            s = ac.set_index("unidade").loc[escolha]
            c1, c2 = st.columns(2)
            c1.plotly_chart(radar([(u, [s.loc[u, f"PI_{k}"] for k in PILARES], cores[i]) for i, u in enumerate(escolha)],
                                  "Pilares institucionais (média dos percentis)", (0, max(80, s[[f"PI_{k}" for k in PILARES]].values.max() + 5))), **W)
            comp = s[[f"pct_{p}" for p in RI.PERFIS]].rename(columns=lambda c: c[4:]).reset_index().melt(
                id_vars="unidade", var_name="perfil", value_name="% dos docentes")
            c2.plotly_chart(px.bar(comp, x="% dos docentes", y="unidade", color="perfil", orientation="h", height=400,
                                   color_discrete_map=COR_PERFIL, title="Composição por perfil docente"), **W)
            chave = RI.NIVEIS[nivel_c]
            dd = d.copy()
            dd["unidade"] = (dd["sigla"] + " – " + dd["campus"].astype(str)) if nivel_c == "Campus" else dd[chave[0]].astype(str)
            dd = dd[dd["unidade"].isin(escolha)]
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.box(dd, x="unidade", y="SP", color="unidade", color_discrete_sequence=cores, height=380,
                                   title="Distribuição do Score por Perfil dos docentes", points=False), **W)
            tit = (100 * pd.crosstab(dd["unidade"], dd["titulacao"], normalize="index")).reset_index().melt(
                id_vars="unidade", var_name="titulação", value_name="%")
            c2.plotly_chart(px.bar(tit, x="%", y="unidade", color="titulação", orientation="h", height=380,
                                   title="Titulação dos docentes"), **W)
            linhas = ["docentes", "faixa", "SI", "IC_inf", "IC_sup", "pos_SI", "SP_medio", "pos_SP_medio", "IPA_T_medio",
                      "pos_IPA_T_medio", "IM_medio", "pos_IM_medio", "pct_doutores", "qualisA_por_docente",
                      "artigos_por_docente", "orient_mestrado_por_docente", "extensao_por_docente", "gestao_por_docente",
                      "software_por_docente", "pilar_forte", "perfil_predominante"]
            st.dataframe(s[[c for c in linhas if c in s.columns]].T, **W)
            como_ler("As posições (pos_…) referem-se ao ranking das unidades do mesmo nível, com os filtros atuais. "
                     "Se os intervalos (IC_inf a IC_sup) de duas unidades se sobrepõem, a diferença no SI não é estatisticamente clara.")

    # ---------- regiões e localização
    with abas[3]:
        st.subheader("🗺️ Regiões e localização")
        ar = RI.agregar(d, "Região")
        c1, c2 = st.columns(2)
        c1.plotly_chart(radar([(r["regiao"], [r[f"PI_{k}"] for k in PILARES], cor) for (_, r), cor in
                               zip(ar.iterrows(), [AZUL, LARANJA, VERDE, ROXO, CINZA])],
                              "Pilares por região", (0, 80)), **W)
        rg = T["regiao_padronizada"].rename(columns={"qualisA_bruto": "observado",
                                                     "qualisA_padronizado_titulacao": "padronizado pela titulação"}).melt(
            id_vars="regiao", value_vars=["observado", "padronizado pela titulação"], var_name="medida")
        c2.plotly_chart(px.bar(rg, x="regiao", y="value", color="medida", barmode="group", height=400,
                               color_discrete_sequence=[AZUL, LARANJA], labels={"value": "Qualis A por docente"},
                               title="Qualis A: observado × padronizado pela titulação (Rede)"), **W)
        if INS:
            i1 = INS["insight_1_desigualdade_regional"]
            st.markdown(f"**Achado:** {i1['texto']} A amplitude entre regiões cai **{i1['reducao_pct']:.0f}%** quando "
                        "todas têm a mesma composição de titulação.")
        dl = d[d["localizacao"].isin(["Capital", "Interior"])]
        loc = dl.groupby("localizacao").agg(docentes=("SP", "size"), pct_doutores=("nivel_titulacao", lambda s: 100 * (s >= 4).mean()),
                                            qualisA_por_docente=("art_qualis_A", "mean"), orient_mestrado=("orient_mestrado", "mean"),
                                            extensao=("extensao", "mean"), gestao=("gestao", "mean"), SP_medio=("SP", "mean"),
                                            **{f"pilar_{PILARES[k].lower()}": (f"P_{k}", "mean") for k in PILARES}).round(2)
        st.markdown("**Capital × interior (com os filtros atuais)**")
        st.dataframe(loc.T, **W)
        como_ler("Localização do campus a partir do município informado no próprio Portal Integra; o DF inteiro conta como capital. "
                 "Reitorias e docentes sem campus ficam de fora desta comparação.")

    # ---------- funil
    with abas[4]:
        st.subheader("🔻 Funil de seleção (tipo PRISMA) e cobertura")
        c1, c2 = st.columns(2)
        fs = T["funil_selecao"]
        c1.plotly_chart(go.Figure(go.Funnel(y=fs["etapa"], x=fs["n"], marker_color=AZUL)).update_layout(
            height=440, margin=dict(l=10, r=10, t=30, b=10), title="Instituições e pessoas"), **W)
        fp = T["funil_producao"]
        c2.plotly_chart(go.Figure(go.Funnel(y=fp["etapa"], x=fp["docentes"], marker_color=LARANJA,
                                            textinfo="value+percent initial")).update_layout(
            height=440, margin=dict(l=10, r=10, t=30, b=10), title="Registro da produção"), **W)
        st.markdown("**Por que 33 instituições e não 41?** A Rede Federal tem 38 IFs, 2 CEFETs e a UTFPR; "
                    "só entram as que publicam dados no Portal Integra com cobertura suficiente.")
        st.dataframe(T["instituicoes_excluidas"], hide_index=True, **W)
        ci = T["cobertura_instituicoes"]
        c1, c2 = st.columns(2)
        c1.markdown("**IFs com mais dados**"); c1.dataframe(ci.head(8)[["sigla", "docentes_efetivos", "registros_producao", "pct_doutores"]], hide_index=True)
        c2.markdown("**IFs com menos dados**"); c2.dataframe(ci.tail(8)[["sigla", "docentes_efetivos", "registros_producao", "pct_doutores"]], hide_index=True)

    # ---------- séries
    with abas[5]:
        st.subheader("📈 Séries temporais")
        if "serie_rede" in SERIES:
            sr = SERIES["serie_rede"]; sr = sr[sr["ano"].between(2005, 2025)]
            vars_ = sorted({c.rsplit("_", 1)[0] for c in sr.columns if c.endswith("_por100")})
            v = st.selectbox("Indicador", vars_, index=vars_.index("artigos_cv") if "artigos_cv" in vars_ else 0,
                             help="artigos_cv = artigos datados pelo Lattes (todas as instituições); art_qualis_* = só as 12 instituições cujo portal data o estrato Qualis.")
            medida = st.radio("Medida", ["por100", "pct_ativos", "total"], horizontal=True,
                              format_func={"por100": "por 100 docentes", "pct_ativos": "% de docentes ativos", "total": "total"}.get)
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.line(sr, x="ano", y=f"{v}_{medida}", markers=True, height=340, title="Rede",
                                    color_discrete_sequence=[AZUL]), **W)
            if "serie_regiao" in SERIES:
                rg = SERIES["serie_regiao"]; rg = rg[rg["ano"].between(2005, 2025)]
                c2.plotly_chart(px.line(rg, x="ano", y=f"{v}_{medida}", color="regiao", height=340, title="Por região"), **W)
            if "serie_instituicao" in SERIES:
                si = SERIES["serie_instituicao"]
                if f_if:
                    si = si[si["sigla"].isin(f_if)]
                st.plotly_chart(px.line(si, x="janela", y="pos_IM", color="sigla", markers=True, height=420,
                                        title="Posição das instituições por triênio (índice multidimensional; 1 = melhor)")
                                .update_yaxes(autorange="reversed"), **W)
            st.caption("Os últimos anos sofrem atraso de registro no Lattes. Séries Qualis e triênios usam as 12 instituições "
                       "cujo portal data os itens Qualis.")

    # ---------- dados agregados
    with abas[6]:
        st.subheader("📋 Dados agregados")
        nivel_d = st.radio("Nível", ["Instituição", "Campus", "Região"], horizontal=True, key="niv_dados")
        ad = RI.agregar(d, nivel_d, None, 1)
        tot = d.copy()
        chave = RI.NIVEIS[nivel_d]
        if nivel_d == "Campus":
            tot = tot[~tot["localizacao"].isin(["Reitoria", "Sem campus"]) & tot["campus"].notna()]
        somas = tot.groupby(chave)[["artigos_total", "art_qualis_A", "livros", "capitulos", "orient_mestrado",
                                    "orient_doutorado", "software_patentes", "projetos", "extensao"]].sum().add_prefix("total_")
        ad = ad.merge(somas.reset_index(), on=chave, how="left")
        st.dataframe(ad, hide_index=True, **W, height=520)
        st.download_button("⬇️ Baixar CSV", ad.to_csv(index=False).encode("utf-8-sig"), f"agregado_{nivel_d.lower()}.csv", "text/csv")

    # ---------- sobre
    with abas[7]:
        st.subheader("ℹ️ Sobre o segmento Instituição")
        st.markdown("""
Este segmento responde: **como as instituições, campi e regiões se comparam quando a atuação docente é vista em várias
dimensões, e não só pela produção científica?**

#### 1. De onde vêm os números
- Fonte: Portal Integra (currículos Lattes e indicadores Qualis), docentes efetivos de 33 instituições (34.282 pessoas).
- Cada docente tem quatro **pilares** (0 a 100), obtidos por análise fatorial de 22 indicadores (ver *Sobre (Docente)*).
- O pilar é a posição do docente na Rede: 70 significa que ele supera 70% dos docentes efetivos naquela dimensão;
  quem não tem nenhum registro na dimensão fica com 0.

#### 2. Pilar institucional
""")
        st.latex(r"PI_{jk} = \frac{1}{n_j}\sum_{i \in j} P_{ik}")
        st.markdown("Média do pilar *k* entre os $n_j$ docentes efetivos da unidade *j* (região, instituição ou campus).")
        st.markdown("#### 3. Score Institucional (SI)")
        st.latex(r"s_i = \sum_{k=1}^{4} w_k\,P_{ik}, \qquad SI_j = \frac{1}{n_j}\sum_{i \in j} s_i, \qquad \sum_k w_k = 1")
        st.markdown("""
- **Pesos padrão: 25% para cada pilar.** A Lei nº 11.892/2008 atribui aos Institutos Federais ensino, pesquisa, extensão
  e inovação ao mesmo tempo; por isso nenhuma dimensão recebe peso maior por padrão. Os pesos podem ser alterados na aba
  *Ranking institucional* para testar a sensibilidade.
- Com pesos iguais, $SI_j$ é a média dos quatro pilares institucionais.
- **Média, não soma:** mede a intensidade por docente e não favorece instituições grandes.
""")
        st.markdown("#### 4. Incerteza e faixas")
        st.latex(r"EP_j = \frac{dp(s_i)}{\sqrt{n_j}}, \qquad IC_{95\%} = SI_j \pm 1{,}96\,EP_j")
        st.markdown("""
Percorrendo o ranking do primeiro ao último, uma unidade fica na **faixa** do líder enquanto o limite superior do seu IC
alcança o limite inferior do IC do líder; senão, abre-se uma nova faixa. Dentro da mesma faixa, a ordem não é
estatisticamente clara.

#### 5. Índices de comparação
- **SP médio:** média do Score por Perfil dos docentes (cada docente ponderado pelo seu perfil).
- **IPA-T médio / IPA-R médio:** médias dos índices de pesos fixos adaptados das tabelas oficiais (UTFPR; IFMG e IFPE).
- **IM médio:** média do índice multidimensional (média simples dos quatro fatores).

#### 6. Outras colunas
- **Pilar forte:** o pilar institucional mais alto da unidade.
- **Perfil predominante:** o perfil docente mais frequente na unidade.
- **Localização:** capital ou interior, pelo município do campus no Portal Integra.

#### 7. Cuidados de leitura
Os dados são autodeclarados no Lattes; instituições com mais docentes de baixo registro aparecem com SI menor, o que pode
refletir sub-registro e não necessariamente menor atuação. O ranking deve ser lido como diagnóstico, não como premiação.
""")


# ================================================================== DOCENTE
def segmento_docente():
    abas = st.tabs(["🏆 Ranking docente", "🔀 Comparar professores", "🧩 Perfis (Clusters)", "📊 Classificação por área",
                    "📚 Publicações", "💡 Quem ganha e quem perde", "📋 Dados detalhados", "ℹ️ Sobre (Docente)"])

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
                         hide_index=True, **W)
        else:
            if escopo == "Dentro do perfil":
                r = d.sort_values(["perfil", "rank_SP_no_perfil"]).groupby("perfil").head(n_top); pos_col = "rank_SP_no_perfil"
            else:
                r = d.sort_values(f"rank_{idx}").head(n_top); pos_col = f"rank_{idx}"
            mc = st.columns(3)
            for i, (_, x) in enumerate(r.head(3).iterrows()):
                with mc[i]:
                    st.markdown(f"### {['🥇', '🥈', '🥉'][i]} {x['nome']}")
                    st.caption(f"{x['sigla']} · {x['campus']} · {x['perfil']} · {x['titulacao']}")
                    st.metric(INDICES[idx].split(" (")[0], f"{x[idx]:.1f}")
            cols = list(dict.fromkeys([pos_col, "nome", "sigla", "campus", "perfil", "titulacao", idx, "SP", "IPA_T", "IM",
                                       "rank_SP", "rank_IPA_T", "rank_IM", "art_qualis_A", "orient_mestrado", "gestao", "extensao"]))
            st.dataframe(r[cols], hide_index=True, **W,
                         column_config={idx: st.column_config.ProgressColumn(INDICES[idx], min_value=0, max_value=100, format="%.1f")})
        como_ler("""
**Score por Perfil (SP)** — índice proposto. Cada docente é avaliado sobretudo pelas dimensões em que o seu perfil se destaca:
80% do peso nos pilares dominantes do perfil e 20% complementar, igual entre os quatro pilares. Fórmula completa e exemplo
na aba *Sobre (Docente)*. **IPA-T** e **IPA-R** são tabelas de pesos fixos (oficiais, adaptadas); **IM** dá o mesmo peso às
quatro dimensões; **PC1** usa os pesos estatísticos do primeiro componente principal.
*Geral* compara todos; *Dentro do perfil* compara cada docente com os do seu grupo; *Por pilar* ordena por uma única dimensão.
""")
        st.markdown("---")
        st.subheader("🔎 Ficha do docente — pontos e score")
        nome = st.selectbox("Docente", d.sort_values("rank_SP")["nome"].fillna("?").head(5000).unique())
        x = d[d["nome"] == nome].iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{x['nome']}** — {x['sigla']} · {x['campus']} ({x['localizacao']})  \n"
                        f"{x['titulacao']} · {x['grande_area']} / {x['area']} · perfil **{x['perfil']}**")
            st.dataframe(pd.DataFrame({"Índice": [INDICES[k] for k in INDICES],
                                       "Valor (0–100)": [round(x[k], 1) for k in INDICES],
                                       "Posição na Rede": [int(x[f"rank_{k}"]) for k in INDICES]}), hide_index=True, **W)
            st.markdown(f"Posição dentro do perfil: **{int(x['rank_SP_no_perfil'])}º de {int(x['n_perfil'])}**")
            pes = T["score_perfil_pesos"].set_index("perfil").loc[x["perfil"]]
            comp = pd.DataFrame({"Pilar": list(PILARES.values()), "Pilar (0–100)": [x[f"P_{k}"] for k in PILARES],
                                 "Peso no perfil": [pes[f"peso_{k}"] for k in PILARES]})
            comp["Contribuição"] = (comp["Pilar (0–100)"] * comp["Peso no perfil"]).round(2)
            st.dataframe(comp, hide_index=True, **W)
            st.caption("SP = " + " + ".join(f"{p:.2f}×{v:.1f}" for p, v in zip(comp["Peso no perfil"], comp["Pilar (0–100)"]))
                       + f" = {comp['Contribuição'].sum():.1f}")
        with c2:
            med = d0[d0["perfil"] == x["perfil"]][[f"P_{k}" for k in PILARES]].median().tolist()
            st.plotly_chart(radar([(str(x["nome"]).split()[0], [x[f"P_{k}"] for k in PILARES], LARANJA),
                                   ("mediana do perfil", med, AZUL)], "Pilares (percentil na Rede)"), **W)
        pts = pd.DataFrame({"Item": [ROT_ITEM.get(p[4:], p[4:]) for p in PTS],
                            "Pontos IPA-T": [x[p] for p in PTS]}).sort_values("Pontos IPA-T", ascending=False)
        fig = px.bar(pts[pts["Pontos IPA-T"] > 0], x="Pontos IPA-T", y="Item", orientation="h", height=420,
                     title=f"De onde vêm os pontos na tabela de pesos fixos (IPA-T bruto = {x['pts_IPA_T_total']:.1f})",
                     color_discrete_sequence=[AZUL])
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, **W)

    with abas[1]:
        st.subheader("🔀 Comparação de professores")
        nomes = st.multiselect("Escolha de 2 a 4 docentes", d.sort_values("rank_SP")["nome"].dropna().unique()[:5000], max_selections=4)
        if len(nomes) >= 2:
            sel_d = d[d["nome"].isin(nomes)].drop_duplicates("nome")
            cores = [AZUL, LARANJA, VERDE, ROXO]
            st.plotly_chart(radar([(r["nome"], [r[f"P_{k}"] for k in PILARES], cores[i])
                                   for i, (_, r) in enumerate(sel_d.iterrows())], "Pilares"), **W)
            linhas = (["sigla", "campus", "perfil", "titulacao", "grande_area"] + list(INDICES) + [f"rank_{k}" for k in INDICES] +
                      ["art_qualis_A", "art_qualis_B", "livros", "capitulos", "orient_mestrado", "orient_doutorado",
                       "software_patentes", "projetos", "extensao", "gestao", "atividades_ensino", "pts_IPA_T_total"])
            st.dataframe(sel_d.set_index("nome")[linhas].T, **W)
        else:
            st.info("Selecione pelo menos dois docentes.")

    with abas[2]:
        st.subheader("🧩 Perfis de atuação (K-Means sobre os 4 fatores)")
        c1, c2 = st.columns([3, 2])
        with c1:
            smp = d.sample(min(len(d), 6000), random_state=1)
            eixo = st.radio("Eixos", ["F1 × F3", "F1 × F2", "F3 × F4"], horizontal=True)
            ex, ey = eixo.split(" × ")
            st.plotly_chart(px.scatter(smp, x=ex, y=ey, color="perfil", color_discrete_map=COR_PERFIL, opacity=0.45, height=460,
                                       labels={ex: PILARES[ex], ey: PILARES[ey]}, hover_name="nome"), **W)
        with c2:
            cnt = d["perfil"].value_counts().reset_index()
            st.plotly_chart(px.pie(cnt, names="perfil", values="count", color="perfil", color_discrete_map=COR_PERFIL,
                                   hole=0.45, height=300), **W)
            st.dataframe(T["clusters_k4_k6"], hide_index=True, **W)
        st.plotly_chart(radar([(p, [d0.loc[d0.perfil == p, f"P_{k}"].median() for k in PILARES], COR_PERFIL[p])
                               for p in RI.PERFIS], "Perfil típico (mediana dos pilares)"), **W)
        c1, c2 = st.columns(2)
        c1.markdown("**Associação do perfil com outras variáveis (V de Cramér)**")
        c1.dataframe(T["clusters_associacao"], hide_index=True, **W)
        c2.markdown("**Perfil por titulação (% da linha)**")
        c2.dataframe(T["perfil_x_titulacao"], hide_index=True, **W)
        como_ler("K-Means com k entre 4 e 6; k = 5 teve o maior silhouette (0,25). O silhouette moderado indica que os perfis são "
                 "regiões de maior densidade num contínuo, e não categorias estanques. O V de Cramér mostra que o perfil depende mais "
                 "da titulação e da área do que da instituição.")

    with abas[3]:
        st.subheader("📊 Classificação por área do conhecimento")
        ga = d.groupby("grande_area").agg(docentes=("SP", "size"), qualis_A=("art_qualis_A", "mean"),
                                          pct_doutores=("nivel_titulacao", lambda s: 100 * (s >= 4).mean()),
                                          SP=("SP", "mean"), IM=("IM", "mean")).reset_index()
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(ga.sort_values("docentes"), x="docentes", y="grande_area", orientation="h", height=380,
                               title="Docentes por grande área", color_discrete_sequence=[AZUL]), **W)
        g2 = T["areas_grande_area"].sort_values("qualisA_obs_esp")
        fig = px.bar(g2, x="qualisA_obs_esp", y="grande_area", orientation="h", height=380,
                     title="Qualis A observado ÷ esperado pela titulação (Rede)",
                     color=g2["qualisA_obs_esp"] < 1, color_discrete_map={True: LARANJA, False: AZUL})
        fig.add_vline(x=1, line_dash="dash"); fig.update_layout(showlegend=False)
        c2.plotly_chart(fig, **W)
        st.markdown("**Agrupamento das áreas (K-Means sobre o perfil médio de cada área com ≥ 100 docentes)**")
        st.dataframe(T["areas_clusters"], hide_index=True, **W)
        st.dataframe(ga.round(2), hide_index=True, **W)

    with abas[4]:
        st.subheader("📚 Produção registrada")
        tot = {"Artigos": "artigos_total", "Qualis A": "art_qualis_A", "Qualis B": "art_qualis_B", "Livros": "livros",
               "Capítulos": "capitulos", "Software/patentes": "software_patentes", "Orient. mestrado": "orient_mestrado"}
        mc = st.columns(len(tot))
        for i, (k, v) in enumerate(tot.items()):
            mc[i].metric(k, fmt_int(d[v].sum()))
        estr = d[["art_A1", "art_A2", "art_A3", "art_A4", "art_qualis_B", "art_qualis_C_sem"]].sum()
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(x=estr.index, y=estr.values, labels={"x": "estrato", "y": "artigos"}, height=340,
                               title="Artigos por estrato Qualis", color_discrete_sequence=[AZUL]), **W)
        c2.plotly_chart(px.histogram(d, x=np.log1p(d["art_qualis_A"]), nbins=40, height=340,
                                     title="Distribuição de log(1 + Qualis A)", color_discrete_sequence=[LARANJA]), **W)

    with abas[5]:
        st.subheader("💡 Quem ganha e quem perde ao trocar o índice")
        gp = T["ganha_perde"]
        dim = st.radio("Dimensão", gp["dimensao"].unique(), horizontal=True)
        g = gp[gp.dimensao == dim].rename(columns={"desloc_mediano_IPAT_para_IM": "IPA-T → IM",
                                                   "desloc_mediano_IPAT_para_SP": "IPA-T → Score por Perfil"}).melt(
            id_vars="grupo", value_vars=["IPA-T → IM", "IPA-T → Score por Perfil"], var_name="troca", value_name="posições")
        st.plotly_chart(px.bar(g, x="grupo", y="posições", color="troca", barmode="group", height=360,
                               color_discrete_sequence=[AZUL, LARANJA]), **W)
        st.markdown("**Composição do top 10% em cada índice (% por perfil)**")
        st.dataframe(T["top10_composicao_perfil"], hide_index=True, **W)
        if INS:
            i2, i3 = INS["insight_2_carreira_longa"], INS["insight_3_um_terco_baixo_registro"]
            c1, c2 = st.columns(2)
            c1.metric("Carreira longa × Qualis A", f"ρ = {i2['spearman_anos_carreira_qualisA']:.2f}")
            c1.dataframe(T["carreira_faixas"], hide_index=True, **W)
            c2.metric("Perfil de baixo registro", f"{i3['pct_baixo_registro']:.1f}%")
            c2.markdown(f"{i3['texto']} {i3['pct_sem_producao']:.1f}% não têm nenhuma produção lançada e "
                        f"{i3['pct_baixo_registro_lattes_desatualizado']:.1f}% dos docentes de baixo registro não atualizam o Lattes há mais de 2 anos.")
        como_ler("Deslocamento = posição no IPA-T − posição no outro índice (positivo: o grupo sobe). Valores medianos por grupo. "
                 "Estes números são da Rede inteira, com os parâmetros padrão (80/20).")

    with abas[6]:
        st.subheader("📋 Tabela completa (filtro atual)")
        cols = ["rank_SP", "nome", "sigla", "campus", "localizacao", "regiao", "titulacao", "grande_area", "area", "perfil",
                "SP", "IPA_T", "IPA_R", "IM", "P_F1", "P_F2", "P_F3", "P_F4", "art_qualis_A", "art_qualis_B", "artigos_total",
                "livros", "capitulos", "orient_mestrado", "software_patentes", "projetos", "extensao", "gestao", "numero_lattes"]
        st.dataframe(d.sort_values("rank_SP")[cols], hide_index=True, **W, height=560)
        st.download_button("⬇️ Baixar CSV", d[cols].to_csv(index=False).encode("utf-8-sig"), "docentes_filtrados.csv", "text/csv")

    with abas[7]:
        st.subheader("ℹ️ Sobre o segmento Docente")
        st.markdown("""
Este segmento responde: **quanto a escolha dos pesos altera quem é reconhecido entre os docentes, e como ranquear
respeitando os diferentes perfis de atuação?**

#### 1. Variáveis
22 indicadores por docente, de toda a trajetória registrada no Lattes/Integra: titulação; artigos por estrato Qualis;
livros, capítulos e trabalhos em eventos; trabalhos técnicos e software/patentes; orientações e bancas; atividades de
ensino, projetos, extensão, gestão, material didático e eventos. Contagens transformadas por log(1 + x) e padronizadas.

#### 2. Fatores e pilares
A análise fatorial (KMO = 0,94; quatro fatores pela análise paralela; rotação varimax) identificou quatro dimensões.
Cada **pilar** é a média das variáveis padronizadas com carga ≥ 0,40 no fator (escala somada), convertida em posição na Rede:
""")
        st.latex(r"P_{ik} = 100 \times \frac{\#\{\text{docentes com escala } k \text{ menor que a de } i\}}{N - 1}")
        st.dataframe(T["score_perfil_pilares_vars"], hide_index=True, **W)
        st.markdown("""
#### 3. Perfis
K-Means sobre os escores fatoriais, com k = 5 (melhor silhouette entre 4 e 6): pesquisador-formador, pesquisador acadêmico,
gestor-formador, extensionista-difusor e baixo registro.

#### 4. Score por Perfil (SP) — proposta do trabalho
Para cada perfil *p*, $c_{pk}$ é o centróide do perfil no pilar *k* (em desvios-padrão) e $D_p$ o conjunto de **pilares
dominantes** ($c_{pk} > 0{,}25$):
""")
        st.latex(r"w_{pk} = 0{,}80\,\frac{c_{pk}}{\sum_{l \in D_p} c_{pl}}\;\mathbb{1}[k \in D_p] \; + \; 0{,}20 \cdot \tfrac{1}{4}")
        st.latex(r"SP_i = \sum_{k=1}^{4} w_{p(i)k}\,P_{ik}")
        st.markdown("Perfil sem pilar dominante (baixo registro): $w_{pk} = 1/4$. Pesos resultantes:")
        pw = T["score_perfil_pesos"][["perfil", "peso_F1", "peso_F2", "peso_F3", "peso_F4", "pilares_dominantes"]].copy()
        pw.columns = ["perfil"] + list(PILARES.values()) + ["pilares dominantes"]
        st.dataframe(pw, hide_index=True, **W)
        ex = d0.sort_values("rank_SP").iloc[0]
        pe = T["score_perfil_pesos"].set_index("perfil").loc[ex["perfil"]]
        st.markdown(f"**Exemplo:** {ex['nome']} ({ex['perfil']}): SP = " +
                    " + ".join(f"{pe[f'peso_{k}']:.2f} × {ex[f'P_{k}']:.1f}" for k in PILARES) + f" = **{ex['SP']:.1f}**.")
        st.markdown("""
A proporção 80/20 é uma escolha normativa; variando-a de 60/40 a 100/0, o ranking quase não muda (Spearman ≥ 0,99).
É uma adaptação da lógica *benefit of the doubt*: cada docente é avaliado principalmente pelo que o seu grupo faz de melhor.

#### 5. Índices de comparação
""")
        st.latex(r"IPA_i = \sum_{v} \beta_v\,\log(1 + x_{iv}) \;+\; 3\cdot\text{titulação}_i \quad\text{(reescalado 0–100)}")
        st.markdown("""
- **IPA-T:** pesos $\\beta_v$ da tabela de promoção a Titular EBTT da UTFPR, normalizados para artigo Qualis A = 4.
- **IPA-R:** média normalizada das tabelas de RSC do IFMG (Res. 3/2014) e do IFPE (Res. 76/2014).
- **IM:** média simples dos quatro escores fatoriais (mesmo peso para cada dimensão), reescalada 0–100.
- **PC1:** escore do primeiro componente principal (pesos estatísticos), reescalado 0–100.

#### 6. Como os índices foram comparados
Correlação de Spearman (concordância global) e medidas de posição: coincidência do top 10%, permanência no mesmo decil,
mudança de dois ou mais decis e deslocamento mediano. A correlação alta (≈ 0,9) convive com grandes mudanças individuais —
por isso as medidas de posição são as que importam para decisões sobre pessoas.

#### 7. Cuidados de leitura
Dados autodeclarados; um terço dos docentes está no perfil de baixo registro, o que pode refletir sub-registro. O ranking é um
instrumento de análise da sensibilidade dos critérios, não uma medida definitiva de mérito.
""")
        st.caption("Autor: Edvaldo Pinto Maninho Júnior · TSI/IFB · Orientador: Prof. Gustavo Henrique Dornelas de Deus")


if seg == opcoes[0]:
    segmento_instituicao()
else:
    segmento_docente()
