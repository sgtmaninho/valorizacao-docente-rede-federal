"""Figuras e verificações complementares para o texto do TCC (data/analise/doc/)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import StandardScaler

from analise_exploratoria import VARS_MULTI, CONTAGENS, varimax

BASE = Path(__file__).parent
AN = BASE / "data" / "analise"
OUT = AN / "doc"; OUT.mkdir(parents=True, exist_ok=True)
AZUL, LARANJA, VERDE, ROXO, CINZA = "#2a78d6", "#eb6834", "#4fa36b", "#8e5bc4", "#8a8f98"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
PIL = ["Produção científica", "Difusão e extensão", "Gestão e ensino", "Pós-graduação e inovação"]


def solucao(Xdf, n=4):
    X = Xdf.copy(); X[[c for c in CONTAGENS if c in X]] = np.log1p(X[[c for c in CONTAGENS if c in X]])
    X = X.loc[:, X.std() > 0]
    Z = StandardScaler().fit_transform(X); R = np.corrcoef(Z, rowvar=False)
    ev, evec = np.linalg.eigh(R); o = np.argsort(ev)[::-1][:n]
    L = varimax(evec[:, o] * np.sqrt(ev[o]))
    return pd.DataFrame(L, index=X.columns)


def tucker(A, B):
    """congruência máxima após emparelhar fatores (permutação e sinal)."""
    from itertools import permutations
    A, B = A.align(B, join="inner", axis=0)
    C = np.array([[abs(np.dot(A[i], B[j]) / np.sqrt(np.dot(A[i], A[i]) * np.dot(B[j], B[j])))
                   for j in B.columns] for i in A.columns])
    p = max(permutations(range(C.shape[1])), key=lambda q: sum(C[i, q[i]] for i in range(C.shape[0])))
    return [round(float(C[i, p[i]]), 3) for i in range(C.shape[0])]


def main():
    b = pd.read_csv(AN / "base_docentes.csv", encoding="utf-8-sig", low_memory=False)
    a = b[b["F1"].notna()].copy()
    X = a[VARS_MULTI].fillna(0).astype(float)
    ref = solucao(X)
    # ordena a referência na ordem canônica usando as cargas publicadas
    cg = pd.read_csv(AN / "fatorial_cargas.csv", encoding="utf-8-sig", index_col=0)[["F1", "F2", "F3", "F4"]]
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(a)); h1, h2 = idx[: len(a) // 2], idx[len(a) // 2:]
    est = {"metade_1": tucker(cg, solucao(X.iloc[h1])), "metade_2": tucker(cg, solucao(X.iloc[h2])),
           "sem_lattes_desatualizado": tucker(cg, solucao(X[(a["anos_sem_atualizar_lattes"] <= 2).values]))}
    for r, d in a.groupby("regiao"):
        est[f"regiao_{r}"] = tucker(cg, solucao(X.loc[d.index]))
    inst = {s: tucker(cg, solucao(X.loc[d.index])) for s, d in a.groupby("sigla") if len(d) >= 300}
    est["instituicoes_min_por_fator"] = [round(min(v[i] for v in inst.values()), 3) for i in range(4)]
    est["instituicoes_pct_todos_085"] = round(100 * np.mean([min(v) >= 0.85 for v in inst.values()]), 1)
    est["instituicoes_mediana_por_fator"] = [round(float(np.median([v[i] for v in inst.values()])), 3) for i in range(4)]
    est["instituicoes_n"] = len(inst)
    est["instituicoes_menor"] = sorted(((min(v), s) for s, v in inst.items()))[:3]
    (OUT / "estabilidade_tucker.json").write_text(json.dumps(est, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(est, ensure_ascii=False))

    # Figura: autovalores x análise paralela
    Xl = X.copy(); Xl[[c for c in CONTAGENS if c in Xl]] = np.log1p(Xl[[c for c in CONTAGENS if c in Xl]])
    Z = StandardScaler().fit_transform(Xl.loc[:, Xl.std() > 0])
    ev = np.sort(np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False)))[::-1]
    sims = []
    for _ in range(20):
        Rn = np.corrcoef(rng.standard_normal(Z.shape), rowvar=False)
        sims.append(np.sort(np.linalg.eigvalsh(Rn))[::-1])
    p95 = np.percentile(sims, 95, axis=0)
    fig, ax = plt.subplots(figsize=(6.2, 3))
    k = np.arange(1, 13)
    ax.plot(k, ev[:12], "o-", color=AZUL, label="autovalores observados")
    ax.plot(k, p95[:12], "s--", color=LARANJA, label="percentil 95 – dados aleatórios")
    ax.axvline(4.5, color=CINZA, lw=0.7, ls=":"); ax.set_xticks(k)
    ax.set_xlabel("componente"); ax.set_ylabel("autovalor"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_analise_paralela.png"); plt.close(fig)

    # Figura: pilares medianos por perfil (heatmap)
    sp = pd.read_csv(AN / "ranking" / "score_perfil_docentes.csv", encoding="utf-8-sig")
    m = a[["pessoa_uid", "perfil"]].merge(sp, on="pessoa_uid")
    hm = m.groupby("perfil")[["P_F1", "P_F2", "P_F3", "P_F4"]].median()
    ordem = ["Pesquisador-formador", "Pesquisador acadêmico", "Gestor-formador", "Extensionista-difusor", "Baixo registro"]
    hm = hm.reindex(ordem)
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    im = ax.imshow(hm.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.values[i, j]; ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="white" if v > 60 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels([p.replace(" e ", "\ne ") for p in PIL], fontsize=8)
    ax.set_yticks(range(5)); ax.set_yticklabels(ordem, fontsize=8)
    fig.colorbar(im, ax=ax, label="pilar (percentil)", shrink=0.8)
    for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_perfis_pilares.png"); plt.close(fig)

    # Figura: arquitetura da solução
    fig, ax = plt.subplots(figsize=(7.6, 2.6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
    etapas = [("1 Extração", "extractor.py\nextrair_series.py\nAPI Portal Integra\n(40 portais)", AZUL),
              ("2 Transformação", "transformer.py\n22 variáveis,\nlimpeza, efetivos", AZUL),
              ("3 Análise", "analise_exploratoria.py\nAFE, K-Means, índices\nanalise_series.py", LARANJA),
              ("4 Ranking", "analise_ranking.py\nScore por Perfil,\nfunil, insights", LARANJA),
              ("5 Painel", "dashboard.py\nStreamlit + Plotly\n11 abas", VERDE)]
    for i, (t, s, c) in enumerate(etapas):
        x = 1 + i * 20
        ax.add_patch(FancyBboxPatch((x, 3), 17, 24, boxstyle="round,pad=0.4", fc=c, ec="none", alpha=0.12))
        ax.text(x + 8.5, 23, t, ha="center", fontsize=9, fontweight="bold", color=c)
        ax.text(x + 8.5, 12.5, s, ha="center", va="center", fontsize=7.2)
        if i < 4:
            ax.annotate("", xy=(x + 20.2, 15), xytext=(x + 17.8, 15), arrowprops=dict(arrowstyle="->", color=CINZA))
    fig.tight_layout(); fig.savefig(OUT / "fig_arquitetura.png"); plt.close(fig)

    # Séries temporais
    S = AN / "series"
    sr = pd.read_csv(S / "serie_rede.csv"); rg = pd.read_csv(S / "serie_regiao.csv")
    sr = sr[sr.ano.between(2008, 2025)]; rg = rg[rg.ano.between(2008, 2025)]
    st = pd.read_csv(S / "serie_titulacao.csv"); st = st[st.ano >= 1995]
    jc = pd.read_csv(S / "janelas_concordancia.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9))
    axs[0].fill_between(st.ano, st.pct_doutores_no_quadro_atual, color=LARANJA, alpha=0.35)
    axs[0].plot(st.ano, st.pct_doutores_no_quadro_atual, color=LARANJA)
    axs[0].set_title("(a) % de doutores no quadro atual", fontsize=9, loc="left")
    axs[1].plot(sr.ano, sr.artigos_cv_por100, "o-", color=AZUL, ms=3, label="artigos")
    axs[1].plot(sr.ano, sr.gestao_por100, "^-", color=VERDE, ms=3, label="gestão")
    axs[1].plot(sr.ano, sr.extensao_por100, "s-", color=LARANJA, ms=3, label="extensão")
    axs[1].axvspan(2023.5, 2025.5, color=CINZA, alpha=0.15); axs[1].set_xticks(range(2008, 2026, 3))
    axs[1].set_title("(b) Registros por 100 docentes em carreira", fontsize=9, loc="left"); axs[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_series_rede.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9))
    cores = {"Centro-Oeste": AZUL, "Nordeste": LARANJA, "Norte": VERDE, "Sudeste": ROXO, "Sul": CINZA}
    for r, d in rg.groupby("regiao"):
        axs[0].plot(d.ano, d.artigos_cv_por100, color=cores.get(r), label=r, lw=1.4)
    axs[0].axvspan(2023.5, 2025.5, color=CINZA, alpha=0.15); axs[0].set_xticks(range(2008, 2026, 3))
    axs[0].set_title("(a) Artigos por 100 docentes, por região", fontsize=9, loc="left"); axs[0].legend(frameon=False, fontsize=7)
    for comp, c in [("IPA_T x IM", AZUL), ("IPA_R x IM", LARANJA), ("IPA_T x IPA_R", CINZA)]:
        d = jc[jc.comparacao == comp]
        axs[1].plot(d.janela, d.spearman, "o-", color=c, label=comp.replace("_", "-").replace(" x ", " × "))
    axs[1].set_title("(b) Spearman entre índices por triênio", fontsize=9, loc="left")
    axs[1].legend(frameon=False, fontsize=7); axs[1].tick_params(axis="x", labelsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_series_regiao_janelas.png"); plt.close(fig)


if __name__ == "__main__":
    main()
