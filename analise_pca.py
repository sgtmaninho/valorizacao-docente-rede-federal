"""PCA + clusterização (K-Means) sobre data/processed/docentes.csv.

Contrasta o índice de pesos fixos (IPA) com pesos derivados dos dados (PC1),
que é o argumento metodológico central do TCC.

Saídas em data/processed/:
  pca_variancia.csv, pca_cargas.csv, docentes_pca.csv, clusters_perfil.csv,
  instituicoes_pca.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import config as C
from transformer import VARIAVEIS_PCA

# Índice de pesos fixos (inspirado nos critérios de pontuação tipo RSC/CNPq)
PESOS_IPA = {"nivel_titulacao": 3.0, "art_qualis_A": 4.0, "art_qualis_B": 2.0, "art_outros": 1.0,
             "livros_capitulos": 2.0, "trab_eventos": 0.5, "orientacoes": 1.5, "bancas": 0.5,
             "prod_tecnica": 2.0, "projetos": 1.0, "extensao": 1.0, "gestao": 1.0,
             "anos_carreira": 0.0}


def main(k_min=3, k_max=7, seed=42):
    doc = pd.read_csv(C.PROCESSED_DIR / "docentes.csv")
    X = doc[VARIAVEIS_PCA].fillna(0).astype(float)
    Xt = X.copy()
    contagens = [c for c in VARIAVEIS_PCA if c not in ("nivel_titulacao", "anos_carreira")]
    Xt[contagens] = np.log1p(Xt[contagens])          # contagens são muito assimétricas
    Z = StandardScaler().fit_transform(Xt)

    pca = PCA(random_state=seed).fit(Z)
    var = pd.DataFrame({"componente": [f"PC{i+1}" for i in range(pca.n_components_)],
                        "variancia_explicada": pca.explained_variance_ratio_,
                        "acumulada": np.cumsum(pca.explained_variance_ratio_),
                        "autovalor": pca.explained_variance_})
    var.to_csv(C.PROCESSED_DIR / "pca_variancia.csv", index=False)
    n_kaiser = max(2, int((pca.explained_variance_ > 1).sum()))
    cargas = pd.DataFrame(pca.components_[:n_kaiser].T, index=VARIAVEIS_PCA,
                          columns=[f"PC{i+1}" for i in range(n_kaiser)])
    # orienta PC1 para que "mais produção" seja positivo
    if cargas["PC1"].sum() < 0:
        cargas["PC1"] *= -1
    cargas.to_csv(C.PROCESSED_DIR / "pca_cargas.csv")

    scores = Z @ cargas.values
    for i in range(n_kaiser):
        doc[f"PC{i+1}"] = scores[:, i]

    # K-Means: escolhe k pelo melhor silhouette (amostra para ficar rápido)
    Zk = scores
    amostra = np.random.default_rng(seed).choice(len(Zk), size=min(len(Zk), 8000), replace=False)
    melhor = None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Zk)
        s = silhouette_score(Zk[amostra], km.labels_[amostra])
        print(f"k={k}: silhouette={s:.3f}")
        if melhor is None or s > melhor[0]:
            melhor = (s, k, km)
    s, k, km = melhor
    doc["cluster"] = km.labels_
    print(f"Escolhido k={k} (silhouette {s:.3f})")

    # Índices: IPA (pesos fixos) x PC1 (pesos dos dados)
    ipa = sum(Xt[c] * w for c, w in PESOS_IPA.items())
    doc["IPA"] = 100 * (ipa - ipa.min()) / (ipa.max() - ipa.min() or 1)
    pc1 = doc["PC1"]
    doc["indice_PC1"] = 100 * (pc1 - pc1.min()) / (pc1.max() - pc1.min() or 1)
    doc["rank_IPA"] = doc["IPA"].rank(ascending=False)
    doc["rank_PC1"] = doc["indice_PC1"].rank(ascending=False)
    rho = doc[["IPA", "indice_PC1"]].corr(method="spearman").iloc[0, 1]
    print(f"Correlação de Spearman IPA x PC1: {rho:.3f}")
    doc.to_csv(C.PROCESSED_DIR / "docentes_pca.csv", index=False)

    perfil = doc.groupby("cluster")[VARIAVEIS_PCA + ["IPA", "indice_PC1"]].mean().round(2)
    perfil.insert(0, "docentes", doc.groupby("cluster").size())
    perfil.to_csv(C.PROCESSED_DIR / "clusters_perfil.csv")

    inst = doc.groupby(["sigla", "uf"]).agg(docentes=("pessoa_uid", "count"),
                                            IPA_medio=("IPA", "mean"),
                                            PC1_medio=("indice_PC1", "mean"),
                                            **{f"pct_cluster_{c}": ("cluster", lambda s, c=c: 100 * (s == c).mean())
                                               for c in sorted(doc["cluster"].unique())}).round(2)
    inst.to_csv(C.PROCESSED_DIR / "instituicoes_pca.csv")
    print(var.head(n_kaiser).round(3).to_string(index=False))
    print("Cargas (componentes com autovalor > 1):")
    print(cargas.round(2).to_string())


if __name__ == "__main__":
    main()
