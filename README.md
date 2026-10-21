# Valorização Docente na Rede Federal

Painel e pipeline de análise do TCC **"A dimensão multidimensional da valorização docente: ranking por perfis de atuação e o efeito da escolha de pesos na Rede Federal de Educação Profissional, Científica e Tecnológica"**.

- **Autor:** Edvaldo Pinto Maninho Júnior – Tecnologia em Sistemas para Internet, Instituto Federal de Brasília (IFB)
- **Orientador:** Prof. Gustavo Henrique Dornelas de Deus

**Painel online:** https://valorizacao-docente.streamlit.app

## O que é

Os dados públicos do [Portal Integra](https://integra.ifrs.edu.br) (currículos Lattes e indicadores Qualis) de **34.282 docentes efetivos de 33 instituições** da Rede Federal foram usados para:

1. identificar quatro dimensões da atuação docente (análise fatorial): produção científica, difusão e extensão, gestão e ensino, pós-graduação e inovação;
2. agrupar os docentes em cinco perfis (K-Means);
3. comparar índices de pesos fixos, adaptados de tabelas oficiais de pontuação (UTFPR, IFMG, IFPE), com índices derivados dos dados;
4. propor o **Score por Perfil**: 80% do peso nas dimensões dominantes do perfil de cada docente e 20% complementar.

## Painel

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

O painel tem dois segmentos no menu superior; os filtros da barra lateral valem para os dois.

**🏛️ Instituição** – ranking institucional (região, instituição ou campus) pelo **Score Institucional (SI)**, com
intervalo de confiança de 95% e faixas de empate estatístico; comparação de 2 a 4 instituições; regiões e
localização (capital × interior); funil e cobertura; séries temporais; dados agregados; sobre.

**👩‍🏫 Docente** – ranking de docentes pelo **Score por Perfil (SP)** e pelos índices de comparação, com ficha do
docente; comparação de professores; perfis (K-Means); áreas; publicações; quem ganha e quem perde; dados; sobre.

### Fórmulas

- Pilar do docente *i* na dimensão *k*: `P_ik = 100 × (nº de docentes com escala k menor) / (N − 1)`.
- **Score por Perfil:** `SP_i = Σ_k w_pk · P_ik`, com 80% do peso nos pilares dominantes do perfil *p*
  (centróide > 0,25 dp, proporcional ao centróide) e 20% igual entre os quatro pilares.
- **Score Institucional:** `SI_j = média_i∈j ( Σ_k w_k · P_ik )`, pesos padrão de 25% por pilar;
  `IC95% = SI_j ± 1,96 · dp/√n_j`. Implementação em `ranking_institucional.py`.

## Pipeline completo (regenera os dados)

| Etapa | Script | Saída |
|---|---|---|
| Extração (API do Integra, 40 portais) | `extractor.py`, `extrair_series.py` | `data/raw/`, `data/raw_series/` |
| Transformação | `transformer.py` | `data/processed/` |
| Análise fatorial, perfis, índices | `analise_exploratoria.py` | `data/analise/` |
| Séries temporais | `analise_series.py` | `data/analise/series/` |
| Ranking e Score por Perfil | `analise_ranking.py` | `data/analise/ranking/`, `data/dashboard/` |

No macOS, `rodar_extracao.command`, `atualizar_dados.command` e `abrir_dashboard.command` fazem tudo com dois cliques.

## Dados e uso

Os dados vêm de páginas públicas do Portal Integra mantidas pelas próprias instituições, com a data de coleta de setembro–outubro de 2026. Rankings individuais devem ser lidos como análise de sensibilidade de critérios de pontuação, e não como medida definitiva de mérito: o painel mostra como a posição de cada docente muda conforme a ponderação.
