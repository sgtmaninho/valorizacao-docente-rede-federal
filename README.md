# Valorização Docente na Rede Federal

Painel e pipeline de análise do TCC **"A dimensão multidimensional da valorização docente: ranking por perfis de atuação e o efeito da escolha de pesos na Rede Federal de Educação Profissional, Científica e Tecnológica"**.

- **Autor:** Edvaldo Pinto Maninho Júnior – Tecnologia em Sistemas para Internet, Instituto Federal de Brasília (IFB)
- **Orientador:** Prof. Gustavo Henrique Dornelas de Deus

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

Abas: Ranking, Comparar professores, Perfis (Clusters), Classificação por Área, Publicações, Insights, Funil e Cobertura, Séries temporais, Instituições e Campi, Dados Detalhados e Sobre.

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
