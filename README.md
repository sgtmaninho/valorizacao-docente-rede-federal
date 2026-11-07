# Valorização Docente na Rede Federal

Painel e pipeline de análise do TCC **"Ranking de docentes e instituições da Rede Federal por perfis de atuação"**.

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

O painel tem dois segmentos no menu do topo, acima do título; os filtros da barra lateral valem para os dois.

**Período da produção** (barra lateral): *desde o ingresso na Rede Federal* (padrão: só publicações e atividades a partir do
ano de entrada do docente na Rede, exceto a formação acadêmica) ou *currículo completo*.

**🏛️ Instituição** – dois rankings institucionais (região, instituição ou campus): **proporcional** ao número de
docentes (Score Institucional, SI = média, com intervalo de confiança de 95% e faixas de empate estatístico) e
**quantitativo bruto** (soma dos escores dos docentes); comparação de 2 a 4 instituições; regiões e
localização (capital × interior); funil e cobertura; séries temporais; dados agregados; sobre.

**👩‍🏫 Docente** – ranking de docentes pelo **Score por Perfil (SP)** e pelos índices de comparação, com ficha do
docente; comparação de professores; perfis (K-Means); áreas; publicações; quem ganha e quem perde; dados; sobre.

### Fórmulas

- Pilar do docente *i* na dimensão *k*: `P_ik = 100 × (nº de docentes com escala k menor) / (N − 1)`.
- **Score por Perfil:** `SP_i = Σ_k w_pk · P_ik`, com 80% do peso nos pilares dominantes do perfil *p*
  (centróide > 0,25 dp, proporcional ao centróide) e 20% igual entre os quatro pilares.
- **Score Institucional (proporcional):** `SI_j = média_i∈j ( Σ_k w_k · P_ik )`, pesos padrão de 25% por pilar;
  `IC95% = SI_j ± 1,96 · dp/√n_j`.
- **Score bruto:** `SB_j = Σ_i∈j ( Σ_k w_k · P_ik ) / 1000` (milhares de pontos). Implementação em `ranking_institucional.py`.
- **Recorte por ingresso:** ano de ingresso = 1º ano de vínculo empregatício numa instituição da Rede (Lattes);
  cada contagem é multiplicada pela fração dos itens datados a partir desse ano (`recorte_ingresso.py`).

## Pipeline completo (regenera os dados)

| Etapa | Script | Saída |
|---|---|---|
| Extração (API do Integra, 40 portais) | `extractor.py`, `extrair_series.py` | `data/raw/`, `data/raw_series/` |
| Transformação | `transformer.py` | `data/processed/` |
| Análise fatorial, perfis, índices | `analise_exploratoria.py` | `data/analise/` |
| Séries temporais | `analise_series.py` | `data/analise/series/` |
| Ranking e Score por Perfil | `analise_ranking.py` | `data/analise/ranking/`, `data/dashboard/` |
| Ano de ingresso na Rede (coleta) | `extrair_ingresso.py` | `data/raw_ingresso/` |
| Recorte por ingresso | `recorte_ingresso.py` + `TCC_CENARIO=ingresso` nas etapas acima | `data/cenario_ingresso/` |

No macOS, `rodar_extracao.command`, `atualizar_dados.command` e `abrir_dashboard.command` fazem tudo com dois cliques.

## Dados e uso

Os dados vêm de páginas públicas do Portal Integra mantidas pelas próprias instituições, com a data de coleta de setembro–outubro de 2026. Rankings individuais devem ser lidos como análise de sensibilidade de critérios de pontuação, e não como medida definitiva de mérito: o painel mostra como a posição de cada docente muda conforme a ponderação.
