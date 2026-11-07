"""Recorte temporal pelo ingresso na Rede Federal (pedido do orientador, 06/11/2026).

"Considerar somente publicações e atividades a partir da data de entrada na Rede Federal, exceto a
formação acadêmica."

1. ano_ingresso: a partir das atuações profissionais do Lattes (data/raw_ingresso/, coletadas por
   extrair_ingresso.py), é o menor ano de início de um vínculo empregatício (servidor/professor) numa
   instituição da Rede Federal de Educação Profissional, Científica e Tecnológica (IFs, CEFETs, UTFPR,
   Colégio Pedro II e as antigas escolas técnicas e agrotécnicas federais). Vínculos de professor
   substituto, temporário, visitante, bolsista, estagiário ou colaborador não contam.
       fonte = "vinculo"      vínculo empregatício encontrado
       fonte = "atuacao"      só há atuação sem vínculo empregatício marcado na Rede (usa o 1º ano dela)
       fonte = "sem_registro" nenhuma atuação na Rede no Lattes -> o docente fica com o currículo inteiro
2. Fração após o ingresso: para cada tipo de item, f = (itens datados a partir do ingresso) / (itens datados).
   Os indicadores com Qualis usam o ano do próprio item quando o portal o informa (qualis_ano, 12
   instituições); senão, a fração do tipo correspondente no currículo (artigos, livros, capítulos,
   eventos, produção técnica). Itens sem ano em nenhum dos dois recebem a fração geral do docente.
3. Contagem recortada = round(contagem original × f). A titulação (formação acadêmica) não muda.

Uso:  python recorte_ingresso.py          -> grava data/ingresso_docentes.csv e um resumo
      (o transformer.py chama aplicar() quando TCC_CENARIO=ingresso)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime

import pandas as pd

import config as C

DIR_ING = C.BASE_DIR / "data" / "raw_ingresso"
DIR_SER = C.BASE_DIR / "data" / "raw_series"
SAIDA = C.BASE_DIR / "data" / "ingresso_docentes.csv"
ANO_MAX = datetime.now().year

RX_REDE = re.compile(
    r"instituto federal|centro federal de educa|\bcefet|escola t[eé]cnica federal|"
    r"escola agrot[eé]cnica federal|escola agr[ií]cola federal|universidade tecnol[oó]gica federal|\butfpr\b|"
    r"col[eé]gio pedro ii|escola t[eé]cnica vinculada|escola industrial federal", re.I)
RX_SIGLA_IF = re.compile(r"\bIF ?(?:[A-Z]{1,10}|Goiano|Baiano|Farroupilha|Fluminense|Catarinense|Sudeste|Sul)\b")  # IFB, IFSP, IF Goiano...
RX_EXCLUI = re.compile(r"substitut|tempor[aá]ri|visitante|bolsist|estagi|volunt[aá]ri|colaborador|tutor|"
                       r"p[oó]s-?doutor|aluno|estudante|discente|terceiriz", re.I)


def _ano(v):
    try:
        a = int(str(v)[:4])
        return a if 1950 <= a <= ANO_MAX else None
    except (TypeError, ValueError):
        return None


def eh_rede(nome: str) -> bool:
    nome = nome or ""
    return bool(RX_REDE.search(nome) or RX_SIGLA_IF.search(nome)) and not re.search(r"universidade federal (?!tecnol)", nome, re.I)


def ano_ingresso(atuacoes) -> tuple[int | None, str, str | None]:
    if not atuacoes:
        return None, "sem_registro", None
    vinc, atu = [], []
    for a in atuacoes:
        nome = a.get("nomeInstituicao") or ""
        if not eh_rede(nome):
            continue
        for v in a.get("vinculos") or []:
            ano = _ano(v.get("anoInicio"))
            if not ano:
                continue
            texto = " ".join(str(v.get(k, "")) for k in ("tipoDeVinculo", "enquadramentoFuncional",
                                                        "outroVinculoInformado", "outroEnquadramentoFuncionalInformado"))
            if RX_EXCLUI.search(texto) or v.get("tipoDeVinculo") in ("COLABORADOR", "PROFESSOR_VISITANTE"):
                continue
            atu.append((ano, nome))
            if v.get("flagVinculoEmpregaticio") == "SIM" or v.get("tipoDeVinculo") == "SERVIDOR_PUBLICO" \
                    or re.search(r"servidor|efetiv|EBTT|D[I1]-|concurs", texto, re.I):
                vinc.append((ano, nome))
    if vinc:
        a, n = min(vinc); return a, "vinculo", n
    if atu:
        a, n = min(atu); return a, "atuacao", n
    return None, "sem_registro", None


# rótulo do indicador (Integra) -> chave do currículo usada como fração de reserva
RESERVA_IND = [
    (r"^Artigo", r"^pb\.artigosPublicados"),
    (r"^Livro", r"^pb\.livrosECapitulos\.livrosPublicados"),
    (r"^Cap[ií]t", r"^pb\.livrosECapitulos\.capitulos"),
    (r"evento|resumo", r"^pb\.trabalhosEmEventos"),
    (r"software|Patente|cultivar|desenho|marca", r"^pt\.(software|patente|cultivar|desenho|marca|registro)"),
    (r"t[eé]cnic", r"^pt\."),
]


def _frac(series: dict, ing: int) -> tuple[float, int] | None:
    tot = sum(series.values())
    if not tot:
        return None
    return sum(n for y, n in series.items() if int(y) >= ing) / tot, tot


def _frac_regex(cv_ano: dict, rx: str, ing: int):
    s = Counter()
    for k, ser in cv_ano.items():
        if re.search(rx, k):
            s.update({int(y): n for y, n in ser.items()})
    return _frac(s, ing)


def carregar() -> pd.DataFrame:
    """Lê raw_ingresso (+ qualis_ano de raw_series) e devolve 1 linha por docente com ano e frações."""
    linhas = []
    qualis = {}
    for arq in DIR_SER.glob("*.jsonl"):
        for l in arq.open(encoding="utf-8"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("qualis_ano"):
                qualis[r["pessoa_uid"]] = r["qualis_ano"]
    for arq in sorted(DIR_ING.glob("*.jsonl")):
        for l in arq.open(encoding="utf-8"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if not r.get("ok_cv"):
                continue
            ano, fonte, inst = ano_ingresso(r.get("atuacoes"))
            linhas.append({"pessoa_uid": r["pessoa_uid"], "ano_ingresso": ano, "fonte_ingresso": fonte,
                           "instituicao_ingresso": inst, "_cv_ano": r.get("cv_ano") or {},
                           "_qualis_ano": qualis.get(r["pessoa_uid"]) or {}})
    return pd.DataFrame(linhas).drop_duplicates("pessoa_uid", keep="last")


def recortar_registro(r: dict, info: dict) -> dict:
    """Devolve uma cópia do registro bruto (data/raw) com as contagens recortadas no ingresso."""
    ing = info.get("ano_ingresso")
    r = dict(r)
    r["ano_ingresso"], r["fonte_ingresso"] = ing, info.get("fonte_ingresso", "sem_registro")
    if not ing or pd.isna(ing):
        return r
    ing = int(ing)
    cv_ano, q_ano = info.get("_cv_ano") or {}, info.get("_qualis_ano") or {}
    geral = Counter()
    for ser in cv_ano.values():
        geral.update({int(y): n for y, n in ser.items()})
    f_geral = (_frac(geral, ing) or (1.0, 0))[0]

    ind = r.get("indicadores")
    if ind:
        novo = {}
        for rot, v in ind.items():
            f = _frac(q_ano.get(rot) or {}, ing)
            if f is None:
                for rx_i, rx_cv in RESERVA_IND:
                    if re.search(rx_i, rot, re.I):
                        f = _frac_regex(cv_ano, rx_cv, ing)
                        break
            novo[rot] = int(round(v * (f[0] if f else f_geral)))
        r["indicadores"] = novo
    cv = r.get("cv")
    if cv and cv.get("contagens"):
        cv = dict(cv)
        cont = {}
        for k, v in cv["contagens"].items():
            if k.startswith("dc.formacaoComplementar"):      # formação: fora do recorte
                cont[k] = v
                continue
            f = _frac(cv_ano.get(k) or {}, ing)
            cont[k] = int(round(v * (f[0] if f else f_geral)))
        cv["contagens"] = cont
        r["cv"] = cv
    return r


def main():
    df = carregar()
    df[["pessoa_uid", "ano_ingresso", "fonte_ingresso", "instituicao_ingresso"]].to_csv(SAIDA, index=False)
    print(f"{len(df)} docentes com currículo lido em {DIR_ING}")
    print(df["fonte_ingresso"].value_counts(normalize=True).round(3).to_string())
    print(df["ano_ingresso"].describe().round(1).to_string())
    print(f"Gravado: {SAIDA}")


if __name__ == "__main__":
    main()
