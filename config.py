"""Configuração do projeto: instituições da Rede Federal com instância do Portal Integra.

A plataforma Integra (Rede Integra, desenvolvida pelo IFRS) roda uma instância por
instituição em https://integra.<dominio>.edu.br com a MESMA API. O extrator testa
cada uma antes de extrair e pula as que não responderem — por isso a lista inclui
também instituições candidatas (quanto mais, mais docentes e mais lugares).
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# sigla -> (nome, UF, domínio)
INSTITUICOES = {
    # Confirmadas com dados reais
    "IFB":   ("Instituto Federal de Brasília", "DF", "ifb"),
    "IFBA":  ("Instituto Federal da Bahia", "BA", "ifba"),
    "IFPB":  ("Instituto Federal da Paraíba", "PB", "ifpb"),
    "IFF":   ("Instituto Federal Fluminense", "RJ", "iff"),
    "IFRS":  ("Instituto Federal do Rio Grande do Sul", "RS", "ifrs"),
    "IFPR":  ("Instituto Federal do Paraná", "PR", "ifpr"),
    "IFMG":  ("Instituto Federal de Minas Gerais", "MG", "ifmg"),
    "IFSP":  ("Instituto Federal de São Paulo", "SP", "ifsp"),
    "IFG":   ("Instituto Federal de Goiás", "GO", "ifg"),
    "IFRN":  ("Instituto Federal do Rio Grande do Norte", "RN", "ifrn"),
    "IFCE":  ("Instituto Federal do Ceará", "CE", "ifce"),
    "IFPE":  ("Instituto Federal de Pernambuco", "PE", "ifpe"),
    "IFAL":  ("Instituto Federal de Alagoas", "AL", "ifal"),
    "IFAM":  ("Instituto Federal do Amazonas", "AM", "ifam"),
    "IFPA":  ("Instituto Federal do Pará", "PA", "ifpa"),
    "IFPI":  ("Instituto Federal do Piauí", "PI", "ifpi"),
    "IFTO":  ("Instituto Federal do Tocantins", "TO", "ifto"),
    "IFRO":  ("Instituto Federal de Rondônia", "RO", "ifro"),
    "IFRR":  ("Instituto Federal de Roraima", "RR", "ifrr"),
    # Candidatas (testadas automaticamente)
    "IFAC":  ("Instituto Federal do Acre", "AC", "ifac"),
    "IFAP":  ("Instituto Federal do Amapá", "AP", "ifap"),
    "IFMA":  ("Instituto Federal do Maranhão", "MA", "ifma"),
    "IFS":   ("Instituto Federal de Sergipe", "SE", "ifs"),
    "IFBAIANO": ("Instituto Federal Baiano", "BA", "ifbaiano"),
    "IFSERTAOPE": ("Instituto Federal do Sertão Pernambucano", "PE", "ifsertao-pe"),
    "IFMT":  ("Instituto Federal de Mato Grosso", "MT", "ifmt"),
    "IFMS":  ("Instituto Federal de Mato Grosso do Sul", "MS", "ifms"),
    "IFGOIANO": ("Instituto Federal Goiano", "GO", "ifgoiano"),
    "IFES":  ("Instituto Federal do Espírito Santo", "ES", "ifes"),
    "IFRJ":  ("Instituto Federal do Rio de Janeiro", "RJ", "ifrj"),
    "IFNMG": ("Instituto Federal do Norte de Minas Gerais", "MG", "ifnmg"),
    "IFSUDESTEMG": ("Instituto Federal do Sudeste de Minas Gerais", "MG", "ifsudestemg"),
    "IFSULDEMINAS": ("Instituto Federal do Sul de Minas Gerais", "MG", "ifsuldeminas"),
    "IFTM":  ("Instituto Federal do Triângulo Mineiro", "MG", "iftm"),
    "IFSC":  ("Instituto Federal de Santa Catarina", "SC", "ifsc"),
    "IFC":   ("Instituto Federal Catarinense", "SC", "ifc"),
    "IFSUL": ("Instituto Federal Sul-rio-grandense", "RS", "ifsul"),
    "IFFAR": ("Instituto Federal Farroupilha", "RS", "iffarroupilha"),
    # CEFETs (domínio completo, fora de .edu.br)
    "CEFETRJ": ("Centro Federal de Educação Tecnológica Celso Suckow da Fonseca", "RJ", "cefet-rj.br"),
    "CEFETMG": ("Centro Federal de Educação Tecnológica de Minas Gerais", "MG", "cefetmg.br"),
}


def base_url(sigla: str) -> str:
    dom = INSTITUICOES[sigla][2]
    return f"https://integra.{dom}" if "." in dom else f"https://integra.{dom}.edu.br"


# Extração
PAGE_SIZE = 100            # pessoas por página na listagem
WORKERS_POR_INST = 3       # requisições simultâneas dentro de uma instituição
INSTS_PARALELAS = 12      # instituições extraídas ao mesmo tempo (servidores distintos)
TIMEOUT = 40
MAX_TENTATIVAS = 4
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/128.0 Safari/537.36 TCC-TSI-IFB/1.0 (pesquisa academica)")
