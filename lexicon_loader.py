from __future__ import annotations

import unicodedata
import warnings
from pathlib import Path


# ---------------------------------------------------------------------------
# Formato dos arquivos (inspecionado em 2026-06-04)
#
# OpLexicon v3.0 — oplexicon_v3.0.txt
#   Separador: vírgula
#   Colunas:   term, pos_class, polarity, annotation_type
#   Exemplo:   =D,emot,1,A
#   Exemplo:   excelente,adj,1,M
#   polarity:  1 (positivo), -1 (negativo), 0 (neutro/ignorado)
#
# SentiLex-lem-PT02 — SentiLex-lem-PT02.txt
#   Formato:   lemma.PoS=class;TG=targets;POL:N0=valor;...;ANOT=tipo
#   Exemplo:   abafado.PoS=Adj;TG=HUM:N0;POL:N0=-1;ANOT=MAN
#   Extração:  lemma = tudo antes do primeiro "."
#              polarity = valor após "POL:N0="
#   Encoding:  UTF-8
# ---------------------------------------------------------------------------


def _nfkd(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def load_oplexicon(path: Path | str) -> dict[str, float]:
    """
    Carrega OpLexicon v3.0 e retorna {termo_normalizado: valor_com_sinal}.

    Convenção de sinal (decisão D1 em docs/decisoes.md):
      +1.0 = termo positivo
      -1.0 = termo negativo
      Entradas com polarity=0 são ignoradas.
    """
    path = Path(path)
    if not path.exists():
        warnings.warn(f"OpLexicon não encontrado: {path}. Retornando dicionário vazio.")
        return {}

    lexicon: dict[str, float] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            term = parts[0].strip()
            try:
                polarity = int(parts[2].strip())
            except ValueError:
                continue
            if polarity == 0 or not term:
                continue
            key = _nfkd(term)
            if key:
                lexicon[key] = float(polarity)  # +1.0 ou -1.0

    return lexicon


def load_sentilex(path: Path | str) -> dict[str, float]:
    """
    Carrega SentiLex-lem-PT02 e retorna {lema_normalizado: valor_com_sinal}.

    Formato de entrada: lemma.PoS=class;TG=targets;POL:N0=valor;...;ANOT=tipo
    Extrai o lema (antes do primeiro ".") e o valor de POL:N0.

    Convenção de sinal igual à do load_oplexicon (decisão D1).
    """
    path = Path(path)
    if not path.exists():
        warnings.warn(f"SentiLex não encontrado: {path}. Retornando dicionário vazio.")
        return {}

    lexicon: dict[str, float] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or "POL:N0=" not in line:
                continue

            # Extrai lema: tudo antes do primeiro "."
            lemma = line.split(".")[0].strip()
            if not lemma:
                continue

            # Extrai POL:N0
            try:
                pol_part = [p for p in line.split(";") if p.startswith("POL:N0=")][0]
                polarity = int(pol_part.split("=")[1])
            except (IndexError, ValueError):
                continue

            if polarity == 0:
                continue

            key = _nfkd(lemma)
            if key:
                # Se o lema já existe, manter o de maior magnitude
                if key not in lexicon or abs(polarity) > abs(lexicon[key]):
                    lexicon[key] = float(polarity)

    return lexicon


if __name__ == "__main__":
    from data_processing import DATA_DIR

    lex_dir = DATA_DIR / "lexicons"
    op = load_oplexicon(lex_dir / "oplexicon_v3.0.txt")
    sl = load_sentilex(lex_dir / "SentiLex-lem-PT02.txt")

    print(f"\nOpLexicon v3.0: {len(op)} entradas")
    pos = [(k, v) for k, v in op.items() if v > 0][:5]
    neg = [(k, v) for k, v in op.items() if v < 0][:5]
    print(f"  Positivos (5 ex): {pos}")
    print(f"  Negativos (5 ex): {neg}")

    print(f"\nSentiLex-lem-PT02: {len(sl)} entradas")
    pos = [(k, v) for k, v in sl.items() if v > 0][:5]
    neg = [(k, v) for k, v in sl.items() if v < 0][:5]
    print(f"  Positivos (5 ex): {pos}")
    print(f"  Negativos (5 ex): {neg}")
