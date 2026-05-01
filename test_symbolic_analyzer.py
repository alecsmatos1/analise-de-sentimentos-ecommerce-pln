from __future__ import annotations

from sentiment_analyzer import analyze_symbolic_sentiment


REQUIRED_KEYS = {
    "label",
    "score",
    "confidence",
    "positive_score",
    "negative_score",
    "neutral_evidence",
    "rule_hits",
}

CASES = [
    ("positivo simples", "Produto excelente, chegou rapido.", "positivo", True),
    ("pros positivo", "Gostei: produto otimo.", "positivo", True),
    ("entrega positiva", "Chegou antes do prazo, super recomendo.", "positivo", True),
    ("valor positivo", "Bom custo beneficio e preco justo.", "positivo", True),
    ("concessao positiva", "Embora tenha demorado, o produto e otimo.", "positivo", True),
    ("nao recebimento", "Nao recebi o produto.", "negativo", True),
    ("contraste negativo", "O produto e bonito, mas nao funciona.", "negativo", True),
    ("pros negativo", "Nao gostei: produto ruim.", "negativo", True),
    ("defeito e devolucao", "Veio com defeito e quero devolucao.", "negativo", True),
    ("recomendacao negativa", "Nao recomendo, dinheiro jogado fora.", "negativo", True),
    ("ironia conservadora", "Produto excelente! Chegou quebrado e nao funciona.", "negativo", True),
    ("mudanca temporal negativa", "No comeco era bom, mas depois parou de funcionar.", "negativo", True),
    ("condicional baixa certeza", "Seria otimo se a bateria durasse mais.", "neutro", True),
    ("factual neutro", "Produto entregue ontem.", "neutro", True),
    ("texto vazio", "", "neutro", False),
]


def main() -> None:
    for description, text, expected_label, expect_hits in CASES:
        result = analyze_symbolic_sentiment(text)
        missing_keys = REQUIRED_KEYS - set(result)
        assert not missing_keys, f"{description}: chaves ausentes: {sorted(missing_keys)}"
        assert result["label"] == expected_label, (
            f"{description}: esperado {expected_label}, obtido {result['label']} "
            f"para texto {text!r}; resultado={result}"
        )
        assert isinstance(result["rule_hits"], list), f"{description}: rule_hits deve ser lista"
        if expect_hits:
            assert result["rule_hits"], f"{description}: esperava pelo menos uma regra acionada"

    print(f"OK - {len(CASES)} testes do analisador simbolico passaram")


if __name__ == "__main__":
    main()
