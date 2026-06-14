"""Fixture sintetica de reviews para testes da v2.

Os textos sao demonstrativos e nao vem de nenhuma base comercial real. A
fixture serve apenas para validar contratos de colunas, rotulos e o mapeamento
de notas para classes definido em ``v2/avaliacao.md``.
"""

from __future__ import annotations

from typing import Final, List, Mapping

SAMPLE_REVIEWS: Final[List[Mapping[str, object]]] = [
    {
        "raw_text": "Produto excelente, chegou antes do prazo!",
        "clean_text": "produto excelente chegou antes do prazo",
        "label": "positivo",
        "score": 5,
        "source": "demo",
    },
    {
        "raw_text": "Gostei bastante, recomendo a loja.",
        "clean_text": "gostei bastante recomendo a loja",
        "label": "positivo",
        "score": 4,
        "source": "demo",
    },
    {
        "raw_text": "Atende ao que promete pelo preco.",
        "clean_text": "atende ao que promete pelo preco",
        "label": "positivo",
        "score": 4,
        "source": "demo",
    },
    {
        "raw_text": "Produto entregue, sem mais detalhes.",
        "clean_text": "produto entregue sem mais detalhes",
        "label": "neutro",
        "score": 3,
        "source": "demo",
    },
    {
        "raw_text": "Embalagem padrao, item conforme descricao.",
        "clean_text": "embalagem padrao item conforme descricao",
        "label": "neutro",
        "score": 3,
        "source": "demo",
    },
    {
        "raw_text": "Nao recebi o produto e nao tive retorno.",
        "clean_text": "nao recebi o produto e nao tive retorno",
        "label": "negativo",
        "score": 1,
        "source": "demo",
    },
    {
        "raw_text": "Veio com defeito, quero devolucao.",
        "clean_text": "veio com defeito quero devolucao",
        "label": "negativo",
        "score": 1,
        "source": "demo",
    },
    {
        "raw_text": "Demorou demais e o suporte nao respondeu.",
        "clean_text": "demorou demais e o suporte nao respondeu",
        "label": "negativo",
        "score": 2,
        "source": "demo",
    },
]
