"""Classificador de sentimentos via LLM (OpenAI API)."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Sequence

ALLOWED_LABELS = frozenset({"positivo", "neutro", "negativo"})

PROMPT_TEMPLATE = (
    "Classifique o sentimento do seguinte review de e-commerce em português.\n"
    "Responda com exatamente uma palavra: positivo, neutro ou negativo.\n\n"
    "Review: {text}\n\n"
    "Sentimento:"
)


@dataclass(frozen=True)
class LLMConfig:
    """Configuracao do classificador LLM."""
    model: str = "gpt-4.1-nano"
    temperature: float = 0.0
    max_tokens: int = 10
    prompt_template: str = PROMPT_TEMPLATE
    retry_attempts: int = 3
    retry_delay: float = 1.0
    fallback_label: str = "neutro"


class LLMClassifier:
    """Classificador zero-shot via OpenAI Chat Completions.

    Nao carrega o cliente no __init__ — somente na primeira chamada
    a predict(), para permitir testes com mock sem a biblioteca instalada.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._client = None

    def _get_client(self):
        """Instancia o cliente OpenAI, lendo a chave do ambiente."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai nao esta instalado. Execute: pip install openai"
            ) from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY nao definida. "
                "Configure a variavel de ambiente antes de executar o experimento LLM."
            )
        if self._client is None:
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _classify_one(self, text: str) -> str:
        """Envia um texto para a API e retorna o label normalizado."""
        client = self._get_client()
        cfg = self.config
        prompt = cfg.prompt_template.format(text=text[:1000])

        for attempt in range(cfg.retry_attempts):
            try:
                response = client.chat.completions.create(
                    model=cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
                raw = response.choices[0].message.content.strip().lower()
                # normalizar: extrair primeira palavra que seja um label valido
                for token in re.split(r"[\s,.\-]+", raw):
                    if token in ALLOWED_LABELS:
                        return token
                return cfg.fallback_label
            except Exception:
                if attempt < cfg.retry_attempts - 1:
                    time.sleep(cfg.retry_delay)
        return cfg.fallback_label

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Classifica uma sequencia de textos. Faz uma chamada de API por texto."""
        return [self._classify_one(t) for t in texts]

    def predict_one(self, text: str) -> str:
        """Classifica um unico texto."""
        return self._classify_one(text)


def build_llm_classifier(config: LLMConfig | None = None) -> LLMClassifier:
    """Instancia o classificador LLM com a configuracao fornecida."""
    return LLMClassifier(config)
