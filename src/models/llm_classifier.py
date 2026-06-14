"""Classificador de sentimentos via LLM (Google Gemini API)."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Sequence

ALLOWED_LABELS = frozenset({"positivo", "neutro", "negativo"})

PROMPT_TEMPLATE = (
    "Classifique o sentimento do seguinte review de e-commerce em português.\n"
    "Categorias: positivo, neutro, negativo.\n\n"
    "Review: {text}"
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["positivo", "neutro", "negativo"],
        }
    },
    "required": ["label"],
}


@dataclass(frozen=True)
class LLMConfig:
    """Configuracao do classificador LLM."""
    model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.0
    max_tokens: int = 20
    prompt_template: str = PROMPT_TEMPLATE
    retry_attempts: int = 3
    retry_delay: float = 1.0
    fallback_label: str = "neutro"


class LLMClassifier:
    """Classificador zero-shot via Google Gemini (JSON estruturado).

    Nao carrega o cliente no __init__ — somente na primeira chamada a
    predict(), para permitir testes com mock sem a biblioteca instalada.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._client = None

    def _get_client(self):
        """Instancia o cliente Gemini, lendo a chave do ambiente."""
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai nao esta instalado. "
                "Execute: pip install google-genai"
            ) from exc
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY nao definida. "
                "Configure a variavel de ambiente antes de executar o experimento LLM."
            )
        self._client = genai.Client(api_key=api_key)
        return self._client

    def _parse_response(self, raw: str) -> str:
        """Extrai label do JSON retornado pelo Gemini."""
        try:
            parsed = json.loads(raw)
            label = str(parsed.get("label", "")).lower().strip()
            if label in ALLOWED_LABELS:
                return label
        except (json.JSONDecodeError, AttributeError):
            pass
        for token in re.split(r'[\s,.\-"\'{}:]+', raw.lower()):
            if token in ALLOWED_LABELS:
                return token
        return self.config.fallback_label

    def _classify_one(self, text: str) -> str:
        """Envia um texto para a API e retorna o label normalizado."""
        from google.genai import types

        client = self._get_client()
        prompt = self.config.prompt_template.format(text=text[:1000])
        config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
        )

        for attempt in range(self.config.retry_attempts):
            try:
                response = client.models.generate_content(
                    model=self.config.model,
                    contents=prompt,
                    config=config,
                )
                return self._parse_response(response.text)
            except Exception:
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay)
        return self.config.fallback_label

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Classifica uma sequencia de textos. Faz uma chamada de API por texto."""
        return [self._classify_one(t) for t in texts]

    def predict_one(self, text: str) -> str:
        """Classifica um unico texto."""
        return self._classify_one(text)


def build_llm_classifier(config: LLMConfig | None = None) -> LLMClassifier:
    """Instancia o classificador LLM com a configuracao fornecida."""
    return LLMClassifier(config)
