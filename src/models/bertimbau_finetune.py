"""Wrapper de fine-tuning do BERTimbau para classificacao de sentimentos.

Utiliza ``transformers.Trainer`` para reduzir codigo boilerplate. O modelo
``neuralmind/bert-base-portuguese-cased`` e fine-tuned completo (head + camadas
do encoder), em contraste com ``bertimbau_embeddings.py`` que mantem o BERT
congelado.

``train_and_evaluate`` recebe textos e rotulos crus (strings) e devolve
predicoes no conjunto de teste tambem como strings, alem do tempo de
treinamento medido. Os imports pesados (``torch``, ``transformers``) sao
adiados para dentro da funcao para que importar este modulo nao falhe quando
as dependencias nao estiverem instaladas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence


DEFAULT_LABEL_ORDER: tuple[str, ...] = ("negativo", "neutro", "positivo")


@dataclass(frozen=True)
class FinetuneConfig:
    """Configuracao de fine-tuning do BERTimbau.

    Defaults compativeis com a sprint S07. ``device="cuda"`` e o ideal; caso
    nao haja GPU, ``train_and_evaluate`` faz fallback para CPU automaticamente
    e registra o hardware efetivamente usado em ``training_time_seconds`` e
    ``device_used``.
    """

    model_name: str = "neuralmind/bert-base-portuguese-cased"
    num_labels: int = 3
    num_epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    max_length: int = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    random_state: int = 42
    output_dir: str = "v2/outputs/bert-finetune-checkpoint"
    device: str = "cuda"


@dataclass(frozen=True)
class FinetuneOutput:
    """Predicoes e metadados produzidos por ``train_and_evaluate``."""

    predictions: list[str]
    training_time_seconds: float
    device_used: str


def _resolve_device(requested: str) -> str:
    """Devolve o device efetivamente disponivel.

    Se ``requested == "cuda"`` mas torch nao detecta GPU, faz fallback para
    ``"cpu"`` em vez de quebrar. Caller deve usar o ``device_used`` para
    registrar no meta do output.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - guard
        raise ImportError(
            "torch nao esta instalado. Execute: pip install torch transformers"
        ) from exc
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def train_and_evaluate(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    test_texts: Sequence[str],
    test_labels: Sequence[str],
    config: FinetuneConfig | None = None,
    *,
    label_order: Sequence[str] = DEFAULT_LABEL_ORDER,
) -> FinetuneOutput:
    """Faz fine-tuning do BERTimbau e devolve predicoes no conjunto de teste.

    Args:
        train_texts: textos ja normalizados de treino.
        train_labels: rotulos textuais (``negativo``/``neutro``/``positivo``).
        test_texts: textos de teste.
        test_labels: rotulos de teste (usados apenas para definir o tamanho
            do ``Dataset`` de avaliacao; metricas sao calculadas pelo caller).
        config: ``FinetuneConfig`` opcional. Default cobre o cenario da S07.
        label_order: ordem canonica das classes; define o mapeamento
            string <-> id usado pelo modelo.

    Returns:
        ``FinetuneOutput`` com predicoes textuais alinhadas a ``test_texts``,
        tempo de treinamento e device efetivamente usado.
    """

    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            "torch/transformers/datasets nao instalados. Execute:\n"
            "pip install torch transformers datasets accelerate"
        ) from exc

    cfg = config or FinetuneConfig()
    labels = list(label_order)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    device_used = _resolve_device(cfg.device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def _tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=cfg.max_length,
        )

    train_ds = Dataset.from_dict(
        {
            "text": list(train_texts),
            "label": [label2id[label] for label in train_labels],
        }
    )
    test_ds = Dataset.from_dict(
        {
            "text": list(test_texts),
            "label": [label2id[label] for label in test_labels],
        }
    )
    train_ds = train_ds.map(_tokenize, batched=True)
    test_ds = test_ds.map(_tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    import inspect as _inspect
    _ta_params = _inspect.signature(TrainingArguments.__init__).parameters
    _device_kwarg = (
        {"use_cpu": device_used != "cuda"}
        if "use_cpu" in _ta_params
        else {"no_cuda": device_used != "cuda"}
    )
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        seed=cfg.random_state,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        **_device_kwarg,
    )

    _trainer_params = _inspect.signature(Trainer.__init__).parameters
    _tokenizer_kwarg = (
        {"processing_class": tokenizer}
        if "processing_class" in _trainer_params
        else {"tokenizer": tokenizer}
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        **_tokenizer_kwarg,
    )

    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - started

    pred_output = trainer.predict(test_ds)
    logits = pred_output.predictions
    pred_ids = np.argmax(logits, axis=-1)
    predictions = [id2label[int(idx)] for idx in pred_ids]

    return FinetuneOutput(
        predictions=predictions,
        training_time_seconds=round(float(elapsed), 4),
        device_used=device_used,
    )


__all__ = [
    "DEFAULT_LABEL_ORDER",
    "FinetuneConfig",
    "FinetuneOutput",
    "train_and_evaluate",
]
