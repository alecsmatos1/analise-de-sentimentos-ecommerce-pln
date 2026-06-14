"""Wrapper de modelos lineares para classificacao."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

LinearModelName = Literal["logistic_regression", "linear_svc"]
LinearEstimator = Union[LogisticRegression, LinearSVC]


@dataclass(frozen=True)
class LinearModelConfig:
    """Configuracao do modelo linear.

    `name="logistic_regression"` e o default e cobre a primeira opcao da sprint.
    Trocar para `name="linear_svc"` mantem a mesma interface externa.
    """

    name: LinearModelName = "logistic_regression"
    class_weight: Optional[Union[str, dict]] = "balanced"
    random_state: int = 42
    max_iter: int = 1000
    C: float = 1.0


def build_linear_model(config: Optional[LinearModelConfig] = None) -> LinearEstimator:
    """Instancia um classificador linear pronto para uso em pipeline."""

    cfg = config or LinearModelConfig()
    if cfg.name == "logistic_regression":
        return LogisticRegression(
            class_weight=cfg.class_weight,
            random_state=cfg.random_state,
            max_iter=cfg.max_iter,
            C=cfg.C,
        )
    if cfg.name == "linear_svc":
        return LinearSVC(
            class_weight=cfg.class_weight,
            random_state=cfg.random_state,
            max_iter=cfg.max_iter,
            C=cfg.C,
        )
    raise ValueError(
        f"modelo linear desconhecido: {cfg.name!r}. "
        "Use 'logistic_regression' ou 'linear_svc'."
    )


__all__ = [
    "LinearEstimator",
    "LinearModelConfig",
    "LinearModelName",
    "build_linear_model",
]
