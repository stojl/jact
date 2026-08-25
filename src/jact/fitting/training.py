"""Small, dependency-free AdamW training utilities."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .data import CanonicalBatch

__all__ = [
    "AdamWConfig",
    "FitResult",
    "Trainer",
    "TrainingHistory",
    "TrainState",
    "adamw_init",
    "make_train_step",
]

Params = Any
LossFn = Callable[[Params, CanonicalBatch], Array]


@dataclass(frozen=True)
class AdamWConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    clip_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    warmup_steps: int = 0
    total_steps: int = 0
    minimum_learning_rate_ratio: float = 0.0
    ema_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0 or self.clip_norm < 0.0:
            raise ValueError("weight_decay and clip_norm must be non-negative.")
        if not (0.0 <= self.beta1 < 1.0 and 0.0 <= self.beta2 < 1.0):
            raise ValueError("beta1 and beta2 must be in [0, 1).")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if self.warmup_steps < 0 or self.total_steps < 0:
            raise ValueError("Schedule step counts must be non-negative.")
        if not (0.0 <= self.minimum_learning_rate_ratio <= 1.0):
            raise ValueError("minimum_learning_rate_ratio must be in [0, 1].")
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError("ema_decay must be in [0, 1).")


class TrainState(NamedTuple):
    step: Array
    first_moment: Params
    second_moment: Params
    ema_params: Params


@dataclass(frozen=True)
class TrainingHistory:
    training_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]


@dataclass(frozen=True)
class FitResult:
    params: Params
    state: TrainState
    history: TrainingHistory
    best_epoch: int


def adamw_init(params: Params) -> TrainState:
    zeros = jax.tree.map(jnp.zeros_like, params)
    ema = jax.tree.map(lambda value: jnp.array(value), params)
    return TrainState(
        step=jnp.asarray(0, dtype=jnp.int32),
        first_moment=zeros,
        second_moment=zeros,
        ema_params=ema,
    )


def make_train_step(
    loss_fn: LossFn,
    config: AdamWConfig,
) -> Callable[
    [Params, TrainState, CanonicalBatch], tuple[Params, TrainState, Array, Array]
]:
    """Create a jitted update returning params, state, loss, and gradient norm."""

    @jax.jit
    def train_step(
        params: Params, state: TrainState, batch: CanonicalBatch
    ) -> tuple[Params, TrainState, Array, Array]:
        loss, gradients = jax.value_and_grad(loss_fn)(params, batch)
        gradient_norm = _global_norm(gradients)
        if config.clip_norm > 0.0:
            multiplier = jnp.minimum(1.0, config.clip_norm / (gradient_norm + 1e-12))
            gradients = jax.tree.map(lambda gradient: gradient * multiplier, gradients)

        step = state.step + 1
        first = jax.tree.map(
            lambda old, gradient: config.beta1 * old + (1.0 - config.beta1) * gradient,
            state.first_moment,
            gradients,
        )
        second = jax.tree.map(
            lambda old, gradient: config.beta2 * old
            + (1.0 - config.beta2) * gradient**2,
            state.second_moment,
            gradients,
        )
        first_correction = 1.0 - config.beta1**step
        second_correction = 1.0 - config.beta2**step
        learning_rate = _learning_rate(step, config)
        updated = jax.tree.map(
            lambda parameter, m, v: parameter
            - learning_rate
            * (m / first_correction / (jnp.sqrt(v / second_correction) + config.epsilon)
               + config.weight_decay * parameter),
            params,
            first,
            second,
        )
        if config.ema_decay > 0.0:
            ema = jax.tree.map(
                lambda old, new: config.ema_decay * old
                + (1.0 - config.ema_decay) * new,
                state.ema_params,
                updated,
            )
        else:
            ema = updated
        return updated, TrainState(step, first, second, ema), loss, gradient_norm

    return train_step


@dataclass(frozen=True)
class Trainer:
    """Fixed-shape minibatch trainer with validation early stopping."""

    loss_fn: LossFn
    optimizer: AdamWConfig = AdamWConfig()

    def fit(
        self,
        params: Params,
        training_batches: Iterable[CanonicalBatch],
        *,
        epochs: int,
        validation_batches: Iterable[CanonicalBatch] = (),
        patience: int | None = None,
        use_ema: bool = True,
    ) -> FitResult:
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if patience is not None and patience <= 0:
            raise ValueError("patience must be positive or None.")
        train = tuple(training_batches)
        validation = tuple(validation_batches)
        if not train:
            raise ValueError("training_batches must not be empty.")
        _check_fixed_shapes(train, "training")
        if validation:
            _check_fixed_shapes(validation, "validation")

        update = make_train_step(self.loss_fn, self.optimizer)
        evaluate = jax.jit(self.loss_fn)
        state = adamw_init(params)
        training_history: list[float] = []
        validation_history: list[float] = []
        best_loss = math.inf
        best_epoch = -1
        best_params = params

        for epoch in range(epochs):
            epoch_losses: list[float] = []
            for batch in train:
                params, state, loss, _ = update(params, state, batch)
                epoch_losses.append(float(loss))
            training_history.append(sum(epoch_losses) / len(epoch_losses))

            candidate = state.ema_params if use_ema else params
            if validation:
                losses = [float(evaluate(candidate, batch)) for batch in validation]
                score = sum(losses) / len(losses)
            else:
                score = training_history[-1]
            validation_history.append(score)
            if math.isfinite(score) and score < best_loss:
                best_loss = score
                best_epoch = epoch
                best_params = jax.tree.map(lambda value: jnp.array(value), candidate)
            if patience is not None and epoch - best_epoch >= patience:
                break

        if best_epoch < 0:
            raise FloatingPointError("Training never produced a finite epoch loss.")
        return FitResult(
            params=best_params,
            state=state,
            history=TrainingHistory(
                tuple(training_history), tuple(validation_history)
            ),
            best_epoch=best_epoch,
        )


def _global_norm(tree: Params) -> Array:
    squared = [jnp.sum(value**2) for value in jax.tree.leaves(tree)]
    return jnp.sqrt(sum(squared, jnp.asarray(0.0)))


def _learning_rate(step: Array, config: AdamWConfig) -> Array:
    rate = jnp.asarray(config.learning_rate)
    if config.warmup_steps:
        rate = rate * jnp.minimum(1.0, step / config.warmup_steps)
    if config.total_steps > config.warmup_steps:
        progress = jnp.clip(
            (step - config.warmup_steps)
            / (config.total_steps - config.warmup_steps),
            0.0,
            1.0,
        )
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        ratio = config.minimum_learning_rate_ratio + (
            1.0 - config.minimum_learning_rate_ratio
        ) * cosine
        rate = rate * ratio
    return rate


def _check_fixed_shapes(batches: tuple[CanonicalBatch, ...], label: str) -> None:
    reference = jax.tree.map(lambda value: getattr(value, "shape", None), batches[0])
    for batch in batches[1:]:
        shapes = jax.tree.map(lambda value: getattr(value, "shape", None), batch)
        if shapes != reference:
            raise ValueError(f"All {label} batches must have identical shapes.")
