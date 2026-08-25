"""Reproducible numeric and categorical feature processing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

__all__ = ["CategoricalFeature", "FeatureEncoder", "NumericFeature"]


@dataclass(frozen=True)
class NumericFeature:
    name: str
    mean: float
    scale: float


@dataclass(frozen=True)
class CategoricalFeature:
    name: str
    levels: tuple[str, ...]

    @property
    def unknown_code(self) -> int:
        return len(self.levels)


@dataclass(frozen=True)
class FeatureEncoder:
    """Standardization and one-hot metadata owned by a fitted artifact.

    At fitting time :meth:`transform_columns` accepts string category values.
    JACT callables operate inside JAX, so categorical solve-time kwargs must be
    integer codes produced by :meth:`encode_category`; unseen values use the
    explicit final ``<unknown>`` column.
    """

    numeric: tuple[NumericFeature, ...] = ()
    categorical: tuple[CategoricalFeature, ...] = ()

    @classmethod
    def fit(
        cls,
        *,
        numeric: Mapping[str, ArrayLike] | None = None,
        categorical: Mapping[str, Sequence[Any]] | None = None,
    ) -> FeatureEncoder:
        numeric_specs: list[NumericFeature] = []
        for name, values in (numeric or {}).items():
            array = np.asarray(values, dtype=float)
            if array.ndim != 1 or array.size == 0:
                raise ValueError(f"Numeric feature '{name}' must be non-empty and 1D.")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Numeric feature '{name}' must be finite.")
            mean = float(np.mean(array))
            scale = float(np.std(array))
            numeric_specs.append(
                NumericFeature(name=name, mean=mean, scale=scale if scale > 0 else 1.0)
            )

        categorical_specs: list[CategoricalFeature] = []
        for name, values in (categorical or {}).items():
            strings = tuple(str(value) for value in values)
            if not strings:
                raise ValueError(f"Categorical feature '{name}' must be non-empty.")
            # Sorting gives a stable vocabulary independent of row order.
            levels = tuple(sorted(set(strings)))
            categorical_specs.append(CategoricalFeature(name=name, levels=levels))
        return cls(tuple(numeric_specs), tuple(categorical_specs))

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = [feature.name for feature in self.numeric]
        for feature in self.categorical:
            names.extend(
                f"{feature.name}={level}"
                for level in (*feature.levels, "<unknown>")
            )
        return tuple(names)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def encode_category(self, name: str, values: Sequence[Any]) -> Array:
        spec = self._categorical(name)
        lookup = {level: index for index, level in enumerate(spec.levels)}
        return jnp.asarray(
            [lookup.get(str(value), spec.unknown_code) for value in values],
            dtype=jnp.int32,
        )

    def transform_columns(self, columns: Mapping[str, Any]) -> Array:
        """Transform row columns into a dense ``(rows, features)`` matrix."""
        pieces: list[Array] = []
        row_count: int | None = None
        for spec in self.numeric:
            values = jnp.asarray(columns[spec.name])
            if values.ndim != 1:
                raise ValueError(f"Feature '{spec.name}' must be one-dimensional.")
            row_count = _consistent_rows(row_count, values.shape[0], spec.name)
            pieces.append(((values - spec.mean) / spec.scale)[:, None])
        for spec in self.categorical:
            raw = columns[spec.name]
            if _is_integer_array(raw):
                codes = jnp.asarray(raw, dtype=jnp.int32)
            else:
                codes = self.encode_category(spec.name, raw)
            if codes.ndim != 1:
                raise ValueError(f"Feature '{spec.name}' must be one-dimensional.")
            row_count = _consistent_rows(row_count, codes.shape[0], spec.name)
            valid_codes = jnp.where(
                (codes >= 0) & (codes < spec.unknown_code),
                codes,
                spec.unknown_code,
            )
            pieces.append(jnp.eye(spec.unknown_code + 1)[valid_codes])
        if not pieces:
            raise ValueError("Cannot transform columns with an empty encoder.")
        return jnp.concatenate(pieces, axis=-1)

    def grid(self, t: ArrayLike, d: ArrayLike, columns: Mapping[str, Any]) -> Array:
        """Build ``(batch, duration, features)`` solve-time feature grids."""
        del t  # Clock time is modelled separately by StructuredLogHazardModel.
        duration = jnp.asarray(d)
        if duration.ndim == 1:
            duration = duration[None, :]
        if duration.ndim != 2:
            raise ValueError("d must have shape (D,), (1, D), or (batch, D).")

        batch = _grid_batch_size(columns, self, duration_batch=duration.shape[0])
        if duration.shape[0] not in (1, batch):
            raise ValueError(
                "The leading duration dimension must be one or the "
                "covariate batch size."
            )
        width = duration.shape[-1]
        pieces: list[Array] = []
        for spec in self.numeric:
            values = _as_batch_column(columns[spec.name], batch, spec.name)
            normalized = (values - spec.mean) / spec.scale
            pieces.append(
                jnp.broadcast_to(normalized[:, None, None], (batch, width, 1))
            )
        for spec in self.categorical:
            codes = _as_batch_column(columns[spec.name], batch, spec.name).astype(
                jnp.int32
            )
            codes = jnp.where(
                (codes >= 0) & (codes < spec.unknown_code),
                codes,
                spec.unknown_code,
            )
            one_hot = jnp.eye(spec.unknown_code + 1)[codes]
            pieces.append(
                jnp.broadcast_to(
                    one_hot[:, None, :],
                    (batch, width, spec.unknown_code + 1),
                )
            )
        if not pieces:
            return jnp.empty((batch, width, 0), dtype=duration.dtype)
        return jnp.concatenate(pieces, axis=-1)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "numeric": [
                {"name": value.name, "mean": value.mean, "scale": value.scale}
                for value in self.numeric
            ],
            "categorical": [
                {"name": value.name, "levels": list(value.levels)}
                for value in self.categorical
            ],
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> FeatureEncoder:
        numeric = tuple(
            NumericFeature(
                str(value["name"]), float(value["mean"]), float(value["scale"])
            )
            for value in metadata.get("numeric", ())
        )
        if any(value.scale <= 0.0 for value in numeric):
            raise ValueError("Artifact numeric feature scales must be positive.")
        categorical = tuple(
            CategoricalFeature(
                str(value["name"]), tuple(str(level) for level in value["levels"])
            )
            for value in metadata.get("categorical", ())
        )
        return cls(numeric=numeric, categorical=categorical)

    def _categorical(self, name: str) -> CategoricalFeature:
        for feature in self.categorical:
            if feature.name == name:
                return feature
        raise ValueError(f"Unknown categorical feature '{name}'.")


def _consistent_rows(current: int | None, new: int, name: str) -> int:
    if current is not None and current != new:
        raise ValueError(f"Feature '{name}' has {new} rows; expected {current}.")
    return new


def _is_integer_array(value: Any) -> bool:
    return np.issubdtype(np.asarray(value).dtype, np.integer)


def _grid_batch_size(
    columns: Mapping[str, Any], encoder: FeatureEncoder, *, duration_batch: int = 1
) -> int:
    sizes: list[int] = []
    if duration_batch != 1:
        sizes.append(duration_batch)
    for name in [feature.name for feature in (*encoder.numeric, *encoder.categorical)]:
        if name not in columns:
            raise ValueError(f"Missing required solve-time covariate '{name}'.")
        value = jnp.asarray(columns[name])
        if value.ndim > 1:
            raise ValueError(f"Solve-time covariate '{name}' must be scalar or 1D.")
        if value.ndim == 1:
            sizes.append(value.shape[0])
    if len(set(sizes)) > 1:
        raise ValueError("Non-scalar solve-time covariates disagree on batch size.")
    return sizes[0] if sizes else 1


def _as_batch_column(value: Any, batch: int, name: str) -> Array:
    array = jnp.asarray(value)
    if array.ndim == 0:
        return jnp.broadcast_to(array, (batch,))
    if array.ndim != 1 or array.shape[0] != batch:
        raise ValueError(
            f"Solve-time covariate '{name}' must be scalar or shape ({batch},)."
        )
    return array
