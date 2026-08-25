"""Versioned, non-pickle fitted-model artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from jax import Array

from ..model import Model
from ..state_space import StateSpace
from .features import FeatureEncoder
from .structured import StructuredModelConfig
from .topology import TopologySpec

__all__ = ["FittedArtifact"]

_FORMAT_NAME = "jact.fitting"
_FORMAT_VERSION = 1
Link = Literal["exp", "softplus"]


@dataclass(frozen=True)
class FittedArtifact:
    """Portable parameters, preprocessing, topology, and reconstruction data."""

    topology: TopologySpec
    encoder: FeatureEncoder
    model_config: StructuredModelConfig
    params: Mapping[str, Array]
    time_unit: str
    link: Link = "exp"
    log_hazard_bounds: tuple[float, float] | None = (-30.0, 30.0)
    feature_dtypes: Mapping[str, str] = field(default_factory=dict)
    feature_units: Mapping[str, str] = field(default_factory=dict)
    supported_ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    fitting_metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> FittedArtifact:
        """Validate declarative metadata and the structured parameter tree."""
        if not self.time_unit.strip():
            raise ValueError("time_unit must be a non-empty string.")
        if self.encoder.n_features != self.model_config.n_features:
            raise ValueError(
                "FeatureEncoder output width does not match model_config.n_features."
            )
        if self.link not in ("exp", "softplus"):
            raise ValueError("link must be 'exp' or 'softplus'.")
        if self.log_hazard_bounds is not None:
            lower, upper = self.log_hazard_bounds
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError("log_hazard_bounds must be finite and increasing.")
        declared_features = {
            feature.name
            for feature in (*self.encoder.numeric, *self.encoder.categorical)
        }
        for label, values in (
            ("feature_dtypes", self.feature_dtypes),
            ("feature_units", self.feature_units),
        ):
            unknown = set(values) - declared_features
            if unknown:
                raise ValueError(f"{label} contains unknown features: {unknown}.")
            if any(not str(value).strip() for value in values.values()):
                raise ValueError(f"{label} values must be non-empty strings.")
        for name, limits in self.supported_ranges.items():
            if len(limits) != 2 or limits[0] > limits[1]:
                raise ValueError(f"Supported range for '{name}' is invalid.")
        expected = _expected_parameter_shapes(self.topology, self.model_config)
        if set(self.params) != set(expected):
            raise ValueError(
                f"Parameter names must be {tuple(expected)}; got {tuple(self.params)}."
            )
        for name, shape in expected.items():
            value = np.asarray(self.params[name])
            if value.shape != shape:
                raise ValueError(
                    f"Parameter '{name}' has shape {value.shape}; expected {shape}."
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Parameter '{name}' contains non-finite values.")
        try:
            json.dumps(self.fitting_metadata)
        except (TypeError, ValueError) as exc:
            raise TypeError("fitting_metadata must be JSON serializable.") from exc
        return self

    def link_scores(self, scores: Array) -> Array:
        """Convert finite log-scores/raw scores to non-negative intensities."""
        values = jnp.asarray(scores)
        if self.log_hazard_bounds is not None:
            values = jnp.clip(values, *self.log_hazard_bounds)
        if self.link == "exp":
            return jnp.exp(values)
        return jax_softplus(values)

    def adapter(self) -> Any:
        from .adapter import JACTAdapter

        self.validate()
        return JACTAdapter(self)

    def build_model(
        self, state_space: StateSpace, *, assignment: str = "exits"
    ) -> Model:
        """Validate topology and reconstruct a complete JACT model."""
        return self.adapter().build_model(state_space, assignment=assignment)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically save metadata and arrays in one compressed NPZ file."""
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        parameter_names = tuple(sorted(self.params))
        metadata = {
            "format": _FORMAT_NAME,
            "version": _FORMAT_VERSION,
            "topology": self.topology.to_metadata(),
            "encoder": self.encoder.to_metadata(),
            "model_config": self.model_config.to_metadata(),
            "time_unit": self.time_unit,
            "link": self.link,
            "log_hazard_bounds": self.log_hazard_bounds,
            "feature_dtypes": dict(self.feature_dtypes),
            "feature_units": dict(self.feature_units),
            "supported_ranges": {
                key: list(value) for key, value in self.supported_ranges.items()
            },
            "fitting_metadata": self.fitting_metadata,
            "parameter_names": parameter_names,
        }
        encoded = np.frombuffer(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            ),
            dtype=np.uint8,
        )
        arrays: dict[str, np.ndarray] = {"metadata": encoded}
        for index, name in enumerate(parameter_names):
            arrays[f"parameter_{index}"] = np.asarray(self.params[name])

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(file_descriptor, "wb") as file:
                np.savez_compressed(file, **arrays)
            os.replace(temporary_name, target)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> FittedArtifact:
        """Load and fully validate an artifact without enabling pickle."""
        with np.load(path, allow_pickle=False) as archive:
            if "metadata" not in archive:
                raise ValueError("Artifact has no metadata record.")
            try:
                metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Artifact metadata is malformed.") from exc
            if metadata.get("format") != _FORMAT_NAME:
                raise ValueError("File is not a JACT fitting artifact.")
            if metadata.get("version") != _FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported artifact version {metadata.get('version')!r}; "
                    f"expected {_FORMAT_VERSION}."
                )
            names = tuple(str(name) for name in metadata["parameter_names"])
            params: dict[str, Array] = {}
            for index, name in enumerate(names):
                key = f"parameter_{index}"
                if key not in archive:
                    raise ValueError(f"Artifact is missing parameter '{name}'.")
                params[name] = jnp.asarray(archive[key])

        bounds_value = metadata.get("log_hazard_bounds")
        bounds = (
            None
            if bounds_value is None
            else (float(bounds_value[0]), float(bounds_value[1]))
        )
        artifact = cls(
            topology=TopologySpec.from_metadata(metadata["topology"]),
            encoder=FeatureEncoder.from_metadata(metadata["encoder"]),
            model_config=StructuredModelConfig.from_metadata(metadata["model_config"]),
            params=params,
            time_unit=str(metadata["time_unit"]),
            link=str(metadata.get("link", "exp")),  # type: ignore[arg-type]
            log_hazard_bounds=bounds,
            feature_dtypes={
                str(key): str(value)
                for key, value in metadata.get("feature_dtypes", {}).items()
            },
            feature_units={
                str(key): str(value)
                for key, value in metadata.get("feature_units", {}).items()
            },
            supported_ranges={
                str(key): (float(value[0]), float(value[1]))
                for key, value in metadata.get("supported_ranges", {}).items()
            },
            fitting_metadata=metadata.get("fitting_metadata", {}),
        )
        return artifact.validate()


def jax_softplus(values: Array) -> Array:
    # Stable and available in every supported JAX release.
    return jnp.logaddexp(values, jnp.asarray(0.0, dtype=values.dtype))


def _expected_parameter_shapes(
    topology: TopologySpec, config: StructuredModelConfig
) -> dict[str, tuple[int, ...]]:
    input_size = 2 + config.n_features + config.embedding_size
    return {
        "intercept": (topology.n_edges,),
        "linear": (topology.n_edges, config.n_features),
        "time_spline": (topology.n_edges, len(config.time_knots)),
        "duration_spline": (topology.n_edges, len(config.duration_knots)),
        "source_embedding": (topology.n_states, config.embedding_size),
        "hidden_kernel": (input_size, config.hidden_size),
        "hidden_bias": (config.hidden_size,),
        "projection": (config.hidden_size, config.rank),
        "edge_loading": (topology.n_edges, config.rank),
    }
