from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jact


@pytest.fixture
def state_space():
    return jact.StateSpace(
        states=["healthy", "disabled", "dead"],
        transitions=[
            ("disabled", "dead"),
            ("healthy", "dead"),
            ("healthy", "disabled"),
        ],
    )


def _batch(covariates=None):
    if covariates is None:
        covariates = jnp.zeros((3, 0))
    return jact.fitting.CanonicalBatch.from_arrays(
        source_state=[0, 0, 1],
        event_edge=[0, -1, 2],
        t0=[0.0, 0.0, 1.0],
        t1=[1.0, 2.0, 2.0],
        d0=[0.0, 0.0, 0.0],
        d1=[1.0, 2.0, 1.0],
        covariates=covariates,
    )


def test_topology_uses_state_space_edge_order_and_fingerprint(state_space):
    topology = jact.fitting.TopologySpec.from_state_space(state_space)

    assert topology.states == ("healthy", "disabled", "dead")
    assert topology.edges == (
        ("healthy", "disabled"),
        ("healthy", "dead"),
        ("disabled", "dead"),
    )
    assert topology.edge_sources == (0, 0, 1)
    assert topology.outgoing_edges == ((0, 1), (2,), ())
    assert topology.outgoing_edge_array.shape == (3, 2)
    assert topology.outgoing_mask.tolist() == [
        [True, True],
        [True, False],
        [False, False],
    ]
    assert jact.fitting.TopologySpec.from_metadata(
        topology.to_metadata()
    ) == topology


def test_topology_rejects_different_state_order(state_space):
    topology = jact.fitting.TopologySpec.from_state_space(state_space)
    reordered = jact.StateSpace(
        ["disabled", "healthy", "dead"], state_space.transitions
    )

    with pytest.raises(ValueError, match="does not match"):
        topology.validate_state_space(reordered)


def test_canonical_batch_validates_event_origin_and_duration(state_space):
    topology = jact.fitting.TopologySpec.from_state_space(state_space)
    assert _batch().validate(topology).n_rows == 3

    with pytest.raises(ValueError, match="does not exit"):
        _batch().replace(event_edge=jnp.array([2, -1, 2])).validate(topology)
    with pytest.raises(ValueError, match="duration exposure"):
        _batch().replace(d1=jnp.array([0.5, 2.0, 1.0])).validate(topology)


def test_piecewise_likelihood_is_competing_risks_likelihood(state_space):
    topology = jact.fitting.TopologySpec.from_state_space(state_space)
    batch = _batch().validate(topology)
    log_rates = jnp.log(jnp.array([0.1, 0.2, 0.3]))

    def constant(params, source, t, d, covariates):
        del source, t, d
        return jnp.broadcast_to(params["rates"], (*covariates.shape[:-1], 3))

    result = jact.fitting.piecewise_exponential_nll(
        constant, {"rates": log_rates}, batch, topology, reduction="none"
    )

    expected_log_likelihood = jnp.array(
        [jnp.log(0.1) - 0.3, -0.6, jnp.log(0.3) - 0.3]
    )
    assert jnp.allclose(result.row_log_likelihood, expected_log_likelihood)
    assert jnp.allclose(result.loss, -expected_log_likelihood)


def test_likelihood_weights_and_padding_are_exactly_zero(state_space):
    topology = jact.fitting.TopologySpec.from_state_space(state_space)
    batch = _batch().replace(
        weight=jnp.array([2.0, 10.0, 3.0]),
        valid=jnp.array([True, False, True]),
    )

    def constant(params, source, t, d, covariates):
        del params, source, t, d
        return jnp.zeros((*covariates.shape[:-1], 3))

    result = jact.fitting.piecewise_exponential_nll(
        constant, {}, batch, topology, reduction="none"
    )
    assert result.effective_weight.tolist() == [2.0, 0.0, 3.0]
    assert result.loss[1] == 0.0


def test_padded_rows_may_contain_sentinel_values(state_space):
    topology = jact.fitting.TopologySpec.from_state_space(state_space)
    batch = _batch().replace(
        source_state=jnp.array([0, -999, 1]),
        event_edge=jnp.array([0, 999, 2]),
        t0=jnp.array([0.0, jnp.nan, 1.0]),
        t1=jnp.array([1.0, jnp.nan, 2.0]),
        d0=jnp.array([0.0, jnp.nan, 0.0]),
        d1=jnp.array([1.0, jnp.nan, 1.0]),
        covariates=jnp.empty((3, 0)),
        weight=jnp.array([1.0, jnp.nan, 1.0]),
        valid=jnp.array([True, False, True]),
    ).validate(topology)

    def constant(params, source, t, d, covariates):
        del params, source, t, d
        return jnp.zeros((*covariates.shape[:-1], 3))

    result = jact.fitting.piecewise_exponential_nll(
        constant, {}, batch, topology, reduction="none"
    )
    assert jnp.isfinite(result.loss).all()
    assert result.loss[1] == 0.0
    assert jnp.isfinite(jact.fitting.empirical_intercepts(batch, topology)).all()


def test_fixed_quadrature_integrates_changing_hazard(state_space):
    topology = jact.fitting.TopologySpec.from_state_space(state_space)
    batch = jact.fitting.CanonicalBatch.from_arrays(
        source_state=[0],
        event_edge=[-1],
        t0=[0.0],
        t1=[1.0],
        d0=[0.0],
        d1=[1.0],
        covariates=jnp.zeros((1, 0)),
    )

    def clock_hazard(params, source, t, d, covariates):
        del params, source, d, covariates
        return jnp.stack([t, jnp.full_like(t, -100.0), jnp.zeros_like(t)], axis=-1)

    result = jact.fitting.fixed_quadrature_nll(
        clock_hazard, {}, batch, topology, order=8
    )
    assert jnp.allclose(result.cumulative_hazard, jnp.e - 1.0, rtol=1e-6)


def test_feature_encoder_round_trip_and_grid_unknown_category():
    encoder = jact.fitting.FeatureEncoder.fit(
        numeric={"age": [20.0, 30.0, 40.0]},
        categorical={"smoker": ["no", "yes", "no"]},
    )
    codes = encoder.encode_category("smoker", ["yes", "other"])
    rows = encoder.transform_columns({"age": [30.0, 40.0], "smoker": codes})
    grid = encoder.grid(
        0.0,
        jnp.array([[0.5, 1.5]]),
        {"age": jnp.array([30.0, 40.0]), "smoker": codes},
    )

    assert encoder.n_features == 4
    assert rows.shape == (2, 4)
    assert grid.shape == (2, 2, 4)
    assert jnp.allclose(grid[:, 0], rows)
    assert jact.fitting.FeatureEncoder.from_metadata(
        encoder.to_metadata()
    ) == encoder


def test_structured_model_is_jittable_and_differentiable(state_space):
    topology = jact.fitting.TopologySpec.from_state_space(state_space)
    config = jact.fitting.StructuredModelConfig(
        n_features=2,
        time_knots=(0.0, 1.0, 2.0),
        duration_knots=(0.0, 2.0),
        embedding_size=2,
        hidden_size=4,
        rank=2,
    )
    model = jact.fitting.StructuredLogHazardModel(topology, config)
    params = model.init(jax.random.key(5))
    source = jnp.array([0, 1])
    values = jax.jit(model)(
        params,
        source,
        jnp.array([0.5, 1.5]),
        jnp.array([0.25, 1.0]),
        jnp.ones((2, 2)),
    )
    gradient = jax.grad(
        lambda intercept: jnp.sum(
            model(
                {**params, "intercept": intercept},
                source,
                jnp.array([0.5, 1.5]),
                jnp.array([0.25, 1.0]),
                jnp.ones((2, 2)),
            )
        )
    )(params["intercept"])

    assert values.shape == (2, 3)
    assert jnp.all(jnp.isfinite(values))
    assert jnp.allclose(gradient, 2.0)


def test_empirical_intercepts_use_source_exposure(state_space):
    topology = jact.fitting.TopologySpec.from_state_space(state_space)
    intercepts = jact.fitting.empirical_intercepts(
        _batch(), topology, event_prior=1.0, exposure_prior=1.0
    )

    # Healthy exposure is 3, disabled exposure is 1; each observed edge has one event.
    assert jnp.allclose(jnp.exp(intercepts), jnp.array([0.5, 0.25, 1.0]))


def _artifact(state_space):
    topology = jact.fitting.TopologySpec.from_state_space(state_space)
    encoder = jact.fitting.FeatureEncoder.fit(numeric={"age": [30.0, 50.0]})
    config = jact.fitting.StructuredModelConfig(
        n_features=1, embedding_size=0, hidden_size=0, rank=0
    )
    structured = jact.fitting.StructuredLogHazardModel(topology, config)
    params = structured.init(
        jax.random.key(0), intercept=jnp.log(jnp.array([0.1, 0.2, 0.3]))
    )
    return jact.fitting.FittedArtifact(
        topology=topology,
        encoder=encoder,
        model_config=config,
        params=params,
        time_unit="years",
        feature_dtypes={"age": "float32"},
        feature_units={"age": "years"},
        supported_ranges={"age": (20.0, 100.0)},
        fitting_metadata={"run": "test"},
    )


def test_artifact_round_trip_and_jact_exit_model(state_space, tmp_path: Path):
    artifact = _artifact(state_space).validate()
    path = tmp_path / "model.jact.npz"
    artifact.save(path)
    loaded = jact.fitting.FittedArtifact.load(path)
    model = loaded.build_model(state_space, assignment="exits")
    result = model.solve(
        initial="healthy",
        horizon=1,
        steps_per_unit=40,
        age=jnp.array([30.0, 50.0]),
    )

    assert loaded.topology.fingerprint == artifact.topology.fingerprint
    assert loaded.encoder == artifact.encoder
    assert result.probability.shape == (41, 2, 3)
    assert jnp.allclose(
        result.probability[-1, :, 0], jnp.exp(jnp.array([-0.3, -0.3])), atol=2e-3
    )
    loaded.adapter().validate_hazards(
        state_space,
        t=jnp.asarray(0.5),
        d=jnp.array([[0.25, 0.75]]),
        covariates={"age": jnp.array([30.0, 50.0])},
    )


def test_artifact_global_group_matches_exit_assignment(state_space):
    artifact = _artifact(state_space)
    exits = artifact.build_model(state_space, assignment="exits")
    group = artifact.build_model(state_space, assignment="group")

    assert jnp.allclose(
        exits.solve(
            initial="healthy",
            horizon=1,
            steps_per_unit=10,
            age=jnp.array([30.0, 50.0]),
        ).probability,
        group.solve(
            initial="healthy",
            horizon=1,
            steps_per_unit=10,
            age=jnp.array([30.0, 50.0]),
        ).probability,
    )


def test_artifact_simulation_agrees_with_solve(state_space):
    model = _artifact(state_space).build_model(state_space)
    solved = model.solve(
        initial="healthy",
        horizon=1,
        steps_per_unit=40,
        age=jnp.array([40.0]),
    ).probability[-1, 0]
    simulated = model.simulate(
        initial="healthy",
        horizon=1,
        steps_per_unit=40,
        max_jumps=2,
        replicates=10_000,
        key=jax.random.key(91),
        age=jnp.array([40.0]),
    )
    frequencies = jnp.bincount(simulated.final_state[0], length=3) / 10_000

    assert jnp.allclose(frequencies, solved, atol=0.02)


def test_adamw_trainer_reduces_loss():
    batch = jact.fitting.CanonicalBatch.from_arrays(
        source_state=[0, 0],
        event_edge=[-1, -1],
        t0=[0.0, 0.0],
        t1=[1.0, 1.0],
        d0=[0.0, 0.0],
        d1=[1.0, 1.0],
        covariates=jnp.zeros((2, 0)),
    )

    def loss(params, unused_batch):
        del unused_batch
        return (params["value"] - 2.0) ** 2

    trainer = jact.fitting.Trainer(
        loss,
        jact.fitting.AdamWConfig(
            learning_rate=0.1, weight_decay=0.0, clip_norm=10.0
        ),
    )
    result = trainer.fit({"value": jnp.asarray(0.0)}, [batch], epochs=30)

    assert result.history.training_loss[-1] < result.history.training_loss[0]
    assert abs(float(result.params["value"]) - 2.0) < 0.3


def test_artifact_rejects_nonfinite_parameters(state_space):
    artifact = _artifact(state_space)
    bad_params = dict(artifact.params)
    bad_params["intercept"] = jnp.array([jnp.nan, 0.0, 0.0])

    with pytest.raises(ValueError, match="non-finite"):
        jact.fitting.FittedArtifact(
            **{**artifact.__dict__, "params": bad_params}
        ).validate()


def test_artifact_file_is_not_pickle(state_space, tmp_path: Path):
    path = tmp_path / "portable.npz"
    _artifact(state_space).save(path)

    with np.load(path, allow_pickle=False) as archive:
        assert archive["metadata"].dtype == np.uint8
        assert all(archive[name].dtype != object for name in archive.files)
