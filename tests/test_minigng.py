"""Minimal test suite for MiniGNG.

Each test doubles as a regression test for one of the bugs fixed in the
review (input mutation, fit_predict signature, predict index alignment,
score, sampling, seeding) plus a few basic-behavior checks.
"""

import numpy as np
import pytest

from minigng import MiniGNG


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = rng.rand(200, 4).astype(np.float32)
    y = rng.randint(0, 3, 200)
    return X, y


def fitted(X, y=None, **kw):
    """Fit a small model with a fixed seed for reproducibility."""
    np.random.seed(0)
    params = {"max_units": 20, "n_epochs": 5}
    params.update(kw)
    return MiniGNG(**params).fit(X, y)


# --- #1: fit must not mutate the caller's input ----------------------------

def test_fit_does_not_mutate_input(data):
    X, _ = data
    X0 = X.copy()
    fitted(X)
    assert np.array_equal(X, X0)


def test_fit_preserves_float_dtype(data):
    X, _ = data
    assert fitted(X).units[0].prototype.dtype == np.float32


def test_int_input_not_mutated_and_cast_to_float():
    Xi = np.random.RandomState(1).randint(0, 10, (200, 4))
    Xi0 = Xi.copy()
    g = fitted(Xi)
    assert np.array_equal(Xi, Xi0)
    assert np.issubdtype(g.units[0].prototype.dtype, np.floating)


# --- #2: fit_predict -------------------------------------------------------

def test_fit_predict_clustering(data):
    X, _ = data
    uids, labels = MiniGNG(max_units=20, n_epochs=5).fit_predict(X)
    assert len(uids) == len(X)
    assert labels is None


def test_fit_predict_matches_fit_then_predict(data):
    X, y = data
    np.random.seed(0)
    u1, l1 = MiniGNG(max_units=20, n_epochs=5).fit_predict(X, y)
    np.random.seed(0)
    u2, l2 = MiniGNG(max_units=20, n_epochs=5).fit(X, y).predict(X)
    assert u1 == u2
    assert np.array_equal(l1, l2)


def test_fit_resets_existing_model_state(data):
    X, _ = data
    g = MiniGNG(max_units=20, n_epochs=2)
    np.random.seed(0)
    g.fit(X)
    first_signal_counter = g.signal_counter
    g.signal_counter = 999
    np.random.seed(1)
    g.fit(X[:100])
    assert first_signal_counter == g.n_epochs * len(X)
    assert g.signal_counter == g.n_epochs * 100
    assert 0 < len(g.units) <= g.max_units


# --- #4: predict unit ids align with self.units ----------------------------

def test_predict_unit_ids_align_with_dead_unit(data):
    X, y = data
    g = fitted(X, y)
    g.units[0].count = 0  # force a "dead" unit at a low index
    uids, _ = g.predict(X)
    live = [(i, u) for i, u in enumerate(g.units) if u.count > 0]
    protos = np.array([u.prototype for _, u in live])
    for j, x in enumerate(X):
        nearest = live[int(np.argmin(np.linalg.norm(x - protos, axis=1)))][0]
        assert uids[j] == nearest


def test_predict_classification_labels_are_known_classes(data):
    X, y = data
    uids, labels = fitted(X, y).predict(X)
    assert len(uids) == len(labels) == len(X)
    assert set(labels) <= set(np.unique(y).tolist())


def test_predict_empty_model_returns_tuple():
    uids, labels = MiniGNG().predict(np.zeros((3, 4)))
    assert uids == []
    assert labels is None


# --- #3: score -------------------------------------------------------------

def test_score_in_unit_range(data):
    X, y = data
    assert 0.0 <= fitted(X, y).score(X, y) <= 1.0


def test_score_string_labels():
    X = np.random.RandomState(0).rand(150, 4).astype(np.float32)
    y = np.array(["a", "b", "c"])[np.random.RandomState(1).randint(0, 3, 150)]
    assert 0.0 <= fitted(X, y).score(X, y) <= 1.0


def test_score_requires_classification(data):
    X, _ = data
    g = fitted(X)  # no y -> clustering
    with pytest.raises(ValueError):
        g.score(X, np.zeros(len(X)))


def test_score_requires_matching_target_length(data):
    X, y = data
    with pytest.raises(ValueError, match='same number of samples'):
        fitted(X, y).score(X, y[:-1])


# --- #5 / init / seeding ---------------------------------------------------

def test_init_model_seeds_two_distinct_units(data):
    X, _ = data
    np.random.seed(0)
    g = MiniGNG()
    g.init_model(X)
    assert len(g.units) == 2
    assert not np.array_equal(g.units[0].prototype, g.units[1].prototype)


def test_init_model_requires_two_samples():
    with pytest.raises(ValueError, match='at least 2 samples'):
        MiniGNG().init_model(np.zeros((1, 4)))


def test_reject_invalid_parameters():
    with pytest.raises(ValueError, match='sample must be in the interval'):
        MiniGNG(sample=0)
    with pytest.raises(ValueError, match='max_units must be an integer >= 2'):
        MiniGNG(max_units=1)


def test_fit_is_reproducible_with_seed(data):
    X, _ = data

    def run():
        np.random.seed(42)
        g = MiniGNG(max_units=20, n_epochs=5).fit(X)
        return np.array([u.prototype for u in g.units])

    a, b = run(), run()
    assert a.shape == b.shape
    assert np.allclose(a, b)


# --- basic behavior --------------------------------------------------------

def test_fit_respects_max_units(data):
    X, _ = data
    g = fitted(X, max_units=15)
    assert 0 < len(g.units) <= 15


def test_partial_fit_online_batches(data):
    X, _ = data
    np.random.seed(0)
    g = MiniGNG(max_units=40)
    for batch in np.array_split(X, 10):
        assert g.partial_fit(batch) is g
    assert len(g.units) >= 2


def test_reject_inconsistent_feature_counts(data):
    X, y = data
    g = fitted(X, y)
    with pytest.raises(ValueError, match='Expected 4 features, got 3'):
        g.predict(np.zeros((3, 3)))
    with pytest.raises(ValueError, match='Expected 4 features, got 3'):
        g.partial_fit(np.zeros((10, 3)))


def test_fit_requires_matching_target_length(data):
    X, y = data
    with pytest.raises(ValueError, match='same number of samples'):
        MiniGNG().fit(X, y[:-1])


def test_export_quotes_string_labels(tmp_path):
    X = np.random.RandomState(0).rand(150, 4).astype(np.float32)
    y = np.array(['class "A"', 'class B', 'class C'])[
        np.random.RandomState(1).randint(0, 3, 150)
    ]
    g = fitted(X, y)
    gml_path = tmp_path / 'graph.gml'
    vna_path = tmp_path / 'graph.vna'

    g.save_gml(gml_path)
    g.save_vna(vna_path)

    gml = gml_path.read_text(encoding='utf-8')
    vna = vna_path.read_text(encoding='utf-8')
    assert 'label "class' in gml
    assert '\\"A\\"' in gml
    assert '"class B"' in vna


def test_edges_reference_existing_units(data):
    X, _ = data
    g = fitted(X)
    for e in g.edges:
        assert e.source in g.units
        assert e.target in g.units
