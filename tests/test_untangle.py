"""Tests for the untangling mechanism (untangle=True).

Covers the graph helpers it relies on — network_size_compare, exists_path,
and no_curling — which were previously untested, plus an end-to-end fit with
untangle enabled.

The helpers operate on plain Unit graphs, so the tests build small graphs by
hand and call the methods on a bare MiniGNG instance.
"""

import numpy as np
import pytest

from minigng import MiniGNG
from minigng._growing_neural_gas import Unit


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    return rng.rand(200, 4).astype(np.float32)


def unit(*coords):
    return Unit(np.array(coords, dtype=float))


def connect(a, b):
    """Add a symmetric edge between two units (as the model does internally)."""
    a.neighbors.add(b)
    b.neighbors.add(a)


# --- network_size_compare --------------------------------------------------

def test_network_size_compare_isolated_node():
    g = MiniGNG()
    a = unit(0.0)
    assert g.network_size_compare(a, 1) == 0   # network is just itself
    assert g.network_size_compare(a, 3) == -1  # smaller than 3


def test_network_size_compare_chain():
    g = MiniGNG()
    a, b, c = unit(0.0), unit(1.0), unit(2.0)
    connect(a, b)
    connect(b, c)
    assert g.network_size_compare(a, 3) == 0   # exactly 3 nodes
    assert g.network_size_compare(a, 2) == 1   # network larger than 2
    assert g.network_size_compare(a, 5) == -1  # network smaller than 5


# --- exists_path -----------------------------------------------------------

def test_exists_path_connected_chain():
    g = MiniGNG()
    a, b, c = unit(0.0), unit(1.0), unit(2.0)
    connect(a, b)
    connect(b, c)
    assert g.exists_path(a, b) is True       # direct neighbor
    assert g.exists_path(a, c) is True        # two hops away


def test_exists_path_disconnected():
    g = MiniGNG()
    a, b, c, d = unit(0.0), unit(1.0), unit(2.0), unit(3.0)
    connect(a, b)
    connect(c, d)
    assert g.exists_path(a, c) is False
    assert g.exists_path(a, d) is False


# --- no_curling: n_bridges == 0 --------------------------------------------

def test_no_curling_no_bridges_disconnected_small():
    # No common neighbors, no path, both networks small -> connecting is fine.
    g = MiniGNG(max_size_connect=3)
    a, b = unit(0.0), unit(1.0)
    assert g.no_curling(a, b) is True


def test_no_curling_no_bridges_but_path_exists():
    # a and b are already connected through a path (no shared neighbor);
    # connecting them would close a loop, so curling is rejected.
    # max_size_connect=0 skips the size check, isolating the path test.
    g = MiniGNG(max_size_connect=0)
    a, x, y, b = unit(0.0), unit(1.0), unit(2.0), unit(3.0)
    connect(a, x)
    connect(x, y)
    connect(y, b)
    assert g.no_curling(a, b) is False


def test_no_curling_no_bridges_target_network_too_large():
    # b belongs to a network larger than max_size_connect -> denied.
    g = MiniGNG(max_size_connect=2)
    a = unit(0.0)
    b, c, d = unit(1.0), unit(2.0), unit(3.0)
    connect(b, c)
    connect(c, d)  # b's network has 3 nodes > 2
    assert g.no_curling(a, b) is False


# --- no_curling: n_bridges == 1 --------------------------------------------

def test_no_curling_one_bridge_allows():
    g = MiniGNG()
    a, b, x = unit(0.0), unit(1.0), unit(2.0)
    connect(a, x)
    connect(b, x)  # x is the single shared neighbor
    assert g.no_curling(a, b) is True


def test_no_curling_one_bridge_hub_too_connected():
    # The shared neighbor x has more than 6 neighbors -> rejected.
    g = MiniGNG()
    a, b, x = unit(0.0), unit(1.0), unit(2.0)
    connect(a, x)
    connect(b, x)
    for k in range(5):
        connect(x, unit(10.0 + k))  # x now has 7 neighbors
    assert g.no_curling(a, b) is False


# --- no_curling: n_bridges == 2 and > 2 ------------------------------------

def test_no_curling_two_bridges_is_false():
    # With symmetric edges, two shared neighbors x, y both connect back to
    # a and b, so x.neighbors & y.neighbors always contains {a, b} and the
    # branch returns False. (Documents that the "not connected" sub-case of
    # the n_bridges == 2 branch is effectively unreachable for real graphs.)
    g = MiniGNG()
    a, b, x, y = unit(0.0), unit(1.0), unit(2.0), unit(3.0)
    connect(a, x)
    connect(a, y)
    connect(b, x)
    connect(b, y)
    assert g.no_curling(a, b) is False


def test_no_curling_three_bridges_is_false():
    g = MiniGNG()
    a, b = unit(0.0), unit(1.0)
    for k in range(3):
        x = unit(2.0 + k)
        connect(a, x)
        connect(b, x)
    assert g.no_curling(a, b) is False


# --- end-to-end ------------------------------------------------------------

def test_fit_with_untangle_produces_valid_model(data):
    np.random.seed(0)
    g = MiniGNG(max_units=20, n_epochs=5, untangle=True).fit(data)
    assert len(g.units) >= 2
    # Edges reference live units and neighbor sets remain symmetric.
    for e in g.edges:
        assert e.source in g.units
        assert e.target in g.units
        assert e.target in e.source.neighbors
        assert e.source in e.target.neighbors


def test_fit_untangle_with_max_size_connect_zero(data):
    # max_size_connect=0 takes the "skip size check" branch in no_curling.
    np.random.seed(0)
    g = MiniGNG(max_units=20, n_epochs=5, untangle=True, max_size_connect=0).fit(data)
    assert len(g.units) >= 2
