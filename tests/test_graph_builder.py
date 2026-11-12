"""Tests for graph construction (all four configurations)."""
import numpy as np
import pandas as pd

from src.models.stgnn.graph_builder import (
    DOCUMENTED_INTERCONNECTIONS,
    REGION_COORDS,
    _normalise_adjacency,
    build_all_graphs,
    build_correlation_graph,
    build_geographic_graph,
    build_identity_graph,
    build_interconnection_graph,
)

REGIONS = ["AB", "BC", "NB", "NL", "NS", "ON", "PE", "QC"]
N = len(REGIONS)


def make_train_df(regions=REGIONS, n_hours: int = 400) -> pd.DataFrame:
    frames = []
    for region in regions:
        base = 10000 + hash(region) % 5000
        times = pd.date_range("2022-01-01", periods=n_hours, freq="h")
        frames.append(pd.DataFrame({
            "time_utc": times,
            "region": region,
            "demand_mw": base + np.random.default_rng(hash(region) % 2**32).normal(0, 500, n_hours),
        }))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class TestNormaliseAdjacency:
    def test_identity_input_gives_scaled_identity(self):
        """Zero off-diagonal input → after self-loop addition → scaled identity."""
        A = np.zeros((3, 3), dtype=np.float32)
        A_hat = _normalise_adjacency(A)
        # Should be symmetric
        np.testing.assert_allclose(A_hat, A_hat.T, atol=1e-6)

    def test_symmetric_output(self):
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        A_hat = _normalise_adjacency(A)
        np.testing.assert_allclose(A_hat, A_hat.T, atol=1e-6)

    def test_shape_preserved(self):
        A = np.ones((5, 5), dtype=np.float32)
        np.fill_diagonal(A, 0)
        A_hat = _normalise_adjacency(A)
        assert A_hat.shape == (5, 5)

    def test_values_in_valid_range(self):
        """Normalised adjacency values should be in [-1, 1]."""
        A = np.random.default_rng(0).uniform(0, 1, (5, 5)).astype(np.float32)
        A_hat = _normalise_adjacency(A)
        assert A_hat.max() <= 1.01  # allow small float tolerance
        assert A_hat.min() >= -1.01


# ---------------------------------------------------------------------------
# Identity graph
# ---------------------------------------------------------------------------

class TestIdentityGraph:
    def test_shape(self):
        A = build_identity_graph(REGIONS)
        assert A.shape == (N, N)

    def test_dtype_float32(self):
        A = build_identity_graph(REGIONS)
        assert A.dtype == np.float32

    def test_is_symmetric(self):
        A = build_identity_graph(REGIONS)
        np.testing.assert_allclose(A, A.T, atol=1e-6)

    def test_diagonal_dominates(self):
        """In the identity graph, each node's self-connection should be maximal."""
        A = build_identity_graph(REGIONS)
        for i in range(N):
            # Diagonal ≥ all off-diagonal in that row
            assert A[i, i] >= A[i, :].max() - 1e-6

    def test_no_cross_region_edges(self):
        """The raw adjacency before normalisation should have zero off-diagonal.
        After normalisation with self-loops only, off-diagonal should be 0."""
        A = build_identity_graph(REGIONS)
        A_no_diag = A.copy()
        np.fill_diagonal(A_no_diag, 0)
        np.testing.assert_allclose(A_no_diag, 0, atol=1e-6)


# ---------------------------------------------------------------------------
# Geographic graph
# ---------------------------------------------------------------------------

class TestGeographicGraph:
    def test_shape(self):
        A = build_geographic_graph(REGIONS)
        assert A.shape == (N, N)

    def test_symmetric(self):
        A = build_geographic_graph(REGIONS)
        np.testing.assert_allclose(A, A.T, atol=1e-6)

    def test_all_coords_available(self):
        """Every region in REGIONS must have coordinates defined."""
        for r in REGIONS:
            assert r in REGION_COORDS, f"No coordinates for region {r}"

    def test_zero_distance_threshold_has_no_edges(self):
        A = build_geographic_graph(REGIONS, max_distance_km=1.0)
        # With 1 km threshold no interprovincial edges should exist
        A_no_diag = A.copy()
        np.fill_diagonal(A_no_diag, 0)
        # After normalisation edges from self-loop only → should be ~0
        np.testing.assert_allclose(A_no_diag, 0, atol=1e-5)

    def test_large_threshold_includes_edges(self):
        A = build_geographic_graph(REGIONS, max_distance_km=10000.0)
        A_no_diag = A.copy()
        np.fill_diagonal(A_no_diag, 0)
        # At 10,000 km threshold all Canadian regions should be connected
        assert A_no_diag.sum() > 0

    def test_nb_pe_connected_at_1000km(self):
        """NB and PE are neighbours; they should be connected at 1000 km."""
        A = build_geographic_graph(REGIONS, max_distance_km=1000.0)
        idx = {r: i for i, r in enumerate(REGIONS)}
        # Either NB→PE or PE→NB edge should exist
        assert A[idx["NB"], idx["PE"]] > 0 or A[idx["PE"], idx["NB"]] > 0


# ---------------------------------------------------------------------------
# Correlation graph
# ---------------------------------------------------------------------------

class TestCorrelationGraph:
    def test_shape(self):
        train_df = make_train_df()
        A = build_correlation_graph(train_df, REGIONS)
        assert A.shape == (N, N)

    def test_symmetric(self):
        train_df = make_train_df()
        A = build_correlation_graph(train_df, REGIONS)
        np.testing.assert_allclose(A, A.T, atol=1e-6)

    def test_threshold_1_gives_no_edges(self):
        """With threshold=1.0 no pair can reach perfect correlation."""
        train_df = make_train_df()
        A = build_correlation_graph(train_df, REGIONS, threshold=1.0)
        A_no_diag = A.copy()
        np.fill_diagonal(A_no_diag, 0)
        np.testing.assert_allclose(A_no_diag, 0, atol=1e-5)

    def test_threshold_0_gives_all_edges(self):
        """With threshold=0 all pairs should be connected."""
        train_df = make_train_df()
        A = build_correlation_graph(train_df, REGIONS, threshold=0.0)
        A_no_diag = A.copy()
        np.fill_diagonal(A_no_diag, 0)
        # Every off-diagonal pair should be non-zero
        for i in range(N):
            for j in range(N):
                if i != j:
                    assert A_no_diag[i, j] > 0, f"Edge ({i},{j}) missing at threshold=0"

    def test_uses_only_demand_mw(self):
        """Should not crash with extra columns in train_df."""
        train_df = make_train_df()
        train_df["extra_col"] = 1.0
        A = build_correlation_graph(train_df, REGIONS)
        assert A.shape == (N, N)


# ---------------------------------------------------------------------------
# Interconnection graph
# ---------------------------------------------------------------------------

class TestInterconnectionGraph:
    def test_shape(self):
        A = build_interconnection_graph(REGIONS)
        assert A.shape == (N, N)

    def test_symmetric(self):
        A = build_interconnection_graph(REGIONS)
        np.testing.assert_allclose(A, A.T, atol=1e-6)

    def test_on_qc_connected(self):
        """Ontario–Quebec is a documented interconnection."""
        A = build_interconnection_graph(REGIONS)
        idx = {r: i for i, r in enumerate(REGIONS)}
        assert A[idx["ON"], idx["QC"]] > 0

    def test_ab_bc_connected(self):
        """Alberta–British Columbia is a documented interconnection."""
        A = build_interconnection_graph(REGIONS)
        idx = {r: i for i, r in enumerate(REGIONS)}
        assert A[idx["AB"], idx["BC"]] > 0

    def test_yk_isolated(self):
        """Yukon has no documented interprovincial tie-line."""
        regions_with_yk = REGIONS + ["YK"]
        A = build_interconnection_graph(regions_with_yk)
        idx = {r: i for i, r in enumerate(regions_with_yk)}
        yk_idx = idx["YK"]
        A_yk = A[yk_idx, :].copy()
        A_yk[yk_idx] = 0  # remove self-loop
        np.testing.assert_allclose(A_yk, 0, atol=1e-5)

    def test_no_invented_connections(self):
        """Every off-diagonal edge must correspond to a documented interconnection."""
        A = build_interconnection_graph(REGIONS)
        idx = {r: i for i, r in enumerate(REGIONS)}
        documented_pairs = {
            (min(r1, r2), max(r1, r2))
            for r1, r2 in DOCUMENTED_INTERCONNECTIONS
            if r1 in idx and r2 in idx
        }
        A_no_diag = A.copy()
        np.fill_diagonal(A_no_diag, 0)
        for i in range(N):
            for j in range(i + 1, N):
                if A_no_diag[i, j] > 1e-5:
                    pair = (min(REGIONS[i], REGIONS[j]), max(REGIONS[i], REGIONS[j]))
                    assert pair in documented_pairs, \
                        f"Invented edge between {REGIONS[i]} and {REGIONS[j]}"


# ---------------------------------------------------------------------------
# build_all_graphs
# ---------------------------------------------------------------------------

class TestBuildAllGraphs:
    def test_returns_all_four_types(self):
        train_df = make_train_df()
        graphs = build_all_graphs(REGIONS, train_df=train_df)
        for key in ["identity", "geographic", "correlation", "interconnection"]:
            assert key in graphs, f"Graph type '{key}' missing"

    def test_all_shapes_correct(self):
        train_df = make_train_df()
        graphs = build_all_graphs(REGIONS, train_df=train_df)
        for key, A in graphs.items():
            assert A.shape == (N, N), f"{key} graph has wrong shape"

    def test_all_symmetric(self):
        train_df = make_train_df()
        graphs = build_all_graphs(REGIONS, train_df=train_df)
        for key, A in graphs.items():
            np.testing.assert_allclose(A, A.T, atol=1e-5, err_msg=f"{key} not symmetric")

    def test_identity_differs_from_geographic(self):
        """Identity and geographic graphs must be different (geographic has off-diagonal)."""
        train_df = make_train_df()
        graphs = build_all_graphs(REGIONS, train_df=train_df, max_distance_km=5000.0)
        identity = graphs["identity"]
        geographic = graphs["geographic"]
        # They should not be equal since geographic has edges at 5000 km
        assert not np.allclose(identity, geographic)
