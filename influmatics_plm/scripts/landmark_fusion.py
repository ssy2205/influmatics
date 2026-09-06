"""Standalone continuous feature fusion and out-of-sample metric placement.

Run: python scripts/landmark_fusion.py
Test: python -m pytest scripts/landmark_fusion.py -q
Dependencies: numpy, pandas, torch, scipy; pytest for tests only.

No baseline MDS is fitted here. Coordinates at landmark_indices must be a
previously frozen reference configuration fitted WITHOUT held-out information.
Predicted distances must come from a model trained without held-out targets.
These upstream provenance requirements cannot be verified from arrays alone.
Held-out true coordinates are used only for the final Procrustes scoring.
"""

from numbers import Integral

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.spatial import procrustes
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.utils.data import Dataset


class AttributedPairDataset(Dataset):
    """Return frozen float32 (emb_A, emb_B, scalar delta_g, scalar target).

    Node IDs are looked up exactly, without string/integer coercion. Attributes
    are nonnegative integers. Inputs are snapshotted to prevent later mutation
    of caller-owned dictionaries/tensors from changing dataset values.
    """

    def __init__(self, edge_df, embeddings_dict, attribute_dict):
        required = {"node_A", "node_B", "distance"}
        if (not isinstance(edge_df, pd.DataFrame) or not edge_df.columns.is_unique
                or not required.issubset(edge_df.columns)):
            raise ValueError("edge_df needs unique node_A, node_B, distance columns")
        if edge_df[["node_A", "node_B"]].isna().to_numpy().any():
            raise ValueError("Node IDs must not be missing")
        target = edge_df["distance"].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(target).all() or (target < 0).any():
            raise ValueError("Targets must be finite and nonnegative")
        pairs = list(zip(edge_df.node_A, edge_df.node_B))
        nodes = list(dict.fromkeys(node for pair in pairs for node in pair))
        lookup = {node: i for i, node in enumerate(nodes)}
        self.embeddings = torch.empty((len(nodes), 1280), dtype=torch.float32)
        self.attributes = []
        for i, node in enumerate(nodes):
            if node not in embeddings_dict or node not in attribute_dict:
                raise KeyError(f"Missing embedding or attribute for node {node!r}")
            embedding = embeddings_dict[node]
            if (not isinstance(embedding, torch.Tensor)
                    or embedding.shape != (1280,) or not embedding.is_floating_point()):
                raise ValueError(f"Embedding for {node!r} must be a floating Tensor(1280)")
            embedding = embedding.detach().to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(embedding).all():
                raise ValueError(f"Non-finite embedding for {node!r}")
            self.embeddings[i].copy_(embedding)
            attribute = attribute_dict[node]
            if (isinstance(attribute, (bool, np.bool_))
                    or not isinstance(attribute, Integral) or attribute < 0):
                raise ValueError(f"Attribute for {node!r} must be a nonnegative integer")
            if attribute > np.finfo(np.float32).max:
                raise ValueError("Attribute exceeds float32 range")
            self.attributes.append(int(attribute))
        self.pairs = [(lookup[a], lookup[b]) for a, b in pairs]
        self.targets = torch.from_numpy(target)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        a, b = self.pairs[index]
        # Subtract as Python integers before converting the difference to float.
        delta_g = torch.tensor(abs(self.attributes[a] - self.attributes[b]),
                               dtype=torch.float32)
        return self.embeddings[a], self.embeddings[b], delta_g, self.targets[index]


class AugmentedDistanceHead(nn.Module):
    """Exact requested MLP, returning raw regression scores of shape (B,).

    The final Linear is unconstrained: use F.softplus(scores) in the training
    AND prediction pipeline if nonnegative distances are required. Concatenated
    ordered embeddings are not swap invariant. For undirected predictions,
    average both orientations consistently during training and evaluation.
    """

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5121, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1),
        )

    def forward(self, emb_A, emb_B, delta_g):
        if (emb_A.ndim != 2 or emb_A.shape[1] != 1280
                or emb_A.shape != emb_B.shape):
            raise ValueError("Embeddings must have matching shape (B, 1280)")
        if emb_A.device != emb_B.device or emb_A.dtype != emb_B.dtype:
            raise ValueError("Embeddings must share device and floating dtype")
        if not emb_A.is_floating_point():
            raise TypeError("Embeddings must be floating point")
        if delta_g.shape == (len(emb_A),):
            delta_g = delta_g[:, None]
        if delta_g.shape != (len(emb_A), 1):
            raise ValueError("delta_g must have shape (B,) or (B, 1)")
        if delta_g.is_complex():
            raise ValueError("delta_g must be real")
        delta_g = delta_g.to(device=emb_A.device, dtype=emb_A.dtype)
        if not torch.isfinite(delta_g).all() or (delta_g < 0).any():
            raise ValueError("delta_g must be finite and nonnegative")
        features = torch.cat(
            [emb_A, emb_B, torch.abs(emb_A - emb_B), emb_A * emb_B, delta_g], dim=-1
        )
        return self.mlp(features).squeeze(-1)


def _float_array(value, name):
    value = np.asarray(value)
    if value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric array")
    return np.array(value, dtype=np.float64, copy=True)


def _stress_and_gradient(position, landmarks, distances):
    """Mean squared distance residual; tiny smoothing at coincident points."""
    difference = position - landmarks
    radii = np.sqrt(np.sum(difference ** 2, axis=1) + 1e-24)
    residual = radii - distances
    value = np.mean(residual ** 2)
    gradient = 2 * np.mean((residual / radii)[:, None] * difference, axis=0)
    return float(value), gradient


def multilaterate_landmarks(landmark_coords, distances_to_landmarks):
    """Place H nodes from fixed (K, 2) landmarks and (H, K) distances only.

    Minimize mean_k (||x-L_k|| - d_k)^2 using float64 BFGS with an analytic
    gradient and six deterministic starts. Center/scale from landmarks only.
    At least three well-spread, noncollinear landmarks are required. BFGS is
    local optimization; multiple starts do not guarantee a global optimum.
    Raise if no start converges; do not silently drop failed held-out nodes.
    Returns (positions, per-node diagnostics).
    """
    landmarks = _float_array(landmark_coords, "landmark_coords")
    distances = _float_array(distances_to_landmarks, "distances_to_landmarks")
    if landmarks.ndim != 2 or landmarks.shape[1] != 2 or len(landmarks) < 3:
        raise ValueError("Need landmark_coords of shape (K, 2), K >= 3")
    if distances.ndim != 2 or distances.shape[1] != len(landmarks):
        raise ValueError("distances_to_landmarks must have shape (H, K)")
    if not np.isfinite(landmarks).all() or not np.isfinite(distances).all():
        raise ValueError("Landmarks and required distances must be finite")
    if (distances < 0).any():
        raise ValueError("Required distances must be nonnegative")
    center = landmarks.mean(axis=0)
    centered = landmarks - center
    scale = np.linalg.norm(centered) / np.sqrt(len(landmarks))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Landmarks must have positive finite spread")
    normalized = centered / scale
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    if singular_values[-1] <= 1e-8 * singular_values[0]:
        raise ValueError("Landmarks are collinear or numerically ill-conditioned")
    normalized_distances = distances / scale
    if not np.isfinite(normalized_distances).all():
        raise ValueError("Distances overflow relative to landmark scale")
    positions = np.empty((len(distances), 2), dtype=np.float64)
    diagnostics = []
    linear_a = 2 * (normalized[1:] - normalized[0])
    squared_landmarks = np.sum(normalized ** 2, axis=1)
    for i, target in enumerate(normalized_distances):
        # Squared-distance subtraction supplies a truth-free linear initializer.
        linear_b = (squared_landmarks[1:] - squared_landmarks[0]
                    + target[0] ** 2 - target[1:] ** 2)
        initial = np.linalg.lstsq(linear_a, linear_b, rcond=None)[0]
        starts = [initial, np.zeros(2), np.array([1., 0.]), np.array([-1., 0.]),
                  np.array([0., 1.]), np.array([0., -1.])]
        candidates = []
        for start in starts:
            result = minimize(_stress_and_gradient, start,
                              args=(normalized, target), method="BFGS", jac=True,
                              options={"gtol": 1e-8, "maxiter": 1000})
            gradient_norm = float(np.linalg.norm(result.jac, ord=np.inf))
            # A precision-loss status may still meet this explicit tolerance.
            if (np.isfinite(result.fun) and np.isfinite(result.x).all()
                    and np.isfinite(gradient_norm)
                    and (result.success or gradient_norm <= 1e-6)):
                candidates.append(result)
        if not candidates:
            raise RuntimeError(f"BFGS failed for held-out row {i}; no converged start")
        best = min(candidates, key=lambda result: result.fun)
        positions[i] = best.x * scale + center
        diagnostics.append({
            "normalized_stress": float(best.fun),
            "gradient_inf_norm": float(np.linalg.norm(best.jac, ord=np.inf)),
            "optimizer_success": bool(best.success),
            "accepted_starts": len(candidates), "message": str(best.message),
        })
    return positions, diagnostics


def evaluate_out_of_sample_landmarks(true_coords_all, pred_distance_matrix,
                                     landmark_indices, *, return_details=False):
    """Return five Procrustes/axis metrics, computed on non-landmarks only.

    All arrays share a node ordering. true_coords_all[landmark_indices] contains
    the frozen baseline landmark positions. Only the held-out-to-landmark
    rectangle of pred_distance_matrix is consumed; unused entries may be NaN.
    At least two distinct held-out points are needed (two-point shape scores
    are uninformative; use a substantially larger held-out set in practice).
    Constant/numerically collapsed axes yield NaN correlations.

    return_details=True returns (metrics, details), including raw held-out
    positions for checking independence from held-out reference coordinates.
    """
    true = _float_array(true_coords_all, "true_coords_all")
    if true.ndim != 2 or true.shape[1] != 2 or not np.isfinite(true).all():
        raise ValueError("true_coords_all must be finite with shape (N, 2)")
    distances = _float_array(pred_distance_matrix, "pred_distance_matrix")
    if distances.shape != (len(true), len(true)):
        raise ValueError("pred_distance_matrix must have shape (N, N)")
    indices = np.asarray(landmark_indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("landmark_indices must be a 1D integer array")
    if len(indices) < 3 or len(np.unique(indices)) != len(indices):
        raise ValueError("At least three unique landmark indices are required")
    if (indices < 0).any() or (indices >= len(true)).any():
        raise ValueError("Landmark index out of range")
    mask = np.ones(len(true), dtype=bool)
    mask[indices] = False
    held_out = np.flatnonzero(mask)
    if len(held_out) < 2:
        raise ValueError("Need at least two held-out nodes for Procrustes scoring")
    # The solver has no access to held-out reference coordinates.
    predicted, diagnostics = multilaterate_landmarks(
        true[indices], distances[np.ix_(held_out, indices)]
    )
    reference = true[held_out]
    for name, coords in (("Held-out reference", reference), ("Recovered", predicted)):
        if np.linalg.norm(coords - coords.mean(axis=0)) == 0:
            raise ValueError(f"{name} coordinates have zero spread")
    aligned_true, aligned_pred, disparity = procrustes(reference, predicted)
    metrics = {"disparity": float(disparity)}
    for axis in range(2):
        a, b = aligned_true[:, axis], aligned_pred[:, axis]
        tolerance = 100 * np.finfo(np.float64).eps
        constant = np.ptp(a) <= tolerance or np.ptp(b) <= tolerance
        metrics[f"pearson_r_axis{axis + 1}"] = (
            float("nan") if constant else float(pearsonr(a, b).statistic)
        )
        metrics[f"spearman_rho_axis{axis + 1}"] = (
            float("nan") if constant else float(spearmanr(a, b).statistic)
        )
    if return_details:
        return metrics, {
            "landmark_indices": indices.copy(), "held_out_indices": held_out,
            "pred_coords_held_out": predicted, "aligned_true": aligned_true,
            "aligned_pred": aligned_pred, "solver_diagnostics": diagnostics,
        }
    return metrics


def _synthetic_geometry():
    rng = np.random.default_rng(42)
    landmarks = np.array([[-2., -2.], [2., -2.], [2., 2.], [-2., 2.]])
    coords = np.vstack([landmarks, rng.normal(size=(16, 2))])
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return coords, distances, np.arange(4)


def test_dataset_and_forward_backward():
    from torch.utils.data import DataLoader

    torch.manual_seed(42)
    embeddings = {i: torch.randn(1280, requires_grad=True) for i in range(8)}
    attributes = {i: i * 2 for i in range(8)}
    frame = pd.DataFrame({"node_A": [i % 8 for i in range(32)],
                          "node_B": [(i + 3) % 8 for i in range(32)],
                          "distance": np.linspace(0, 2, 32)})
    dataset = AttributedPairDataset(frame, embeddings, attributes)
    a, b, delta, target = next(iter(DataLoader(dataset, batch_size=32)))
    assert a.shape == b.shape == (32, 1280)
    assert delta.shape == target.shape == (32,)
    assert delta[0].item() == 6 and not a.requires_grad
    torch.testing.assert_close(a[0], embeddings[0].detach())
    model = AugmentedDistanceHead()
    scores = model(a, b, delta)
    assert scores.shape == (32,)
    loss = nn.SmoothL1Loss()(torch.nn.functional.softplus(scores), target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    assert all(e.grad is None for e in embeddings.values())
    assert model(a[:1], b[:1], delta[:1, None]).shape == (1,)


def test_exact_recovery_and_held_out_only_scoring():
    coords, distances, landmarks = _synthetic_geometry()
    metrics, details = evaluate_out_of_sample_landmarks(
        coords, distances, landmarks, return_details=True
    )
    np.testing.assert_allclose(details["pred_coords_held_out"], coords[4:], atol=1e-6)
    assert metrics["disparity"] < 1e-12
    assert metrics["pearson_r_axis1"] > .999999
    assert metrics["spearman_rho_axis2"] > .999999
    assert not np.intersect1d(landmarks, details["held_out_indices"]).size
    _, _, expected = procrustes(coords[4:], details["pred_coords_held_out"])
    np.testing.assert_allclose(metrics["disparity"], expected)


def test_no_held_out_truth_or_unused_distance_leakage():
    coords, distances, landmarks = _synthetic_geometry()
    _, original = evaluate_out_of_sample_landmarks(
        coords, distances, landmarks, return_details=True
    )
    changed = coords.copy()
    changed[4:] = np.random.default_rng(9).normal(size=(16, 2)) * 30
    sparse = np.full_like(distances, np.nan)
    sparse[4:, :4] = distances[4:, :4]
    metrics, second = evaluate_out_of_sample_landmarks(
        changed, sparse, landmarks, return_details=True
    )
    np.testing.assert_array_equal(original["pred_coords_held_out"],
                                  second["pred_coords_held_out"])
    assert metrics["disparity"] > .1


def test_gradient_and_noisy_multilateration():
    from scipy.optimize import check_grad

    coords, distances, landmarks = _synthetic_geometry()
    target = distances[7, landmarks]
    fun = lambda x: _stress_and_gradient(x, coords[landmarks], target)[0]
    jac = lambda x: _stress_and_gradient(x, coords[landmarks], target)[1]
    assert check_grad(fun, jac, np.array([.3, -.4])) < 1e-6
    noisy = distances[4:, :4] + np.random.default_rng(6).normal(0, .02, (16, 4))
    pred, diagnostic = multilaterate_landmarks(coords[:4], noisy)
    assert np.sqrt(np.mean((pred - coords[4:]) ** 2)) < .05
    assert all(row["accepted_starts"] > 0 for row in diagnostic)


def test_invalid_inputs_and_constant_axis():
    import pytest

    coords, distances, landmarks = _synthetic_geometry()
    for bad_indices in ([0, 0, 2], [0, 1, 50], [0., 1., 2.], [0, 1]):
        with pytest.raises(ValueError):
            evaluate_out_of_sample_landmarks(coords, distances, bad_indices)
    bad = distances.copy()
    bad[4, 0] = -1
    with pytest.raises(ValueError):
        evaluate_out_of_sample_landmarks(coords, bad, landmarks)
    with pytest.raises(ValueError, match="collinear"):
        multilaterate_landmarks(np.array([[0, 0], [1, 0], [2, 0]]), np.ones((2, 3)))
    coords[4:, 1] = 0
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    metrics = evaluate_out_of_sample_landmarks(coords, distances, landmarks)
    assert np.isnan(metrics["pearson_r_axis2"])
    assert np.isnan(metrics["spearman_rho_axis2"])


def test_feature_order_and_dataset_validation():
    import pytest

    a, b = torch.ones(1, 1280), torch.full((1, 1280), 3.)
    model = AugmentedDistanceHead().eval()
    observed = []
    handle = model.mlp[0].register_forward_pre_hook(
        lambda module, inputs: observed.append(inputs[0].detach().clone())
    )
    with torch.no_grad():
        model(a, b, torch.tensor([7.]))
    handle.remove()
    expected = torch.cat([a, b, b - a, a * b, torch.tensor([[7.]])], dim=1)
    torch.testing.assert_close(observed[0], expected)
    frame = pd.DataFrame({"node_A": [0], "node_B": [1], "distance": [1.]})
    embeddings = {0: a[0], 1: b[0]}
    with pytest.raises(KeyError):
        AttributedPairDataset(frame, embeddings, {0: 1})
    with pytest.raises(ValueError):
        AttributedPairDataset(frame, embeddings, {0: 1, 1: 2.5})
    with pytest.raises(ValueError):
        AttributedPairDataset(frame, {0: torch.ones(12), 1: b[0]}, {0: 1, 1: 2})


def test_solver_failure_is_explicit(monkeypatch):
    import sys
    from types import SimpleNamespace
    import pytest

    def failed_minimize(*args, **kwargs):
        return SimpleNamespace(fun=1., x=np.zeros(2), jac=np.ones(2), success=False)

    monkeypatch.setattr(sys.modules[__name__], "minimize", failed_minimize)
    coords, distances, _ = _synthetic_geometry()
    with pytest.raises(RuntimeError, match="no converged start"):
        multilaterate_landmarks(coords[:4], distances[4:, :4])


if __name__ == "__main__":
    # Limit threads only in the demo, without changing imported-module behavior.
    torch.set_num_threads(1)
    test_dataset_and_forward_backward()
    coords, distances, landmarks = _synthetic_geometry()
    scores = evaluate_out_of_sample_landmarks(coords, distances, landmarks)
    assert scores["disparity"] < 1e-12
    print("B=32: forward/backward passed; source embeddings remain frozen.")
    print("Synthetic out-of-sample metrics (16 held-out nodes):")
    print(scores)
