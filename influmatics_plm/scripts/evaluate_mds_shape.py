"""Recover 2D coordinates from distances and evaluate Procrustes shape fit.

Coordinate rows and distance-matrix rows/columns must use the same node order.
CLI inputs are NPY arrays. MDS uses predicted distances only, never true
coordinates for initialization or restart selection.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import MDS


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric array")
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must not contain NaN or infinity")
    return array


def _coords(value, name):
    array = _real_array(value, name)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 2:
        raise ValueError(f"{name} must have shape (N, 2), with N >= 2")
    return array


def reconstruct_2d(pred_distance_matrix, *, n_init=4, max_iter=1000, eps=1e-6):
    """Metric MDS with a fixed seed; return raw reconstructed (N, 2) coordinates.

    Reject missing/negative distances and a nonzero diagonal. Tiny symmetry
    and diagonal errors within 1e-8 are corrected on a copy. Zero off-diagonal
    distances are valid; an entirely zero matrix has no identifiable shape.
    Runtime is approximately O(n_init * iterations * N**2), memory O(N**2).
    """
    distances = _real_array(pred_distance_matrix, "pred_distance_matrix")
    if (distances.ndim != 2 or distances.shape[0] != distances.shape[1]
            or len(distances) < 2):
        raise ValueError("pred_distance_matrix must be square with N >= 2")
    if (distances < 0).any():
        raise ValueError("Distances must be nonnegative")
    if not np.allclose(distances, distances.T, atol=1e-8, rtol=0):
        raise ValueError("pred_distance_matrix must be symmetric")
    if not np.allclose(np.diag(distances), 0, atol=1e-8, rtol=0):
        raise ValueError("Distance-matrix diagonal must be zero")
    distances = distances * 0.5 + distances.T * 0.5
    np.fill_diagonal(distances, 0.0)
    if not np.any(distances > 0):
        raise ValueError("All-zero distances cannot define a shape")
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
              n_init=n_init, max_iter=max_iter, eps=eps)
    return mds.fit_transform(distances)


def align_and_score(true_coords_2d, pred_coords_2d):
    """Return (metrics, aligned_true, aligned_pred); both axes share one fit.

    Procrustes removes translation, rotation/reflection and overall scale.
    Correlations on constant or numerically collapsed axes are undefined (NaN).
    Axis correlations depend on the reference orientation, unlike disparity.
    """
    true = _coords(true_coords_2d, "true_coords_2d")
    pred = _coords(pred_coords_2d, "pred_coords_2d")
    if true.shape != pred.shape:
        raise ValueError("True and predicted coordinates must have identical shapes")
    for name, array in (("true", true), ("predicted", pred)):
        if not np.any(array != array[0]):
            raise ValueError(f"{name} coordinates need at least two distinct points")
    aligned_true, aligned_pred, disparity = procrustes(true, pred)
    metrics = {"disparity": float(disparity)}
    tolerance = 100 * np.finfo(np.float64).eps
    for axis in range(2):
        x, y = aligned_true[:, axis], aligned_pred[:, axis]
        if np.ptp(x) <= tolerance or np.ptp(y) <= tolerance:
            pearson, spearman = float("nan"), float("nan")
        else:
            pearson = float(pearsonr(x, y).statistic)
            spearman = float(spearmanr(x, y).statistic)
        metrics[f"pearson_r_axis{axis + 1}"] = pearson
        metrics[f"spearman_rho_axis{axis + 1}"] = spearman
    return metrics, aligned_true, aligned_pred


def evaluate_shape(true_coords_2d, pred_distance_matrix, *,
                   return_coordinates=False, n_init=4, max_iter=1000, eps=1e-6):
    """Return the requested five-metric dictionary by default.

    With return_coordinates=True, return
    (metrics, pred_coords_2d, aligned_true, aligned_pred) without repeating MDS.
    """
    true = _coords(true_coords_2d, "true_coords_2d")
    if np.shape(pred_distance_matrix) != (len(true), len(true)):
        raise ValueError("Distance matrix dimensions must match coordinate row count")
    if not np.any(true != true[0]):
        raise ValueError("True coordinates need at least two distinct points")
    pred = reconstruct_2d(pred_distance_matrix, n_init=n_init,
                          max_iter=max_iter, eps=eps)
    metrics, aligned_true, aligned_pred = align_and_score(true, pred)
    if return_coordinates:
        return metrics, pred, aligned_true, aligned_pred
    return metrics


def plot_alignment(aligned_true, aligned_pred, *, connect_pairs=True,
                   save_path=None, ax=None):
    """Overlay already aligned coordinates; return (figure, axes).

    Coordinates should be the outputs of align_and_score/evaluate_shape.
    Call plt.show() explicitly when an interactive window is desired.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    true = _coords(aligned_true, "aligned_true")
    pred = _coords(aligned_pred, "aligned_pred")
    if true.shape != pred.shape:
        raise ValueError("Aligned coordinates must have matching shapes")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    if connect_pairs:
        ax.add_collection(LineCollection(np.stack([true, pred], axis=1),
                                        colors="0.6", linewidths=0.6,
                                        alpha=0.4, zorder=1))
    ax.scatter(true[:, 0], true[:, 1], s=42, facecolors="none",
               edgecolors="#0072B2", label="Reference", zorder=2)
    ax.scatter(pred[:, 0], pred[:, 1], s=30, marker="x", color="#D55E00",
               label="Reconstructed", zorder=3)
    disparity = float(np.sum((true - pred) ** 2))
    ax.set(title=f"Procrustes alignment | disparity = {disparity:.6g}",
           xlabel="Aligned axis 1 (normalized)",
           ylabel="Aligned axis 2 (normalized)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(path, dpi=180, bbox_inches="tight")
    return ax.figure, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--true-coords", required=True, help="(N, 2) NPY file")
    parser.add_argument("--pred-distances", required=True, help="(N, N) NPY file")
    parser.add_argument("--output-dir", default="reports/mds_shape")
    parser.add_argument("--n-init", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()
    metrics, pred, aligned_true, aligned_pred = evaluate_shape(
        np.load(args.true_coords, allow_pickle=False),
        np.load(args.pred_distances, allow_pickle=False),
        return_coordinates=True, n_init=args.n_init,
        max_iter=args.max_iter, eps=args.eps,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "coordinates.npz", pred_coords_2d=pred,
             aligned_true=aligned_true, aligned_pred=aligned_pred)
    serializable = {key: value if np.isfinite(value) else None
                    for key, value in metrics.items()}
    payload = json.dumps(serializable, indent=2, allow_nan=False)
    (output / "metrics.json").write_text(payload + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, _ = plot_alignment(aligned_true, aligned_pred, save_path=output / "alignment.png")
    plt.close(fig)
    print(payload)


if __name__ == "__main__":
    main()
