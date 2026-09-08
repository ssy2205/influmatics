import numpy as np
from scipy.optimize import least_squares

def estimate_node_coordinates(anchor_coords, distances_to_anchors):
    """
    Estimate 2D coordinates of new nodes using multilateration based on distances to anchor points.
    
    Args:
        anchor_coords: np.ndarray of shape (K, 2) containing coordinates of K anchors.
        distances_to_anchors: np.ndarray of shape (N, K) containing predicted distances from N nodes to K anchors.
        
    Returns:
        estimated_coords: np.ndarray of shape (N, 2) containing estimated 2D coordinates for N nodes.
    """
    N, K = distances_to_anchors.shape
    assert anchor_coords.shape[0] == K
    assert anchor_coords.shape[1] == 2
    
    estimated_coords = np.zeros((N, 2))
    
    def residuals(coords, anchors, target_distances):
        # coords: (2,)
        # anchors: (K, 2)
        # target_distances: (K,)
        dist = np.linalg.norm(anchors - coords, axis=1)
        return dist - target_distances

    # Simple initial guess: center of the anchor points
    initial_guess = np.mean(anchor_coords, axis=0)
    
    for i in range(N):
        target_dist = distances_to_anchors[i]
        # Optimize coordinates using least squares
        res = least_squares(residuals, initial_guess, args=(anchor_coords, target_dist))
        estimated_coords[i] = res.x
        
    return estimated_coords

if __name__ == '__main__':
    # 1. Generate 5 synthetic anchors
    np.random.seed(42)
    K = 5
    anchor_coords = np.random.rand(K, 2) * 100
    print(f"Generated {K} anchor coordinates:\n{anchor_coords}\n")
    
    # 2. Generate 10 synthetic nodes (ground truth)
    N = 10
    true_node_coords = np.random.rand(N, 2) * 100
    
    # 3. Calculate true distances from nodes to anchors
    distances_to_anchors = np.zeros((N, K))
    for i in range(N):
        distances_to_anchors[i] = np.linalg.norm(anchor_coords - true_node_coords[i], axis=1)
        
    # Optional: add a tiny amount of noise to simulate real-world observations
    # distances_to_anchors += np.random.normal(0, 1e-4, size=(N, K))
    
    # 4. Estimate node coordinates based purely on anchor coordinates and distances
    estimated_coords = estimate_node_coordinates(anchor_coords, distances_to_anchors)
    
    # 5. Verify the recovery by computing the error
    print("Verification (True vs Estimated Coordinates):")
    for i in range(N):
        error = np.linalg.norm(true_node_coords[i] - estimated_coords[i])
        print(f"Node {i+1}:")
        print(f"  True:      [{true_node_coords[i, 0]:.4f}, {true_node_coords[i, 1]:.4f}]")
        print(f"  Estimated: [{estimated_coords[i, 0]:.4f}, {estimated_coords[i, 1]:.4f}]")
        print(f"  Error:     {error:.6e}")
        
    avg_error = np.mean(np.linalg.norm(true_node_coords - estimated_coords, axis=1))
    print(f"\nAverage Localization Error: {avg_error:.6e}")
    
    if avg_error < 1e-5:
        print("SUCCESS: 2D coordinates stably and accurately recovered.")
    else:
        print("WARNING: Localization error is large.")
