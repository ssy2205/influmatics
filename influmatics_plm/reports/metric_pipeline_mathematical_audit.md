Mathematical and evaluation-integrity audit — 2026-09-07

The proposed absolute-difference MLP with a final Softplus cannot satisfy an exact zero diagonal. The existing reconstruction also incorporates predictions into its reference, so its Procrustes disparity is not an independent measure of reconstruction accuracy.

Scope: source review and abstract mathematical analysis. The repository connects the supplied architecture to viral sequence embeddings and antigenic-distance reconstruction. An end-to-end predictive refactor and training implementation are excluded because improving this application could support immune-evasion modeling. No datasets, embeddings, or checkpoints were loaded; no training or reconstruction was run. Saved split integrity and numerical results have not been certified.

1. **The original head has no guarantees for the requested geometric properties.**

   In [models.py](/Users/shiftyellow/Documents/project-local/influmatics_project/influmatics_code/influmatics_plm/scripts/models.py:20), exchanging the endpoints changes the ordered concatenation of the first two feature blocks. Symmetric difference and product features do not cancel the unrestricted dependence on those ordered blocks. Symmetry is therefore not guaranteed.

   When both inputs equal u, the concatenated features are [u, u, 0, u*u], which are generally nonzero. The network also has unconstrained biases. Its self-distance need not vanish. The unrestricted final affine layer can return negative values.

   Dropout makes separate training-mode evaluations stochastic. Even with an otherwise symmetric input representation, independently sampled masks do not provide exact equality across separate calls. Evaluation-mode determinism and mathematical symmetry are separate requirements.

   The reconstruction script averages the two endpoint orders, clips negative values, and initializes the matrix diagonal to zero. Those operations impose three properties on the assembled matrix; they do not establish those properties for DistanceHead itself, and they do not establish a triangle inequality. Source: [06_reconstruct_mds_map.py](/Users/shiftyellow/Documents/project-local/influmatics_project/influmatics_code/influmatics_plm/scripts/06_reconstruct_mds_map.py:169).

2. **The requested Softplus correction is internally inconsistent.**

   Write delta = |u-v| to avoid overwriting the endpoint name v. For a deterministic finite-valued function f, the proposed score is

   d(u,v) = softplus(f(delta)).

   On the diagonal, delta = 0, so

   d(u,u) = log(1 + exp(f(0))) > 0.

   This follows directly from the [PyTorch Softplus definition](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Softplus.html) and holds for finite logits in real arithmetic. Floating-point underflow can produce a numerical zero, but that is not an architectural zero-diagonal guarantee.

   Thus absolute differences address deterministic endpoint symmetry, and Softplus addresses non-negativity, but their composition does not address the zero diagonal. Subtracting the value at delta = 0 is also not a general proof of non-negativity for an unrestricted f.

3. **The listed conditions are weaker than all metric axioms.**

   Full identity of indiscernibles is d(u,v) = 0 if and only if u = v. The condition d(u,u) = 0 states only one implication. A metric additionally satisfies d(u,w) <= d(u,v) + d(v,w).

   A simple counterexample is d(x,y) = |x-y|^2 on the real line. It is symmetric, nonnegative, and zero exactly when x = y, but d(0,2) = 4 > d(0,1) + d(1,2) = 2. Enforcing the three displayed properties therefore cannot justify a claim that all metric axioms hold.

   Shared Siamese weights alone also do not prove separation: a shared representation can map distinct inputs to the same output. Moreover, a metric on embedding vectors cannot distinguish two different entity IDs whose frozen embeddings coincide.

   Nonnegative regression targets alone do not establish that the observations are mutually symmetric, satisfy triangle inequalities, or admit an exact two-dimensional Euclidean realization. The word “metric” in metric MDS refers to fitting distance magnitudes rather than only their ordering; it is not a certificate that arbitrary input dissimilarities already form a mathematical metric.

4. **Prediction-based reference completion makes evaluation circular.**

   [06_reconstruct_mds_map.py](/Users/shiftyellow/Documents/project-local/influmatics_project/influmatics_code/influmatics_plm/scripts/06_reconstruct_mds_map.py:182) replaces every missing reference entry with a candidate prediction. The reference configuration consequently changes when the candidate model changes.

   Consider a valid, nondegenerate predicted distance matrix, an observed zero diagonal, and no observed off-diagonal reference distances. The completed reference then equals the prediction matrix exactly. Identical deterministic MDS fits followed by Procrustes can return zero disparity despite there being no observed pairwise evidence at all. With partial observations, the same dependency can artificially improve apparent agreement; it need not affect every dataset or optimizer outcome monotonically.

   Reusing the MDS estimator is not the cause of this dependency. The dependency already exists in its input matrix.

5. **Weighted MDS excludes missing observations but does not establish a unique reference shape.**

   Let E contain observed unordered off-diagonal pairs. The reference fitting criterion is

   S(X) = sum over (i,j) in E of (||x_i-x_j||_2 - D_ij)^2.

   This is binary-weight stress with W_ij = 1 for observed pairs and W_ij = 0 for missing pairs, and W_ii = 0. An observed zero distance remains an observation. No prediction contributes to this reference criterion. SMACOF minimizes stress by majorization; the weighted treatment of missing distances is described by [de Leeuw and Mair](https://escholarship.org/content/qt8362w1v8/qt8362w1v8_noSplash_0eb83392646930557236ff6271e4c29e.pdf).

   The sum above deliberately excludes undefined entries: multiplying NaN by a numerical zero does not make it disappear. Absence of observations and observed zero distances have different meanings.

   A disconnected observation graph cannot determine relative component positions. Connectivity alone is also insufficient: three nodes with only distances (1,2) and (2,3) observed can form different angles at node 2 while fitting both observations exactly. These configurations need not be congruent. Consequently, an estimated reference can be independent of predictions and still be underdetermined. A fixed random seed supplies reproducibility, not missing geometric evidence or a global-optimality guarantee.

   The documented [scikit-learn smacof API](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.smacof.html) exposes no general observation-weight matrix. Its nonmetric zero-as-missing convention changes the objective and loses valid observed zeros; it is not an equivalent implementation of the weighted metric objective.

   Shortest-path completion introduces graph-derived estimates. It is not recovered ground truth. Path lengths can exceed unknown direct distances; with inconsistent observed distances, a shorter path can even replace an observed edge length. Disconnected pairs have no finite path. Filling only missing entries does not in general make inconsistent retained observations into a metric.

6. **Procrustes disparity measures shape agreement and cannot certify distance calibration.**

   [SciPy Procrustes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html) removes translation and permits rotation, reflection, and scaling. Configurations related by a uniform change of scale can therefore have zero disparity while their raw pairwise distances disagree substantially.

   Coordinate rows must correspond to the same node IDs in the same order. Collapsed configurations have no meaningful normalized shape. Distance error on observed pairs and MDS stress describe information that Procrustes discards. With a sparse reference, disparity also reflects reference nonidentifiability, optimization choices, and projection into two dimensions.

   Axis-wise correlations depend on the chosen reference orientation. In particular, Spearman correlation of coordinate axes is not Spearman correlation of pairwise distances. A reference estimated from incomplete dissimilarities should be described as an estimated reference configuration, rather than known true coordinates.

7. **The split source supports the stated test holdout, but saved artifacts remain unverified.**

   Define V_s as the union of both endpoint-ID columns in split s. The requested condition is V_test intersect (V_train union V_val) = empty.

   [temporal_group_split.py](/Users/shiftyellow/Documents/project-local/influmatics_project/influmatics_code/influmatics_plm/scripts/temporal_group_split.py:67) constructs the test-node set and excludes every edge touching those IDs from train and validation. Under its assumption of consistently represented IDs, that construction enforces the stated test-node condition.

   The function explicitly permits train-validation node overlap. This does not violate invariant 4 as written; requiring all three sets to be mutually disjoint would be a stronger specification. Splitting edge rows or grouping on only one endpoint would not establish node disjointness.

   [05_train_distance_head.py](/Users/shiftyellow/Documents/project-local/influmatics_project/influmatics_code/influmatics_plm/scripts/05_train_distance_head.py:54) reads saved split files without checking their endpoint-set intersections. Source inspection therefore cannot certify the files used in a particular run. Nor does freezing an encoder by itself establish that preprocessing, encoder provenance, or model selection is independent of held-out information.

8. **The proposed loss combination is a surrogate, not a guarantee about Spearman rho.**

   Smooth L1 concerns numerical distance residuals. Margin ranking concerns ordered pairs of scores. Combining the two does not enforce metric axioms and is not direct optimization of Spearman correlation. The [PyTorch MarginRankingLoss documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MarginRankingLoss.html) defines a margin-based pairwise ordering objective, which differs from correlation of ranks over an evaluation set. Equal targets have no strict ordering.

   In the current evaluation source, undefined Spearman values are changed to 0.0. This conflates an undefined statistic with a defined zero correlation. Reported metrics are also rounded before checkpoint comparison, so the selection uses reduced precision. Source: [05_train_distance_head.py](/Users/shiftyellow/Documents/project-local/influmatics_project/influmatics_code/influmatics_plm/scripts/05_train_distance_head.py:35).

9. **Additional source-level integrity concerns remain.**

   Repeated observations of the same unordered pair overwrite earlier entries in the reconstruction matrix, making the reference depend on row order if values disagree. No replicate-resolution policy is recorded at that assignment. Source: [06_reconstruct_mds_map.py](/Users/shiftyellow/Documents/project-local/influmatics_project/influmatics_code/influmatics_plm/scripts/06_reconstruct_mds_map.py:163).

   The dataset class silently substitutes a zero embedding for a missing key. The current training caller filters keys first, which limits this problem there, but the dataset class itself does not preserve a distinction between missing data and an actual zero vector. Source: [dataset.py](/Users/shiftyellow/Documents/project-local/influmatics_project/influmatics_code/influmatics_plm/scripts/dataset.py:19).

   The standalone reconstruction snippet does not validate n_nodes against either matrix, matrix symmetry, diagonal values, finite observed entries, nonnegative distances, or node ordering. The observed matrix is sparse, but dense prediction matrices and ordinary dense MDS still require quadratic storage. Neither sparsity nor the frozen encoder makes that reconstruction cost disappear.

Validation performed: reviewed the relevant source and primary library/method documentation; checked the claims against the mathematical counterexamples above. This is a documentation-only audit, with no changes to executable code or existing numerical reports.
