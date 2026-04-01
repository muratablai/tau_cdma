"""
emergent.py — Emergent Quantum Numbers as Projections of Latent Spreading Codes
=================================================================================

Section 14 of the Unified Framework v4.5.

Implements:
  - LatentCodeModel: learns latent codes z_k and generator G_ψ from data
  - Emergence test: checks if known QNs are recoverable as linear projections of z_k
  - Conservation test: checks if z_parent ≈ f(z_daughter1, z_daughter2)
  - Dimensional counting (Prediction P9)
  - Toy gauge theory validation (Stage 1)
  - SM particle code space analysis (Stage 2)

Key equations from v4.1:
  a_k = G_ψ(z_k)  where z_k ∈ ℝ^d
  Q̂_k = w_Q^T z_k + c_Q  (linear projection)
  ε_Q = (1/K) Σ_k (Q_k - Q̂_k)²  (reconstruction error)
"""

import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans


class LatentCodeModel:
    """Learn latent spreading codes from observed data.

    Two-stage approach:
      1. Learn template matrix A via Poisson NMF
      2. Embed templates into latent space via PCA or autoencoder

    The latent code z_k captures the essential structure of template a_k
    in a low-dimensional space.

    Parameters
    ----------
    d : int — latent code dimensionality
    method : str — 'pca' (default) or 'autoencoder'
    """

    def __init__(self, d=12, method='pca'):
        self.d = d
        self.method = method
        self.z_codes = None       # (K, d) latent codes
        self.A_templates = None   # (M, K) learned templates
        self.pca_model = None
        self.explained_var = None

    def fit_from_templates(self, A):
        """Learn latent codes from a known template matrix.

        Each column of A is a spreading code; we find the latent representation.

        Parameters
        ----------
        A : ndarray (M, K) — template matrix

        Returns
        -------
        self (fitted)
        """
        self.A_templates = A.copy()
        M, K = A.shape

        if self.method == 'pca':
            # PCA on the template columns (transpose: K samples in M dimensions)
            d_use = min(self.d, K, M)
            pca = PCA(n_components=d_use)
            self.z_codes = pca.fit_transform(A.T)  # (K, d_use)
            self.pca_model = pca
            self.explained_var = pca.explained_variance_ratio_
            self.d = d_use
        elif self.method == 'autoencoder':
            # Simple linear autoencoder (equivalent to PCA for linear case)
            # For nonlinear: would use PyTorch, but keeping dependency-free
            d_use = min(self.d, K, M)
            pca = PCA(n_components=d_use)
            self.z_codes = pca.fit_transform(A.T)
            self.pca_model = pca
            self.explained_var = pca.explained_variance_ratio_
            self.d = d_use
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def fit_from_data(self, Y, K_components=7, nmf_iter=2000):
        """Learn latent codes from observed data via NMF + PCA.

        Parameters
        ----------
        Y : ndarray (n_samples, M) — observed histograms
        K_components : int — number of NMF components
        nmf_iter : int — NMF iterations

        Returns
        -------
        self
        """
        from .nmf import poisson_nmf
        A_hat, _, _ = poisson_nmf(Y, K_components, n_iter=nmf_iter)
        return self.fit_from_templates(A_hat)

    def reconstruct_templates(self):
        """Reconstruct templates from latent codes (for validation)."""
        if self.pca_model is not None and self.z_codes is not None:
            A_recon = self.pca_model.inverse_transform(self.z_codes).T  # (M, K)
            return np.maximum(A_recon, 0.0)
        return None

    def effective_dimensionality(self, threshold=0.01):
        """Compute effective dimensionality d_eff of the latent space.

        Uses explained variance ratio: d_eff = number of components with
        explained variance ratio > threshold.

        Also computes participation ratio: PR = (Σ λ_i)² / Σ λ_i²

        Returns
        -------
        d_eff : int — effective dimensionality (variance threshold)
        participation_ratio : float — continuous measure of dimensionality
        singular_values : ndarray — singular values of template matrix
        """
        if self.explained_var is None:
            return 0, 0.0, np.array([])

        d_eff = np.sum(self.explained_var > threshold)

        # Participation ratio
        pr = np.sum(self.explained_var)**2 / np.sum(self.explained_var**2)

        # SVD of original templates for more detailed analysis
        if self.A_templates is not None:
            _, s, _ = svd(self.A_templates, full_matrices=False)
        else:
            s = np.array([])

        return int(d_eff), float(pr), s


def emergence_test(z_codes, quantum_numbers, qn_names=None):
    """Test whether known quantum numbers are recoverable as linear projections.

    For each quantum number Q:
      Q̂_k = w_Q^T z_k + c_Q
      ε_Q = (1/K) Σ_k (Q_k - Q̂_k)²

    Parameters
    ----------
    z_codes : ndarray (K, d) — latent codes
    quantum_numbers : ndarray (K, n_qn) — true quantum numbers
    qn_names : list of str or None — names for each QN

    Returns
    -------
    results : dict with:
        'errors' : ndarray (n_qn,) — reconstruction error ε_Q per QN
        'r_squared' : ndarray (n_qn,) — R² score per QN
        'projections' : ndarray (n_qn, d) — projection vectors w_Q
        'intercepts' : ndarray (n_qn,) — intercepts c_Q
        'qn_names' : list of str
    """
    K, d = z_codes.shape
    n_qn = quantum_numbers.shape[1]

    if qn_names is None:
        qn_names = [f'QN_{i}' for i in range(n_qn)]

    errors = np.zeros(n_qn)
    r_squared = np.zeros(n_qn)
    projections = np.zeros((n_qn, d))
    intercepts = np.zeros(n_qn)

    for i in range(n_qn):
        Q = quantum_numbers[:, i]
        reg = LinearRegression()
        reg.fit(z_codes, Q)
        Q_hat = reg.predict(z_codes)

        errors[i] = np.mean((Q - Q_hat)**2)
        r_squared[i] = reg.score(z_codes, Q)
        projections[i] = reg.coef_
        intercepts[i] = reg.intercept_

    return {
        'errors': errors,
        'r_squared': r_squared,
        'projections': projections,
        'intercepts': intercepts,
        'qn_names': qn_names,
    }


def conservation_test(z_parent, z_daughters, mode='additive'):
    """Test whether latent codes satisfy conservation at vertices.

    Additive: z_parent ≈ z_d1 + z_d2
    Learned: z_parent ≈ f(z_d1, z_d2) for learned linear f

    Parameters
    ----------
    z_parent : ndarray (n_vertices, d) — parent latent codes
    z_daughters : list of ndarray (n_vertices, d) — daughter codes
    mode : str — 'additive' or 'learned'

    Returns
    -------
    results : dict with:
        'residual_norm' : float — mean ||z_parent - f(z_daughters)||
        'conservation_score' : float — 1 - (residual / ||z_parent||)
        'mode' : str
    """
    d = z_parent.shape[1]
    n = z_parent.shape[0]

    if mode == 'additive':
        z_sum = np.zeros_like(z_parent)
        for z_d in z_daughters:
            z_sum += z_d
        residual = z_parent - z_sum
        residual_norm = np.mean(np.linalg.norm(residual, axis=1))
        parent_norm = np.mean(np.linalg.norm(z_parent, axis=1))
        score = 1.0 - residual_norm / max(parent_norm, 1e-30)

    elif mode == 'learned':
        # Learn linear mapping: z_parent = W₁·z_d1 + W₂·z_d2 + b
        X = np.hstack(z_daughters)  # (n, d * n_daughters)
        reg = LinearRegression()
        reg.fit(X, z_parent)
        z_pred = reg.predict(X)
        residual = z_parent - z_pred
        residual_norm = np.mean(np.linalg.norm(residual, axis=1))
        parent_norm = np.mean(np.linalg.norm(z_parent, axis=1))
        score = 1.0 - residual_norm / max(parent_norm, 1e-30)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        'residual_norm': float(residual_norm),
        'conservation_score': float(score),
        'mode': mode,
    }


def discreteness_test(z_codes, max_clusters=15):
    """Test whether latent space has discrete cluster structure.

    If quantum numbers are truly fundamental, z_k should cluster
    into discrete groups.

    Uses silhouette score to assess clustering quality.

    Parameters
    ----------
    z_codes : ndarray (K, d) — latent codes
    max_clusters : int — maximum number of clusters to try

    Returns
    -------
    results : dict with:
        'optimal_k' : int — optimal number of clusters
        'silhouette_scores' : list — score per K
        'inertias' : list — inertia per K
        'is_discrete' : bool — True if clustering is significantly better than random
    """
    from sklearn.metrics import silhouette_score

    K = z_codes.shape[0]
    max_k = min(max_clusters, K - 1)

    if max_k < 2:
        return {
            'optimal_k': 1,
            'silhouette_scores': [],
            'inertias': [],
            'is_discrete': False,
        }

    scores = []
    inertias = []

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(z_codes)
        inertias.append(km.inertia_)
        try:
            s = silhouette_score(z_codes, labels)
            scores.append(s)
        except:
            scores.append(-1.0)

    optimal_k = np.argmax(scores) + 2 if scores else 2
    best_score = max(scores) if scores else 0.0

    return {
        'optimal_k': int(optimal_k),
        'silhouette_scores': scores,
        'inertias': inertias,
        'is_discrete': best_score > 0.5,  # threshold for "meaningful" clustering
    }


# === Toy Gauge Theory Validation (Stage 1 from v4.1) ===

def generate_toy_gauge_theory(n_particles=20, d_true=4, n_bins=100,
                                n_qn=3, rng=None):
    """Generate toy U(1)×SU(2) data for validation.

    Creates synthetic particles with:
      - d_true independent quantum numbers (some integer, some continuous)
      - Templates generated as functions of quantum numbers
      - Known conservation laws at vertices

    Parameters
    ----------
    n_particles : int — number of distinct "particle types"
    d_true : int — true number of independent parameters
    n_bins : int — detector bins
    n_qn : int — number of discrete quantum numbers (<= d_true)
    rng : numpy RNG

    Returns
    -------
    data : dict with templates, QNs, decay vertices, etc.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Generate discrete quantum numbers
    qn = rng.integers(-2, 3, size=(n_particles, n_qn))

    # Generate continuous parameters (mass, width-like)
    continuous = rng.exponential(1.0, size=(n_particles, d_true - n_qn))
    continuous[:, 0] = np.abs(continuous[:, 0]) * 1000 + 200  # "mass"

    # Full latent code
    z_true = np.hstack([qn.astype(float), continuous])

    # Generate templates as nonlinear function of z
    # a_k(m) = Σ_j w_j · φ_j(m; z_k)
    m_bins = np.linspace(0, 2000, n_bins)
    A = np.zeros((n_bins, n_particles))
    for k in range(n_particles):
        mass_k = continuous[k, 0]
        width_k = continuous[k, 1] * 100 if d_true - n_qn > 1 else 100.0
        # BW-like template centered at "mass" with "width"
        for q in range(n_qn):
            mass_k += qn[k, q] * 50  # QNs shift the mass
        A[:, k] = np.exp(-0.5 * ((m_bins - mass_k) / max(width_k, 10.0))**2)
        A[:, k] /= max(np.sum(A[:, k]), 1e-30) * (m_bins[1] - m_bins[0])

    # Generate conserving "decay" vertices
    n_vertices = min(n_particles // 3, 10)
    vertices = []
    for v in range(n_vertices):
        parent = v
        d1, d2 = v + n_vertices, v + 2 * n_vertices
        if d2 < n_particles:
            # Conservation: QNs of parent = QNs of daughters (approximately)
            # In toy model, we enforce this
            qn[d1] = rng.integers(-1, 2, n_qn)
            qn[d2] = qn[parent] - qn[d1]  # exact conservation
            z_true[d1, :n_qn] = qn[d1]
            z_true[d2, :n_qn] = qn[d2]
            vertices.append((parent, d1, d2))

    return {
        'A': A,
        'z_true': z_true,
        'qn': qn,
        'continuous': continuous,
        'vertices': vertices,
        'm_bins': m_bins,
        'n_particles': n_particles,
        'd_true': d_true,
        'n_qn': n_qn,
    }


def run_stage1_validation(d_latent=6, verbose=True):
    """Run Stage 1: toy gauge theory validation.

    Generates toy data, learns latent codes, tests emergence and conservation.

    Key insight: template-space dimensionality from PCA is generally HIGHER
    than the parameter-space dimensionality d_true because the mapping from
    parameters to templates is nonlinear (e.g. peak position is a nonlinear
    function of mass). We use:
      - Participation ratio as the primary d_eff measure (robust to noise)
      - Polynomial features for QN recovery (handles mild nonlinearity)

    Returns
    -------
    results : dict with all test outcomes
    """
    from sklearn.preprocessing import PolynomialFeatures

    # Generate toy data
    toy = generate_toy_gauge_theory(n_particles=20, d_true=4, n_bins=100, n_qn=3)

    # Learn latent codes — use d_true + buffer, not overgenerous d_latent
    d_use = min(d_latent, toy['n_particles'] - 1)
    model = LatentCodeModel(d=d_use, method='pca')
    model.fit_from_templates(toy['A'])

    # Effective dimensionality
    d_eff, pr, sv = model.effective_dimensionality()

    # QN recovery with polynomial features (degree 2) to handle nonlinearity
    qn_names = ['Q', 'I3', 'S'][:toy['n_qn']]
    z = model.z_codes

    # First try linear (standard emergence test)
    emerge_linear = emergence_test(z, toy['qn'], qn_names=qn_names)

    # Then try polynomial features for better recovery
    poly = PolynomialFeatures(degree=2, include_bias=False)
    z_poly = poly.fit_transform(z)

    emerge_poly = emergence_test(z_poly, toy['qn'], qn_names=qn_names)

    # Conservation test at vertices
    consv = None
    if toy['vertices']:
        parents = np.array([toy['vertices'][i][0] for i in range(len(toy['vertices']))])
        d1s = np.array([toy['vertices'][i][1] for i in range(len(toy['vertices']))])
        d2s = np.array([toy['vertices'][i][2] for i in range(len(toy['vertices']))])

        consv = conservation_test(
            z[parents], [z[d1s], z[d2s]], mode='additive'
        )

    # Discreteness test
    discrete = discreteness_test(model.z_codes)

    results = {
        'd_eff': d_eff,
        'participation_ratio': pr,
        'singular_values': sv,
        'emergence': emerge_poly,        # polynomial (primary)
        'emergence_linear': emerge_linear,  # linear (for comparison)
        'conservation': consv,
        'discreteness': discrete,
        'd_true': toy['d_true'],
        'n_qn': toy['n_qn'],
    }

    if verbose:
        print("=== Stage 1: Toy Gauge Theory Validation ===")
        print(f"True dimensionality: {toy['d_true']}, QNs: {toy['n_qn']}")
        print(f"PCA d_eff (threshold): {d_eff}, participation ratio: {pr:.2f}")
        print(f"\nQN recovery (polynomial features, degree 2):")
        for i, name in enumerate(emerge_poly['qn_names']):
            print(f"  {name}: ε={emerge_poly['errors'][i]:.4f}, "
                  f"R²={emerge_poly['r_squared'][i]:.4f} "
                  f"(linear R²={emerge_linear['r_squared'][i]:.4f})")
        if consv:
            print(f"\nConservation test: score={consv['conservation_score']:.4f}")
        print(f"\nDiscreteness: optimal_k={discrete['optimal_k']}, "
              f"discrete={discrete['is_discrete']}")

    return results


# === SM Particle Code Space Analysis (Stage 2 sketch) ===

# Standard Model particle quantum numbers for reference
# (Q, B, Le, Lμ, Lτ, S, C, B', T)
SM_PARTICLES = {
    'e-':       (-1, 0, 1, 0, 0, 0, 0, 0, 0),
    'e+':       (+1, 0,-1, 0, 0, 0, 0, 0, 0),
    'mu-':      (-1, 0, 0, 1, 0, 0, 0, 0, 0),
    'mu+':      (+1, 0, 0,-1, 0, 0, 0, 0, 0),
    'tau-':     (-1, 0, 0, 0, 1, 0, 0, 0, 0),
    'tau+':     (+1, 0, 0, 0,-1, 0, 0, 0, 0),
    'nu_e':     ( 0, 0, 1, 0, 0, 0, 0, 0, 0),
    'nu_mu':    ( 0, 0, 0, 1, 0, 0, 0, 0, 0),
    'nu_tau':   ( 0, 0, 0, 0, 1, 0, 0, 0, 0),
    'u':        (2/3, 1/3, 0, 0, 0, 0, 0, 0, 0),
    'd':        (-1/3,1/3, 0, 0, 0, 0, 0, 0, 0),
    's':        (-1/3,1/3, 0, 0, 0,-1, 0, 0, 0),
    'c':        (2/3, 1/3, 0, 0, 0, 0, 1, 0, 0),
    'b':        (-1/3,1/3, 0, 0, 0, 0, 0,-1, 0),
    't':        (2/3, 1/3, 0, 0, 0, 0, 0, 0, 1),
    'pi+':      (+1, 0, 0, 0, 0, 0, 0, 0, 0),
    'pi-':      (-1, 0, 0, 0, 0, 0, 0, 0, 0),
    'pi0':      ( 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'K+':       (+1, 0, 0, 0, 0, 1, 0, 0, 0),
    'K-':       (-1, 0, 0, 0, 0,-1, 0, 0, 0),
    'proton':   (+1, 1, 0, 0, 0, 0, 0, 0, 0),
    'neutron':  ( 0, 1, 0, 0, 0, 0, 0, 0, 0),
}

SM_QN_NAMES = ['Q', 'B', 'L_e', 'L_mu', 'L_tau', 'S', 'C', "B'", 'T']


def dimensional_counting_prediction():
    """Prediction P9: expected d_eff for different interaction regimes.

    Returns
    -------
    predictions : dict mapping regime → predicted d_eff
    """
    return {
        'strong_vertex': {
            'conserved_qn': 9,  # all 9 additive QNs
            'continuous': 2,     # mass, width
            'spin': 1,
            'd_eff': 12,
            'explanation': 'All QNs conserved at strong/EM vertices',
        },
        'weak_vertex': {
            'conserved_qn': 5,  # Q, B, Le, Lμ, Lτ (at vertex level)
            'continuous': 2,
            'spin': 1,
            'd_eff': 8,
            'explanation': 'Lepton flavors conserved at vertex, S/C/B\'/T violated',
        },
        'weak_propagation': {
            'conserved_qn': 3,  # Q, B, L (total only, due to oscillations)
            'continuous': 2,
            'spin': 1,
            'd_eff': 6,
            'explanation': 'PMNS mixing reduces conserved charges over propagation',
        },
    }


def parity_check_matrix(interaction='strong'):
    """Construct the parity-check matrix H for conservation laws.

    H·Δq = 0 at each vertex.

    Parameters
    ----------
    interaction : str — 'strong', 'em', 'weak_vertex', 'weak_propagation'

    Returns
    -------
    H : ndarray — parity check matrix
    rank : int — rank(H)
    conserved_names : list of str — which QNs are conserved
    """
    # QN order: Q, B, Le, Lμ, Lτ, S, C, B', T
    if interaction in ('strong', 'em'):
        # All 9 conserved
        H = np.eye(9)
        rank = 9
        conserved = SM_QN_NAMES

    elif interaction == 'weak_vertex':
        # Q, B, Le, Lμ, Lτ conserved (individual lepton flavors at vertex)
        # S, C, B', T can change by ±1
        H = np.zeros((5, 9))
        H[0, 0] = 1  # Q
        H[1, 1] = 1  # B
        H[2, 2] = 1  # Le
        H[3, 3] = 1  # Lμ
        H[4, 4] = 1  # Lτ
        rank = 5
        conserved = ['Q', 'B', 'L_e', 'L_mu', 'L_tau']

    elif interaction == 'weak_propagation':
        # Q, B, L (total) conserved
        H = np.zeros((3, 9))
        H[0, 0] = 1          # Q
        H[1, 1] = 1          # B
        H[2, 2] = 1; H[2, 3] = 1; H[2, 4] = 1  # L = Le + Lμ + Lτ
        rank = 3
        conserved = ['Q', 'B', 'L']

    else:
        raise ValueError(f"Unknown interaction: {interaction}")

    return H, rank, conserved


def code_rate(interaction='strong'):
    """Compute code rate R = (n - rank(H)) / n.

    R = 0 means fully constrained (strong); R > 0 means freedom (weak).
    """
    _, rank, _ = parity_check_matrix(interaction)
    n = 9  # total QN dimensions
    return (n - rank) / n
