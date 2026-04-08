"""
ml_receiver.py — Neural Network as CDMA Multiuser Receiver
============================================================

Maps the CDMA receiver hierarchy onto particle physics ML:

  CDMA receiver type          Physics equivalent          Performance bound
  ─────────────────────────   ─────────────────────────   ─────────────────
  Matched filter              1D histogram template fit    CRB(1D)
  Decorrelating detector      Pseudoinverse template fit   CRB(1D), unbiased
  MMSE linear receiver        Regularized template fit     ≤ CRB(1D)
  NN single-user detector     Mass-only NN classifier      ~ CRB(1D)
  NN MMSE multiuser receiver  Mass+features NN              → CRB(multi-D)
  Optimal receiver (INFERNO)  Fisher-trained NN            → CRB(multi-D)*

  * INFERNO targets Fisher-optimal summaries; proven to approach CRB asymptotically
    in nuisance-free settings. With nuisance parameters, its advantage lies in
    profiled Fisher optimization. General optimality not proven.

Key insight: the NN does NOT beat the CRB. What changes is WHICH CRB:
going from 1D to multi-D lowers the bound because the FIM gains rank.
The NN is the practical tool to approach the tighter bound.

Predicted maximum gain per channel:
  gain_k = CRB_1D(k) / CRB_multiD(k) = eta_multiD(k) / eta_1D(k)

This is the CDMA near-far resistance improvement from multiuser detection.

References:
  INFERNO (de Castro & Dorigo, Comput.Phys.Commun. 244, 2019)
  IMNNs (Charnock et al., Phys.Rev.D 97, 2018)
  Fisher info flow in ANNs (Weimar et al., Phys.Rev.X 2025)
  MadMiner (Brehmer et al., Eur.Phys.J.C 80, 2020)
  Verdú, Multiuser Detection (Cambridge, 1998)
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize


def simulate_events(theta, N, sigma_det=15.0, features='multi', seed=42):
    """Simulate individual tau decay events with v0.5.0 visible-mass physics.

    Single-particle channels (e, mu, pi) produce delta-function masses
    at rest mass, smeared by detector resolution. Hadronic channels
    produce Breit-Wigner distributed masses with physical thresholds.

    Parameters
    ----------
    theta : (K,) branching ratios
    N : number of events
    sigma_det : detector resolution in MeV
    features : 'mass_only' or 'multi'
    
    Returns
    -------
    X, y_true, feature_names
    """
    from tau_cdma.tau.templates import (M_E, M_MU, M_PI, M_RHO, G_RHO,
                                         M_A1, G_A1, M_PI2PI0, G_PI2PI0,
                                         M_OTHER, G_OTHER, THRESH_RHO,
                                         THRESH_A1, THRESH_PI2PI0, THRESH_OTHER)
    rng = np.random.default_rng(seed)
    K = len(theta)
    N = int(N)
    y_true = rng.choice(K, size=N, p=theta)

    # Channel properties for visible mass generation
    props = [
        dict(mass=M_E,      width=0,        threshold=0,             ntk=1, eid=0.90, muid=0.01, pi0=0.00),
        dict(mass=M_MU,     width=0,        threshold=0,             ntk=1, eid=0.005,muid=0.95, pi0=0.00),
        dict(mass=M_PI,     width=0,        threshold=0,             ntk=1, eid=0.02, muid=0.01, pi0=0.05),
        dict(mass=M_RHO,    width=G_RHO,    threshold=THRESH_RHO,    ntk=2, eid=0.02, muid=0.01, pi0=0.70),
        dict(mass=M_A1,     width=G_A1,     threshold=THRESH_A1,     ntk=3, eid=0.02, muid=0.01, pi0=0.05),
        dict(mass=M_PI2PI0, width=G_PI2PI0, threshold=THRESH_PI2PI0, ntk=1, eid=0.02, muid=0.01, pi0=0.70),
        dict(mass=M_OTHER,  width=G_OTHER,  threshold=THRESH_OTHER,  ntk=2, eid=0.02, muid=0.01, pi0=0.10),
    ]

    m_vis = np.zeros(N)
    for k in range(K):
        mask = y_true == k
        n_k = np.sum(mask)
        if n_k == 0:
            continue
        p = props[k]
        if p['width'] == 0:
            # Single-particle: delta at rest mass
            m_vis[mask] = p['mass'] * np.ones(n_k)
        else:
            # Hadronic: Cauchy (Breit-Wigner) with threshold rejection
            samples = np.empty(n_k)
            generated = 0
            while generated < n_k:
                batch = n_k - generated
                candidates = rng.standard_cauchy(batch * 2) * p['width'] / 2 + p['mass']
                valid = candidates[candidates >= p['threshold']]
                take = min(len(valid), batch)
                samples[generated:generated + take] = valid[:take]
                generated += take
            m_vis[mask] = samples

    m_vis = np.clip(m_vis, 0, 2000)
    m_vis += rng.normal(0, sigma_det, size=N)
    m_vis = np.clip(m_vis, 0, 2000)

    if features == 'mass_only':
        return m_vis.reshape(-1, 1), y_true, ['m_vis']

    ntk = np.zeros(N)
    eid = np.zeros(N)
    muid = np.zeros(N)
    pi0 = np.zeros(N)
    for k in range(K):
        mask = y_true == k
        n_k = np.sum(mask)
        if n_k == 0:
            continue
        p = props[k]
        ntk[mask] = np.maximum(1, rng.poisson(p['ntk'], n_k))
        eid[mask] = (rng.random(n_k) < p['eid']).astype(float)
        muid[mask] = (rng.random(n_k) < p['muid']).astype(float)
        pi0[mask] = (rng.random(n_k) < p['pi0']).astype(float)

    X = np.column_stack([m_vis, ntk, eid, muid, pi0])
    return X, y_true, ['m_vis', 'n_tracks', 'is_e', 'is_mu', 'has_pi0']


def run_ml_layer(bench, verbose=True, quick=False):
    """Layer 4: NN as CDMA multiuser receiver.
    
    Demonstrates the full CDMA receiver hierarchy:
    (a) eta predicts mass-only NN channel-level performance  
    (b) Multi-feature NN resolves μ-π confusion (targeted fusion)
    (c) Predicted gain ceiling from CRB ratio matches observed NN improvement
    (d) CDMA receiver taxonomy: matched filter → decorrelator → MMSE → NN
    """
    from tau_cdma.core.fisher import poisson_fim, crb as compute_crb
    from tau_cdma.core.interference import interference_matrix, multiuser_efficiency

    theta = bench['theta']
    N = bench['N']
    K = len(theta)
    labels = bench['templates'].short_labels
    eta_1d = bench['eta']
    A = bench['A']
    bg = bench['background']
    M = A.shape[0]

    N_train = 100_000 if not quick else 30_000
    N_test = 50_000 if not quick else 15_000
    n_est_trials = 50 if not quick else 15
    N_est = 50_000 if not quick else 10_000

    if verbose:
        print(f"\n  --- Layer 4: NN as CDMA Multiuser Receiver ---")

    # ==== Compute theoretical CRBs for 1D and multi-D ====
    F_1d = poisson_fim(A, theta, N_est, bg)
    crb_1d = compute_crb(F_1d, regularize=True)

    # Multi-D CRB (with PID features — proper joint product-space model)
    pid_block = np.zeros((3, K))
    pid_block[0, 0] = 1.0   # e -> PID=e
    pid_block[1, 1] = 1.0   # mu -> PID=mu
    pid_block[2, 2:] = 1.0  # hadrons -> PID=had
    for k in range(K):
        s = pid_block[:, k].sum()
        if s > 0:
            pid_block[:, k] /= s
    # Joint template: mass x PID (product space)
    M_joint = M * 3
    A_pid = np.zeros((M_joint, K))
    for k in range(K):
        A_pid[:, k] = np.outer(A[:, k], pid_block[:, k]).flatten()
    for k in range(K):
        s = A_pid[:, k].sum()
        if s > 0:
            A_pid[:, k] /= s
    bg_aug = np.full(M_joint, 1e-6)

    F_pid = poisson_fim(A_pid, theta, N_est, bg_aug)
    crb_pid = compute_crb(F_pid, regularize=True)

    # Predicted gain ceiling per channel: sqrt(CRB_1D / CRB_5D)
    predicted_gain = np.zeros(K)
    for k in range(K):
        s1 = crb_1d[k] if np.isfinite(crb_1d[k]) and crb_1d[k] > 0 else np.inf
        sp = crb_pid[k] if np.isfinite(crb_pid[k]) and crb_pid[k] > 0 else 1e-30
        if np.isfinite(s1):
            predicted_gain[k] = np.sqrt(s1 / sp)
        else:
            predicted_gain[k] = np.inf

    # ==== Generate and split data ====
    X_1d, y, _ = simulate_events(theta, N_train + N_test,
                                  features='mass_only', seed=42)
    X_multi, _, fnames = simulate_events(theta, N_train + N_test,
                                          features='multi', seed=42)
    idx = np.arange(len(y))
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    tr, te = idx[:N_train], idx[N_train:N_train + N_test]

    # ==== Train classifiers ====
    clf_1d = MLPClassifier((32, 16), max_iter=500, random_state=42,
                           early_stopping=True, validation_fraction=0.1)
    clf_1d.fit(X_1d[tr], y[tr])

    clf_m = MLPClassifier((64, 32), max_iter=500, random_state=42,
                          early_stopping=True, validation_fraction=0.1)
    clf_m.fit(X_multi[tr], y[tr])

    # ==== (a) Classification vs eta ====
    pred_1d = clf_1d.predict(X_1d[te])
    pred_m = clf_m.predict(X_multi[te])
    acc_1d = np.mean(pred_1d == y[te])
    acc_m = np.mean(pred_m == y[te])

    C_1d = confusion_matrix(y[te], pred_1d, labels=range(K))
    C_m = confusion_matrix(y[te], pred_m, labels=range(K))
    pca_1d = np.diag(C_1d) / np.maximum(C_1d.sum(axis=1), 1)
    pca_m = np.diag(C_m) / np.maximum(C_m.sum(axis=1), 1)

    # e-mu confusion rates
    C_1d_norm = C_1d / np.maximum(C_1d.sum(axis=1, keepdims=True), 1)
    C_m_norm = C_m / np.maximum(C_m.sum(axis=1, keepdims=True), 1)

    mupi_confusion_1d = C_1d_norm[2, 1] + C_1d_norm[1, 2]  # μ-π near-far
    mupi_confusion_5d = C_m_norm[2, 1] + C_m_norm[1, 2]

    if verbose:
        print(f"\n    (a) Classification performance:")
        print(f"        Overall: mass-only NN = {acc_1d:.3f}, mass+features NN = {acc_m:.3f}")
        print(f"        {'Ch':>6s} {'NN(m)':>7s} {'NN(m+)':>7s} {'η_1D':>7s}"
              f" {'Interpretation':>35s}")
        for k in range(K):
            if eta_1d[k] < 0.05:
                interp = "η≈0 → 1D fails, multi-D rescues"
            elif eta_1d[k] > 0.5:
                interp = "η high → both succeed"
            else:
                interp = "η mid → multi-D improves"
            print(f"        {labels[k]:>6s} {pca_1d[k]:>7.3f} {pca_m[k]:>7.3f} "
                  f"{eta_1d[k]:>7.4f} {interp:>35s}")

        print(f"\n        μ-π cross-confusion (near-far effect):")
        print(f"          mass-only: P(μ|π)={C_1d_norm[2,1]:.3f}, P(π|μ)={C_1d_norm[1,2]:.3f}")
        print(f"          mass+feat: P(μ|π)={C_m_norm[2,1]:.3f}, P(π|μ)={C_m_norm[1,2]:.3f}")

    eta_corr = np.corrcoef(eta_1d, pca_1d)[0, 1]

    # Bayes-optimal ceiling (Shannon)
    from tau_cdma.core.shannon import bayes_confusion as compute_bayes
    bc_1d = compute_bayes(A, theta)
    bayes_acc = bc_1d['accuracy']

    if verbose:
        print()
        print(f"        Bayes-optimal ceiling (theoretical max for 1D):")
        print(f"        {'Ch':>6s} {'Bayes':>7s} {'NN(m)':>7s} {'NN(m+)':>7s} {'Gap':>12s}")
        for k in range(K):
            gap_str = ""
            if bayes_acc[k] < 0.01:
                gap_str = "ceiling = 0%"
            elif pca_1d[k] < bayes_acc[k] - 0.05:
                gap_str = f"NN underperforms"
            else:
                gap_str = f"near ceiling"
            print(f"        {labels[k]:>6s} {bayes_acc[k]:>7.3f} {pca_1d[k]:>7.3f} "
                  f"{pca_m[k]:>7.3f} {gap_str:>12s}")
        print(f"        {'Total':>6s} {bc_1d['overall']:>7.3f} {acc_1d:>7.3f} {acc_m:>7.3f}")
        print(f"\n        corr(η, NN_1D_accuracy) = {eta_corr:.2f}")
        print(f"        corr(η, Bayes_accuracy) = "
              f"{np.corrcoef(eta_1d, bayes_acc)[0,1]:.2f}")

    bayes_eta_corr = np.corrcoef(eta_1d, bayes_acc)[0, 1]

    # ==== (b) BR estimation: NN counting vs histogram fit ====
    br_hist = np.zeros((n_est_trials, K))
    br_nn = np.zeros((n_est_trials, K))

    for trial in range(n_est_trials):
        X_trial, y_trial, _ = simulate_events(
            theta, N_est, features='multi', seed=1000 + trial)

        pred = clf_m.predict(X_trial)
        for k in range(K):
            br_nn[trial, k] = np.sum(pred == k) / N_est

        hist, _ = np.histogram(X_trial[:, 0], bins=M, range=(0, 1800))

        def nll(t, h=hist):
            t = np.abs(t)
            lam = N_est * (A @ t) + bg
            lam = np.maximum(lam, 1e-30)
            return np.sum(lam - h * np.log(lam))

        res = minimize(nll, theta, method='L-BFGS-B',
                       bounds=[(1e-6, 1)] * K)
        br_hist[trial] = np.abs(res.x)

    var_h = np.var(br_hist, axis=0)
    var_n = np.var(br_nn, axis=0)
    bias_h = np.mean(br_hist, axis=0) - theta
    bias_n = np.mean(br_nn, axis=0) - theta
    mse_h = var_h + bias_h ** 2
    mse_n = var_n + bias_n ** 2

    observed_gain = np.zeros(K)
    for k in range(K):
        if mse_n[k] > 0:
            observed_gain[k] = np.sqrt(mse_h[k] / mse_n[k])
        else:
            observed_gain[k] = np.inf

    if verbose:
        print(f"\n    (b) BR estimation ({n_est_trials} MC trials, N={N_est:.0e}):")
        print(f"        {'Ch':>6s} {'θ':>7s} {'σ_hist':>7s} {'σ_NN':>7s} "
              f"{'σ_CRB1D':>8s} {'σ_CRB5D':>8s} {'Pred':>6s} {'Obs':>6s}")
        for k in range(K):
            sh = np.sqrt(var_h[k])
            sn = np.sqrt(var_n[k])
            sc1 = np.sqrt(crb_1d[k]) if np.isfinite(crb_1d[k]) and crb_1d[k] > 0 else np.inf
            sc5 = np.sqrt(crb_pid[k]) if np.isfinite(crb_pid[k]) and crb_pid[k] > 0 else np.inf
            pg = f"{predicted_gain[k]:.1f}×" if np.isfinite(predicted_gain[k]) and predicted_gain[k] < 1000 else "∞"
            og = f"{observed_gain[k]:.1f}×"
            sc1s = f"{sc1:.5f}" if np.isfinite(sc1) else "∞"
            sc5s = f"{sc5:.5f}" if np.isfinite(sc5) else "∞"
            print(f"        {labels[k]:>6s} {theta[k]:>7.4f} {sh:>7.4f} {sn:>7.4f} "
                  f"{sc1s:>8s} {sc5s:>8s} {pg:>6s} {og:>6s}")

    # ==== (c) CDMA receiver taxonomy ====
    if verbose:
        print(f"\n    (c) CDMA Receiver Taxonomy:")
        print(f"    ┌────────────────────────────┬────────────────────────────┬──────────────┐")
        print(f"    │ CDMA Receiver               │ Physics Equivalent         │ Bound        │")
        print(f"    ├────────────────────────────┼────────────────────────────┼──────────────┤")
        print(f"    │ Matched filter              │ 1D histogram fit           │ CRB(1D)      │")
        print(f"    │ Decorrelating detector      │ R⁻¹ template fit           │ CRB(1D)      │")
        print(f"    │ MMSE linear receiver        │ Regularized template fit   │ ≤ CRB(1D)    │")
        print(f"    │ NN single-user detector     │ Mass-only NN classifier    │ ~ CRB(1D)    │")
        print(f"    │ NN multiuser receiver       │ Mass+features NN           │ → CRB(m+PID) │")
        print(f"    │ Fisher-opt (INFERNO/IMNN)   │ Fisher-trained NN          │ → CRB(5D)*   │")
        print(f"    └────────────────────────────┴────────────────────────────┴──────────────┘")
        print()
        print(f"    Qualitative interpretation:")
        print(f"      The histogram template fit is a MATCHED FILTER: it correlates")
        print(f"      the observed spectrum with each template (spreading code) and")
        print(f"      extracts branching ratios. It suffers from MAI — channels with")
        print(f"      overlapping hadronic templates interfere, inflating estimation variance.")
        print()
        print(f"      Adding detector features (PID, tracks, π⁰ count) is equivalent")
        print(f"      to WIDEBAND CDMA: more spreading dimensions resolve previously")
        print(f"      aliased users. The multi-feature NN acts as a nonlinear MMSE")
        print(f"      receiver that approaches the multi-dimensional CRB.")
        print()
        print(f"      INFERNO/IMNN methods explicitly train NNs to maximize Fisher")
        print(f"      information — they are FISHER-OPTIMAL SUMMARY CONSTRUCTORS.")
        print(f"      In nuisance-free settings, this aligns with Bayes-optimal")
        print(f"      classification. General optimality is not proven.")
        print(f"      The CRB ratio provides a heuristic scale for achievable gain:")
        print(f"        gain_k ~ σ_1D(k) / σ_5D(k) = √[CRB_1D / CRB_5D]")
        print()
        print(f"      The NN does NOT beat the CRB. What improves is WHICH CRB:")
        print(f"      going from 1D to multi-D lowers the bound because the FIM gains")
        print(f"      rank (aliased eigenvalues become non-degenerate). The NN is the")
        print(f"      practical tool to approach this tighter bound.")

    # ==== Checks (v0.5.0: hadronic confusion, not e-μ aliasing) ====
    checks = {
        'Mass+features NN > mass-only NN accuracy': float(acc_m) > float(acc_1d) + 0.02,
        'η predicts Bayes accuracy (r > 0.5)': float(bayes_eta_corr) > 0.5,
        'Bayes μ > 50% in 1D (no collapse)': float(bayes_acc[1]) > 0.50,
    }

    passed = all(checks.values())

    return {
        'acc_1d': acc_1d, 'acc_multi': acc_m,
        'pca_1d': pca_1d, 'pca_multi': pca_m,
        'bayes_accuracy': bayes_acc, 'bayes_overall': bc_1d['overall'],
        'eta_corr': eta_corr, 'bayes_eta_corr': bayes_eta_corr,
        'var_hist': var_h, 'var_nn': var_n,
        'mse_hist': mse_h, 'mse_nn': mse_n,
        'predicted_gain': predicted_gain, 'observed_gain': observed_gain,
        'crb_1d': crb_1d, 'crb_pid': crb_pid,
        'confusion_1d': C_1d_norm, 'confusion_multi': C_m_norm,
        'checks': checks, 'passed': passed,
    }
