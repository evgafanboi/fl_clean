import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CRONUS_THRESHOLD = 9.0

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def make_logits(n_clients, n_samples, C, rng):
    # Simulate label-skew: each client has 1–4 dominant classes (capped by C)
    logits = rng.standard_normal((n_clients, n_samples, C)).astype(np.float32) * 0.5
    for k in range(n_clients):
        n_dominant = min(C, rng.randint(1, 5))
        dominant = rng.choice(C, size=n_dominant, replace=False)
        logits[k, :, dominant] += rng.uniform(2.0, 5.0, size=(n_dominant, n_samples))
    return logits


def eig_stats(logits):
    # logits: (n_clients, n_samples, C) → per-sample covariance over clients
    # S[s] shape: (n_clients, C) → covariance (C, C)
    n_clients, n_samples, C = logits.shape
    S = logits.transpose(1, 0, 2)  # (n_samples, n_clients, C)
    mu = S.mean(axis=1, keepdims=True)
    centered = S - mu
    Sigma = np.einsum('nkd,nke->nde', centered, centered) / (n_clients - 1)
    eigvals = np.linalg.eigvalsh(Sigma)  # (n_samples, C)
    max_eigs = eigvals[:, -1]
    return max_eigs, eigvals



def run_dim_sweep(dims, n_clients, n_samples, n_seeds, threshold):
    rng = np.random.RandomState(0)
    results = {d: {"raw": [], "soft": [], "fpr_raw": [], "fpr_soft": []} for d in dims}

    for C in dims:
        for _ in range(n_seeds):
            logits = make_logits(n_clients, n_samples, C, rng)
            probs = softmax(logits)
            max_raw, _ = eig_stats(logits)
            max_soft, _ = eig_stats(probs)
            results[C]["raw"].append(max_raw.mean())
            results[C]["soft"].append(max_soft.mean())
            results[C]["fpr_raw"].append(float((max_raw > threshold).mean()))
            results[C]["fpr_soft"].append(float((max_soft > threshold).mean()))

    mean_raw  = np.array([np.mean(results[d]["raw"])      for d in dims])
    std_raw   = np.array([np.std(results[d]["raw"])       for d in dims])
    mean_soft = np.array([np.mean(results[d]["soft"])     for d in dims])
    std_soft  = np.array([np.std(results[d]["soft"])      for d in dims])
    fpr_raw   = np.array([np.mean(results[d]["fpr_raw"])  for d in dims])
    fpr_soft  = np.array([np.mean(results[d]["fpr_soft"]) for d in dims])

    return mean_raw, std_raw, mean_soft, std_soft, fpr_raw, fpr_soft


def run_k_sweep(k_values, C, n_samples, n_seeds, threshold):
    rng = np.random.RandomState(55)
    fpr_soft = []
    for K in k_values:
        fprs = []
        for _ in range(n_seeds):
            logits = make_logits(K, n_samples, C, rng)
            max_soft, _ = eig_stats(softmax(logits))
            fprs.append(float((max_soft > threshold).mean()))
        fpr_soft.append(np.mean(fprs))
    return np.array(fpr_soft)


def run_single(C, n_clients, n_samples, rng):
    logits = make_logits(n_clients, n_samples, C, rng)
    probs = softmax(logits)
    max_eigs_raw, eigvals_raw = eig_stats(logits)
    max_eigs_soft, eigvals_soft = eig_stats(probs)
    return {
        "max_raw": max_eigs_raw,
        "max_soft": max_eigs_soft,
    }


def false_positive_rate(max_eigs, threshold):
    return float((max_eigs > threshold).mean())


def null_expected_max_eig_raw(n_clients, C, sigma2=1.0):
    # For a K×C matrix of i.i.d. N(0,σ²) rows (raw logits assumption),
    # the largest eigenvalue of the sample covariance grows as σ²(1+√(C/K))².
    # ONLY valid for raw logit inputs — NOT for softmax probabilities.
    gamma = C / n_clients
    return sigma2 * (1 + np.sqrt(gamma)) ** 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=int, default=34)
    parser.add_argument("--n_clients", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--out", type=str, default="results/plots/eigenvalue_plot/cronus_dim_dependence.png")
    args = parser.parse_args()

    rng = np.random.RandomState(42)
    dims = [2, 4, 8, 16, 34, 64, 128, 256]

    # ── Theoretical null baseline ──────────────────────────────────────────────
    # The null λ_max is the largest eigenvalue you would get from a matrix of
    # PURE NOISE (no attack, no signal). If this already exceeds Cronus's threshold,
    # the filter fires on every benign sample — dimension alone triggers it.
    print(f"=== Cronus hardcoded threshold = {CRONUS_THRESHOLD} ===\n")
    print("NOTE: Cronus filters on SOFTMAX PROBABILITIES (model.predict output).")
    print("Raw-logit null baseline (σ²=1 Gaussian) shown for reference only.\n")
    print(f"{'C':>6}  {'raw null λ_max':>16}  {'vs threshold':>14}")
    for C in dims:
        null = null_expected_max_eig_raw(args.n_clients, C)
        verdict = "BELOW" if null < CRONUS_THRESHOLD else "ABOVE threshold even for pure noise!"
        print(f"  C={C:4d}: raw null λ_max = {null:.4f}  → {verdict}")

    mean_raw, std_raw, mean_soft, std_soft, fpr_raw, fpr_soft = run_dim_sweep(
        dims, args.n_clients, args.n_samples, args.n_seeds, CRONUS_THRESHOLD
    )

    print(f"\nEmpirical mean max eigenvalue (benign only, {args.n_seeds} seeds, {args.n_samples} samples):")
    print(f"{'C':>6}  {'softmax λ_max (Cronus)':>22}  {'FPR soft':>9}  {'raw λ_max (ref)':>16}  {'FPR raw':>8}")
    for i, C in enumerate(dims):
        print(f"  C={C:4d}  {mean_soft[i]:>22.4f}  {fpr_soft[i]:>9.3f}  {mean_raw[i]:>16.4f}  {fpr_raw[i]:>8.3f}")

    stats = run_single(args.C, args.n_clients, args.n_samples, rng)
    fpr_s = false_positive_rate(stats['max_soft'], CRONUS_THRESHOLD)
    fpr_r = false_positive_rate(stats['max_raw'], CRONUS_THRESHOLD)
    print(f"\n  FPR softmax at C={args.C} (threshold={CRONUS_THRESHOLD}): {fpr_s:.4f}  ← Cronus actual input")
    print(f"  FPR raw     at C={args.C} (threshold={CRONUS_THRESHOLD}): {fpr_r:.4f}  ← raw logits, reference only")

    n_client_sweep = [10, 20, 50, 100, 200, 500]
    fpr_k_soft = run_k_sweep(n_client_sweep, args.C, args.n_samples, args.n_seeds, CRONUS_THRESHOLD)

    null_curve_raw = [null_expected_max_eig_raw(args.n_clients, C) for C in dims]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(
        f"Cronus threshold={CRONUS_THRESHOLD} is dimension-dependent  (K={args.n_clients} clients, label-skew simulation)",
        fontsize=12, fontweight="bold"
    )

    # Panel 1: softmax probs λ_max vs C — CRONUS ACTUAL INPUT
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(dims, mean_soft, "g-o", label="Empirical λ_max (benign clients)")
    ax1.fill_between(dims, mean_soft - std_soft, mean_soft + std_soft, alpha=0.2, color="g")
    ax1.axhline(CRONUS_THRESHOLD, color="r", linestyle=":", linewidth=2, label=f"Cronus threshold={CRONUS_THRESHOLD}")
    ax1.set_xlabel("Number of classes C")
    ax1.set_ylabel("Max eigenvalue λ_max")
    ax1.set_title("Softmax probs: λ_max grows with C\n(Cronus actual input — all clients benign)")
    ax1.legend(fontsize=8)
    ax1.set_xscale("log", base=2)

    # Panel 2: raw logits λ_max vs C — reference only
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(dims, mean_raw, "b-o", label="Empirical λ_max (benign clients)")
    ax2.fill_between(dims, mean_raw - std_raw, mean_raw + std_raw, alpha=0.2, color="b")
    ax2.plot(dims, null_curve_raw, "b--", alpha=0.6, label="Null baseline (σ²=1 Gaussian)")
    ax2.axhline(CRONUS_THRESHOLD, color="r", linestyle=":", linewidth=2, label=f"Cronus threshold={CRONUS_THRESHOLD}")
    ax2.set_xlabel("Number of classes C")
    ax2.set_ylabel("Max eigenvalue λ_max")
    ax2.set_title("Raw logits: λ_max vs C\n(reference only — NOT Cronus input)")
    ax2.legend(fontsize=8)
    ax2.set_xscale("log", base=2)

    # Panel 3: FPR vs C — the key result (softmax = Cronus actual)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(dims, fpr_soft, "g-s", label="Softmax (Cronus actual input)", linewidth=2)
    ax3.plot(dims, fpr_raw, "b-o", label="Raw logits (reference)", alpha=0.5)
    ax3.axvline(args.C, color="gray", linestyle="--", alpha=0.7, label=f"CIC-23 C={args.C}")
    ax3.set_xlabel("Number of classes C")
    ax3.set_ylabel("False positive rate on benign clients")
    ax3.set_title(
        f"Fraction of BENIGN samples incorrectly\nflagged as Byzantine (threshold={CRONUS_THRESHOLD})"
    )
    ax3.legend()
    ax3.set_xscale("log", base=2)
    ax3.set_ylim(-0.02, 1.02)

    # Panel 4: FPR vs K at fixed C — softmax FPR (Cronus actual)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(n_client_sweep, fpr_k_soft, "g-s", linewidth=2, label=f"FPR softmax (C={args.C})")
    ax4.axhline(CRONUS_THRESHOLD * 0, color="r", linestyle=":", linewidth=1.5, alpha=0)  # spacer
    ax4.set_xlabel("Number of clients K")
    ax4.set_ylabel("FPR on benign clients (softmax)")
    ax4.set_title(f"More clients → lower λ_max, lower FPR\n(C={args.C} fixed, Cronus threshold={CRONUS_THRESHOLD})")
    ax4.set_ylim(-0.02, 1.02)
    ax4.legend(fontsize=8)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {args.out}")


if __name__ == "__main__":
    main()

