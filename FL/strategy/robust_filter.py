import numpy as np
from typing import List, Tuple, Optional


class RobustFilter:
    """
    Spectral robust mean estimation.

    Ref 1 (theory + MATLAB): Diakonikolas et al., "Being Robust (in High
        Dimensions) Can Be Practical", ICML 2017.
        https://github.com/hoonose/robust-filter
    Ref 2 (Python / FL):     Zhu et al., "Byzantine-Robust Federated Learning
        with Optimal Statistical Rates", 2022.  filterL2_ in
        https://github.com/wanglun1996/secure-robust-federated-learning

    Iteratively projects onto the top eigenvector of the empirical
    covariance and removes the most extreme contributor until the
    top eigenvalue falls within the expected range.

    Parameters
    ----------
    epsilon : float
        Assumed Byzantine fraction.  Higher epsilon = MORE lenient
        (the filter tolerates more variance before triggering).
        Must be in (0, 0.5); values >= 0.5 disable filtering.
    expansion : float
        Tolerance multiplier for the spectral stop condition.
        Ref 2 uses 20.0.  Combined with the auto-estimated sigma
        and the eps-dependent slack, this sets the threshold:
            max_eig  <=  expansion * sigma * (1 + eps * log(1/eps))
        where sigma is the median positive eigenvalue (auto).
    """

    def __init__(self, epsilon: float = 0.2, expansion: float = 20.0, **_kw):
        self.epsilon = epsilon
        self.expansion = expansion

    def _max_removals(self, K: int) -> int:
        if self.epsilon >= 0.5:
            return 0
        return max(1, int(np.ceil(self.epsilon * K)))

    def _threshold(self, eigenvalues: np.ndarray) -> float:
        """Spectral stop threshold, auto-scaled to the data.

        sigma  = median of positive eigenvalues (robust scale estimate).
        slack  = 1 + eps * ln(1/eps)      (from Ref 1, Theorem 1.2).

        Higher epsilon  →  larger slack  →  higher threshold  →  more lenient.
        """
        if self.epsilon >= 0.5:
            return float('inf')
        pos = eigenvalues[eigenvalues > 1e-10]
        sigma = float(np.median(pos)) if len(pos) > 0 else 1.0
        slack = 1.0 + self.epsilon * np.log(1.0 / self.epsilon)
        return self.expansion * sigma * slack

    # ── iterative scalar path ────────────────────────────────────────────

    def compute_robust_mean(
        self,
        samples: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        if len(samples) <= 1:
            return samples[0]

        n = len(samples)
        d = samples[0].shape[0]
        S = np.vstack(samples)
        active = np.ones(n, dtype=bool)
        budget = self._max_removals(n)

        w = np.ones(n, dtype=np.float64) / n if weights is None \
            else np.asarray(weights, dtype=np.float64) / np.sum(weights)

        for _ in range(budget):
            if active.sum() <= 1:
                break
            aw = w[active] / w[active].sum()
            mu = np.average(S[active], axis=0, weights=aw)
            centered = S[active] - mu

            if d == 1:
                eigs = np.array([float(np.cov(centered.T, aweights=aw))])
                v_star = np.array([1.0])
            else:
                Sigma = np.cov(centered.T, aweights=aw)
                eigs, evecs = np.linalg.eigh(Sigma)
                v_star = evecs[:, -1]

            if eigs[-1] <= self._threshold(eigs):
                break

            worst = np.argmax(np.abs(centered @ v_star))
            active[np.where(active)[0][worst]] = False

        aw = w[active] / w[active].sum()
        return np.average(S[active], axis=0, weights=aw)

    def compute_robust_mean_debug(
        self,
        samples: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Optional[float], List[int], Optional[float]]:
        """Returns (robust_mean, max_eigenvalue, removed_indices, max_ratio)."""
        if len(samples) <= 1:
            return samples[0], None, [], None

        n = len(samples)
        d = samples[0].shape[0]
        S = np.vstack(samples)
        active = np.ones(n, dtype=bool)
        budget = self._max_removals(n)

        w = np.ones(n, dtype=np.float64) / n if weights is None \
            else np.asarray(weights, dtype=np.float64) / np.sum(weights)

        g_max_eig = None
        g_max_ratio = None

        for _ in range(budget):
            if active.sum() <= 1:
                break
            aw = w[active] / w[active].sum()
            mu = np.average(S[active], axis=0, weights=aw)
            centered = S[active] - mu

            if d == 1:
                eigs = np.array([float(np.cov(centered.T, aweights=aw))])
                v_star = np.array([1.0])
            else:
                Sigma = np.cov(centered.T, aweights=aw)
                eigs, evecs = np.linalg.eigh(Sigma)
                v_star = evecs[:, -1]

            max_eig = float(eigs[-1])
            pos = eigs[eigs > 1e-10]
            sigma = float(np.median(pos)) if len(pos) > 1 else max_eig
            ratio = max_eig / sigma if sigma > 1e-10 else 0.0

            if g_max_eig is None or max_eig > g_max_eig:
                g_max_eig = max_eig
            if g_max_ratio is None or ratio > g_max_ratio:
                g_max_ratio = ratio

            if max_eig <= self._threshold(eigs):
                break

            worst = np.argmax(np.abs(centered @ v_star))
            active[np.where(active)[0][worst]] = False

        aw = w[active] / w[active].sum()
        mean = np.average(S[active], axis=0, weights=aw)
        removed = sorted(np.where(~active)[0].tolist())
        return mean, g_max_eig, removed, g_max_ratio

    # ── vectorised batch path ────────────────────────────────────────────

    def compute_robust_mean_batch(self, S_batch: np.ndarray) -> tuple:
        """
        Vectorised batch robust mean for logit consensus.
        S_batch: (N, K, D).
        Returns (means, max_eigenvalue, removal_counts, max_ratio).
        """
        N, K, D = S_batch.shape
        mu_S = S_batch.mean(axis=1)

        if self.epsilon >= 0.5:
            return mu_S.astype(np.float32), None, np.zeros(K, dtype=np.int64), None

        centered = S_batch - mu_S[:, np.newaxis, :]
        Sigma = np.einsum('nkd,nke->nde', centered, centered) / (K - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        max_eigs = eigenvalues[:, -1]

        # auto sigma: median of positive eigenvalues per sample
        effective_rank = min(K - 1, D)
        top_eigs = np.sort(np.abs(eigenvalues), axis=1)[:, -effective_rank:]
        median_eigs = np.median(top_eigs, axis=1)
        slack = 1.0 + self.epsilon * np.log(1.0 / self.epsilon)
        thresholds = self.expansion * median_eigs * slack
        below_thresh = max_eigs <= thresholds

        # ratio for logging: max / median
        ratios = max_eigs / np.maximum(median_eigs, 1e-10)

        # top eigenvector & projections
        v_star = eigenvectors[:, :, -1]
        abs_proj = np.abs(np.einsum('nkd,nd->nk', centered, v_star))

        # remove single most-extreme client per triggered sample
        n_idx = np.arange(N)
        worst_client = np.argmax(abs_proj, axis=1)
        filt_mask = np.ones((N, K), dtype=bool)
        filt_mask[n_idx, worst_client] = False
        filt_mask[below_thresh] = True

        mask_sum = filt_mask.sum(axis=1, keepdims=True).astype(np.float64)
        filt_float = filt_mask.astype(np.float64)
        filtered_means = (np.einsum('nkd,nk->nd', S_batch, filt_float)
                          / np.maximum(mask_sum, 1.0))

        use_filtered = ~below_thresh
        means = np.where(use_filtered[:, np.newaxis], filtered_means, mu_S).astype(np.float32)

        removed_mask = (~filt_mask) & use_filtered[:, np.newaxis]
        removal_counts = removed_mask.sum(axis=0).astype(np.int64)

        above = ~below_thresh
        batch_max_eig = float(max_eigs[above].max()) if above.any() else None
        batch_max_ratio = float(ratios[above].max()) if above.any() else None

        return means, batch_max_eig, removal_counts, batch_max_ratio

    # ── convenience ──────────────────────────────────────────────────────

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        sample_sizes: Optional[List[int]] = None,
    ) -> np.ndarray:
        weights = None
        if sample_sizes is not None:
            weights = np.array(sample_sizes, dtype=np.float32)
            weights /= weights.sum()
        return self.compute_robust_mean(client_updates, weights)


class RobustFilterWeights(RobustFilter):
    def __init__(self, epsilon: float = 0.2, **_kw):
        super().__init__(epsilon=epsilon, expansion=20.0)

    def aggregate(
        self,
        client_weights: List[List[np.ndarray]],
        sample_sizes: List[int],
    ) -> List[np.ndarray]:
        n_layers = len(client_weights[0])
        aggregated = []
        for layer_idx in range(n_layers):
            layer_updates = [cw[layer_idx].flatten() for cw in client_weights]
            robust_mean = self.compute_robust_mean(layer_updates, sample_sizes)
            aggregated.append(robust_mean.reshape(client_weights[0][layer_idx].shape))
        return aggregated


class RobustFilterLogits(RobustFilter):
    def __init__(self, epsilon: float = 0.2, **_kw):
        super().__init__(epsilon=epsilon, expansion=20.0)

    def aggregate_per_class_logits(
        self,
        client_logits_list: List[np.ndarray],
        sample_sizes: List[int],
    ) -> np.ndarray:
        num_classes = client_logits_list[0].shape[0]
        aggregated = np.zeros_like(client_logits_list[0])
        for c in range(num_classes):
            aggregated[c, :] = self.compute_robust_mean(
                [cl[c, :] for cl in client_logits_list], sample_sizes,
            )
        return aggregated
