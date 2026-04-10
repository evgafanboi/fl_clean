import numpy as np
from typing import List, Tuple, Optional


class RobustFilter:
    """
    Spectral robust mean estimation (Diakonikolas et al. 2017).

    Iteratively projects onto the top eigenvector of the empirical
    covariance and removes the most extreme contributor until the
    eigenvalue spectrum stabilises.

    epsilon: expected Byzantine fraction.  Controls both the spectral
             trigger (eigenvalue ratio > 1/epsilon) and the removal
             budget (at most ceil(epsilon * K) contributors removed).
    """

    def __init__(self, epsilon: float = 0.2, max_iter: int = 5, **_kw):
        self.epsilon = epsilon
        self.max_iter = max_iter

    def _max_removals(self, K: int) -> int:
        return max(1, int(np.ceil(self.epsilon * K)))

    def tail_bound(self, eigenvalues: np.ndarray, epsilon: float) -> bool:
        """Reference spectrum test.
        Returns True (= should filter) when the top eigenvalue dominates
        the rest beyond what epsilon-level corruption could cause.
        Only non-zero eigenvalues (effective rank) are considered so that
        rank-deficient covariance matrices (K < D) are handled correctly.
        """
        abs_e = np.abs(eigenvalues)
        effective = abs_e[abs_e > 1e-10]
        if len(effective) <= 1:
            return False
        sorted_e = np.sort(effective)
        ratio = sorted_e[-1] / max(sorted_e[:-1].mean(), 1e-10)
        return ratio > 1.0 / epsilon

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

        w = np.ones(n, dtype=np.float64) / n if weights is None \
            else np.asarray(weights, dtype=np.float64) / np.sum(weights)

        for _ in range(min(self.max_iter, self._max_removals(n))):
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
                v_star = evecs[:, np.argmax(np.abs(eigs))]

            if not self.tail_bound(eigs, self.epsilon):
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

        w = np.ones(n, dtype=np.float64) / n if weights is None \
            else np.asarray(weights, dtype=np.float64) / np.sum(weights)

        g_max_eig = None
        g_max_ratio = None

        for _ in range(min(self.max_iter, self._max_removals(n))):
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
                v_star = evecs[:, np.argmax(np.abs(eigs))]

            abs_e = np.abs(eigs)
            max_eig = float(abs_e.max())
            effective = abs_e[abs_e > 1e-10]
            sorted_e = np.sort(effective) if len(effective) > 1 else effective
            ratio = float(sorted_e[-1] / max(sorted_e[:-1].mean(), 1e-10)) \
                if len(sorted_e) > 1 else 0.0

            if g_max_eig is None or max_eig > g_max_eig:
                g_max_eig = max_eig
            if g_max_ratio is None or ratio > g_max_ratio:
                g_max_ratio = ratio

            if ratio <= 1.0 / self.epsilon:
                break

            worst = np.argmax(np.abs(centered @ v_star))
            active[np.where(active)[0][worst]] = False

        aw = w[active] / w[active].sum()
        mean = np.average(S[active], axis=0, weights=aw)
        removed = sorted(np.where(~active)[0].tolist())
        return mean, g_max_eig, removed, g_max_ratio

    # ── vectorised batch path (one-shot quantile removal) ────────────────

    def compute_robust_mean_batch(self, S_batch: np.ndarray) -> tuple:
        """
        Vectorised batch robust mean for logit consensus.
        S_batch: (N, K, D).
        Returns (means, max_eigenvalue, removal_counts, max_ratio).
        """
        N, K, D = S_batch.shape
        mu_S = S_batch.mean(axis=1)
        centered = S_batch - mu_S[:, np.newaxis, :]
        Sigma = np.einsum('nkd,nke->nde', centered, centered) / (K - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        abs_eigs = np.abs(eigenvalues)

        # spectral trigger: eigenvalue ratio using effective rank only
        effective_rank = min(K - 1, D)
        top_eigs = np.sort(abs_eigs, axis=1)[:, -effective_rank:]
        max_eigs = top_eigs[:, -1]
        rest_mean = top_eigs[:, :-1].mean(axis=1) if effective_rank > 1 else max_eigs
        ratios = max_eigs / np.maximum(rest_mean, 1e-10)
        below_thresh = ratios <= (1.0 / self.epsilon)

        # top eigenvector & projections
        n_idx = np.arange(N)
        max_eig_idx = np.argmax(abs_eigs, axis=1)
        v_star = eigenvectors[n_idx, :, max_eig_idx]
        abs_proj = np.abs(np.einsum('nkd,nd->nk', centered, v_star))

        # quantile removal: keep K - ceil(eps*K) lowest-projection clients
        n_keep = K - self._max_removals(K)
        ranks = np.argsort(np.argsort(abs_proj, axis=1), axis=1)
        filt_mask = ranks < n_keep

        mask_sum = filt_mask.sum(axis=1, keepdims=True).astype(np.float64)
        has_valid = mask_sum[:, 0] > 0
        filt_float = filt_mask.astype(np.float64)
        filtered_means = (np.einsum('nkd,nk->nd', S_batch, filt_float)
                          / np.maximum(mask_sum, 1.0))

        use_filtered = (~below_thresh) & has_valid
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
        super().__init__(epsilon=epsilon, max_iter=5)

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
        super().__init__(epsilon=epsilon, max_iter=3)

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
