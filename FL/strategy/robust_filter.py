from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Tuple, Optional


FINITE_CAP = 1e6


def _finite_block(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values.astype(np.float64, copy=False), nan=0.0, posinf=FINITE_CAP, neginf=-FINITE_CAP)


def _adaptive_tail_block(S_block: np.ndarray, ref_hn: np.ndarray) -> Tuple[np.ndarray, float]:
    S_block = _finite_block(S_block)
    rows, K_act, _ = S_block.shape
    mu_S = S_block.mean(axis=1)
    centered = S_block - mu_S[:, np.newaxis, :]
    Sigma = np.einsum('nkd,nke->nde', centered, centered) / (K_act - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    max_eig = float(eigenvalues[:, -1].max())
    v_star = eigenvectors[:, :, -1]
    projections = np.einsum('nkd,nd->nk', centered, v_star)
    med_p = np.median(projections, axis=1, keepdims=True)
    sigma_r = np.maximum(
        np.median(np.abs(projections - med_p), axis=1, keepdims=True) * 1.4826,
        1e-10,
    )
    abs_z = np.abs(projections - med_p) / sigma_r
    sort_idx = np.argsort(abs_z, axis=1)
    sorted_z = np.take_along_axis(abs_z, sort_idx, axis=1)
    in_tail = sorted_z > ref_hn[np.newaxis, :]
    cum_and = np.flip(
        np.cumprod(np.flip(in_tail.astype(np.int8), axis=1), axis=1).astype(bool),
        axis=1,
    )
    is_byzantine = np.zeros((rows, K_act), dtype=np.int64)
    np.put_along_axis(is_byzantine, sort_idx, cum_and.astype(np.int64), axis=1)
    return is_byzantine.sum(axis=0), max_eig


def _cronus_block(S_block: np.ndarray, threshold: float) -> Tuple[np.ndarray, Optional[float]]:
    S_block = _finite_block(S_block)
    rows, K_act, _ = S_block.shape
    mu_S = S_block.mean(axis=1)
    centered = S_block - mu_S[:, np.newaxis, :]
    Sigma = np.einsum('nkd,nke->nde', centered, centered) / (K_act - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    max_eigs = eigenvalues[:, -1]
    above = max_eigs > threshold
    block_max = float(max_eigs[above].max()) if above.any() else None
    v_star = eigenvectors[:, :, -1]
    abs_proj = np.abs(np.einsum('nkd,nd->nk', centered, v_star))
    abs_proj[~above] = 0.0
    return abs_proj.sum(axis=0), block_max


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
        Assumed Byzantine fraction.  Controls threshold sensitivity only.
        Higher epsilon = more lenient threshold. Must be in (0, 0.5);
        values >= 0.5 disable filtering.
    budget : int
        Maximum number of clients to remove. Decoupled from epsilon.
        Early stopping prevents over-filtering when budget > actual outliers.
    """

    def __init__(self, epsilon: float = 0.2, budget: Optional[int] = None, **_kw):
        self.epsilon = epsilon
        self.budget = int(budget) if budget is not None else 0

    def _threshold(self, eigenvalues: np.ndarray) -> float:
        if self.epsilon >= 0.5:
            return float('inf')
        pos = eigenvalues[eigenvalues > 1e-10]
        sigma = float(np.median(pos)) if len(pos) > 0 else 1.0
        slack = 1.0 + self.epsilon * np.log(1.0 / self.epsilon)
        return sigma * slack

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

        for _ in range(self.budget):
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

        w = np.ones(n, dtype=np.float64) / n if weights is None \
            else np.asarray(weights, dtype=np.float64) / np.sum(weights)

        g_max_eig = None
        g_max_ratio = None

        for _ in range(self.budget):
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
        Iterative batch robust mean for logit consensus.
        S_batch: (N, K, D).
        Removes up to budget clients globally (one per iteration, the client
        with the highest total projection magnitude across triggered samples).
        Returns (means, max_eigenvalue, removal_counts, max_ratio).
        """
        S_batch = _finite_block(S_batch)
        N, K, D = S_batch.shape

        if self.epsilon >= 0.5:
            return S_batch.mean(axis=1).astype(np.float32), None, np.zeros(K, dtype=np.int64), None

        active = np.ones(K, dtype=bool)
        removal_counts = np.zeros(K, dtype=np.int64)
        g_max_eig = None
        g_max_ratio = None
        slack = 1.0 + self.epsilon * np.log(1.0 / self.epsilon)

        for _ in range(self.budget):
            K_act = int(active.sum())
            if K_act <= 1:
                break
            S_act = S_batch[:, active, :]
            mu_S = S_act.mean(axis=1)
            centered = S_act - mu_S[:, np.newaxis, :]
            Sigma = np.einsum('nkd,nke->nde', centered, centered) / (K_act - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
            max_eigs = eigenvalues[:, -1]

            effective_rank = min(K_act - 1, D)
            top_eigs = np.sort(np.abs(eigenvalues), axis=1)[:, -effective_rank:]
            median_eigs = np.median(top_eigs, axis=1)
            thresholds = median_eigs * slack
            below_thresh = max_eigs <= thresholds

            above = ~below_thresh
            if above.any():
                cur_max_eig = float(max_eigs[above].max())
                if g_max_eig is None or cur_max_eig > g_max_eig:
                    g_max_eig = cur_max_eig
                ratios = max_eigs / np.maximum(median_eigs, 1e-10)
                cur_max_ratio = float(ratios[above].max())
                if g_max_ratio is None or cur_max_ratio > g_max_ratio:
                    g_max_ratio = cur_max_ratio

            if below_thresh.all():
                break

            v_star = eigenvectors[:, :, -1]
            abs_proj = np.abs(np.einsum('nkd,nd->nk', centered, v_star))
            abs_proj[below_thresh] = 0.0
            worst_active = int(np.argmax(abs_proj.sum(axis=0)))
            worst_global = int(np.where(active)[0][worst_active])
            active[worst_global] = False
            removal_counts[worst_global] = 1

        means = S_batch[:, active, :].mean(axis=1).astype(np.float32)
        return means, g_max_eig, removal_counts, g_max_ratio

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


class CronusRobustFilter:
    """
    Cronus-specific robust filter with a fixed eigenvalue threshold of 9.
    No epsilon, no slack, no tail bound — just budget-limited removal.
    """

    THRESHOLD = 9.0

    def __init__(self, budget: int = 0, workers: int = 8, row_block_size: int = 8192):
        self.budget = int(budget)
        self.workers = max(1, int(workers))
        self.row_block_size = max(1, int(row_block_size))

    def compute_robust_mean_batch(self, S_batch: np.ndarray) -> tuple:
        S_batch = _finite_block(S_batch)
        N, K, D = S_batch.shape
        active = np.ones(K, dtype=bool)
        removal_counts = np.zeros(K, dtype=np.int64)
        g_max_eig = None

        for _ in range(self.budget):
            K_act = int(active.sum())
            if K_act <= 1:
                break
            S_act = S_batch[:, active, :]
            if self.workers > 1 and N > self.row_block_size:
                blocks = [(start, min(start + self.row_block_size, N)) for start in range(0, N, self.row_block_size)]
                proj_sums = np.zeros(K_act, dtype=np.float64)
                cur = None
                with ThreadPoolExecutor(max_workers=min(self.workers, len(blocks))) as pool:
                    futures = [
                        pool.submit(_cronus_block, S_act[start:end], self.THRESHOLD)
                        for start, end in blocks
                    ]
                    for future in futures:
                        block_proj, block_max = future.result()
                        proj_sums += block_proj
                        if block_max is not None and (cur is None or block_max > cur):
                            cur = block_max
            else:
                proj_sums, cur = _cronus_block(S_act, self.THRESHOLD)

            if cur is not None and (g_max_eig is None or cur > g_max_eig):
                g_max_eig = cur

            if cur is None:
                break

            worst_active = int(np.argmax(proj_sums))
            worst_global = int(np.where(active)[0][worst_active])
            active[worst_global] = False
            removal_counts[worst_global] = 1

        means = S_batch[:, active, :].mean(axis=1).astype(np.float32)
        return means, g_max_eig, removal_counts, None


class AdaptiveRobustFilter:
    """
    Distribution-aware Byzantine filter using MAD standardisation and
    Blom half-normal order statistics.  See docs/robust_filter.md.

    Parameters
    ----------
    budget : int
        Maximum total number of clients to remove across all passes.
    tail_threshold : float
        Minimum fraction of samples (default 0.75) for which a client must
        appear in the contiguous outer tail to be flagged as Byzantine.
    workers : int
        Number of row-block worker threads.
    """

    def __init__(self, budget: int = 0, tail_threshold: float = 0.75, workers: int = 8, row_block_size: int = 8192):
        self.budget = int(budget)
        self.tail_threshold = tail_threshold
        self.workers = max(1, int(workers))
        self.row_block_size = max(1, int(row_block_size))

    def _iter_row_blocks(self, n_rows: int):
        for start in range(0, n_rows, self.row_block_size):
            yield start, min(start + self.row_block_size, n_rows)

    def _score_active_set(self, S_act: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        from scipy.special import ndtri

        n_rows, K_act, _ = S_act.shape
        k_vals = np.arange(1, K_act + 1, dtype=np.float64)
        ref_hn = ndtri(0.5 + 0.5 * (k_vals - 0.375) / (K_act + 0.25))

        if self.workers == 1 or n_rows <= self.row_block_size:
            counts, max_eig = _adaptive_tail_block(S_act, ref_hn)
            return counts.astype(np.float64) / n_rows, max_eig

        counts = np.zeros(K_act, dtype=np.int64)
        max_eig = None
        blocks = list(self._iter_row_blocks(n_rows))
        with ThreadPoolExecutor(max_workers=min(self.workers, len(blocks))) as pool:
            futures = [
                pool.submit(_adaptive_tail_block, S_act[start:end], ref_hn)
                for start, end in blocks
            ]
            for future in futures:
                block_counts, block_max_eig = future.result()
                counts += block_counts
                if max_eig is None or block_max_eig > max_eig:
                    max_eig = block_max_eig
        return counts.astype(np.float64) / n_rows, max_eig

    def _select_flagged(self, scores: np.ndarray, budget_left: int) -> np.ndarray:
        flagged = np.where(scores >= self.tail_threshold)[0]
        if len(flagged) <= budget_left:
            return flagged
        order = np.argsort(scores[flagged])[::-1][:budget_left]
        return flagged[order]

    def compute_robust_mean_batch(self, S_batch: np.ndarray) -> tuple:
        S_batch = _finite_block(S_batch)
        _, K, _ = S_batch.shape
        active = np.ones(K, dtype=bool)
        removal_counts = np.zeros(K, dtype=np.int64)
        g_max_eig = None
        removed_total = 0

        while removed_total < self.budget:
            K_act = int(active.sum())
            if K_act <= 2:
                break

            S_act = S_batch[:, active, :]
            scores, cur_max = self._score_active_set(S_act)
            if g_max_eig is None or cur_max > g_max_eig:
                g_max_eig = cur_max

            flagged = self._select_flagged(scores, self.budget - removed_total)
            if len(flagged) == 0:
                break

            global_idx = np.where(active)[0][flagged]
            active[global_idx] = False
            removal_counts[global_idx] = 1
            removed_total += len(flagged)

        means = S_batch[:, active, :].mean(axis=1).astype(np.float32)
        return means, g_max_eig, removal_counts, None


class IterativeRobustFilter(AdaptiveRobustFilter):
    def _select_flagged(self, scores: np.ndarray, budget_left: int) -> np.ndarray:
        flagged = np.where(scores >= self.tail_threshold)[0]
        if len(flagged) == 0:
            return flagged
        best_local = int(flagged[np.argmax(scores[flagged])])
        return np.array([best_local], dtype=np.int64)


class RobustFilterWeights(RobustFilter):
    def __init__(self, epsilon: float = 0.2, budget: Optional[int] = None, **_kw):
        super().__init__(epsilon=epsilon, budget=budget)

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
    def __init__(self, epsilon: float = 0.2, budget: Optional[int] = None, **_kw):
        super().__init__(epsilon=epsilon, budget=budget)

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
