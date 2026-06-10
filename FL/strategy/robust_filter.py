from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Tuple, Optional

from ..colors import COLORS


FINITE_CAP = 1e6


def _finite_block(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values.astype(np.float64, copy=False), nan=0.0, posinf=FINITE_CAP, neginf=-FINITE_CAP)


def _expand_exceeding_tie_blocks(sorted_values: np.ndarray, exceeds: np.ndarray) -> np.ndarray:
    if sorted_values.shape[1] <= 1:
        return exceeds

    same_as_prev = np.isclose(sorted_values[:, 1:], sorted_values[:, :-1], rtol=1e-9, atol=1e-12)
    expanded = exceeds.copy()

    # Push any triggered rank across equal-value ties to the right.
    for col in range(1, expanded.shape[1]):
        expanded[:, col] |= expanded[:, col - 1] & same_as_prev[:, col - 1]

    # Then propagate back left so the whole tie block is treated uniformly.
    for col in range(expanded.shape[1] - 2, -1, -1):
        expanded[:, col] |= expanded[:, col + 1] & same_as_prev[:, col]

    return expanded


_CRONUS_SUB_BATCH = 512


def _cronus_first_mask(centered: np.ndarray, v_star: np.ndarray) -> np.ndarray:
    abs_proj = np.abs(np.einsum('nkd,nd->nk', centered, v_star, optimize=True))
    thresholds = np.sqrt(np.random.random(size=len(centered)))[:, np.newaxis] * abs_proj.max(axis=1, keepdims=True)
    keep = abs_proj < thresholds
    empty = ~keep.any(axis=1)
    if empty.any():
        empty_rows = np.flatnonzero(empty)
        keep[empty_rows, np.argmin(abs_proj[empty_rows], axis=1)] = True
    return keep


def _cronus_refine_masks(
    S_block: np.ndarray,
    threshold: float,
    budget: int,
    survivor_mask: np.ndarray,
) -> np.ndarray:
    pending = np.ones(len(S_block), dtype=bool)

    for _ in range(int(budget)):
        row_idx = np.flatnonzero(pending)
        if row_idx.size == 0:
            break

        active = survivor_mask[row_idx]
        counts = active.sum(axis=1)
        valid = counts > 1
        if not valid.all():
            pending[row_idx[~valid]] = False
            row_idx = row_idx[valid]
            if row_idx.size == 0:
                break
            active = active[valid]
            counts = counts[valid]

        block = S_block[row_idx]
        mu = np.einsum('rk,rkd->rd', active, block, optimize=True) / counts[:, np.newaxis]
        centered = (block - mu[:, np.newaxis, :]) * active[:, :, np.newaxis]
        Sigma = np.einsum('nkd,nke->nde', centered, centered, optimize=True) / (counts - 1)[:, np.newaxis, np.newaxis]
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        max_eigs = eigenvalues[:, -1]
        triggered = max_eigs > threshold

        if not triggered.any():
            pending[row_idx] = False
            break

        pending[row_idx[~triggered]] = False
        active = active[triggered]
        centered = centered[triggered]
        row_idx = row_idx[triggered]

        abs_proj = np.abs(np.einsum('nkd,nd->nk', centered, eigenvectors[triggered, :, -1], optimize=True))
        thresholds = np.sqrt(np.random.random(size=len(row_idx)))[:, np.newaxis] * abs_proj.max(axis=1, keepdims=True)
        keep = active & (abs_proj < thresholds)
        empty = ~keep.any(axis=1)
        if empty.any():
            empty_rows = np.flatnonzero(empty)
            masked_proj = np.where(active[empty_rows], abs_proj[empty_rows], np.inf)
            keep[empty_rows, np.argmin(masked_proj, axis=1)] = True
        survivor_mask[row_idx] = keep

    return survivor_mask


def _cronus_exact_block(
    S_block: np.ndarray,
    threshold: float,
    budget: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float], np.ndarray]:
    rows, K, _ = S_block.shape
    survivor_mask = np.ones((rows, K), dtype=bool)
    pre_top_eigs = np.zeros(rows, dtype=np.float64)

    for sb0 in range(0, rows, _CRONUS_SUB_BATCH):
        sb1 = min(sb0 + _CRONUS_SUB_BATCH, rows)
        block = S_block[sb0:sb1]
        centered = block - block.mean(axis=1, keepdims=True)
        Sigma = np.einsum('nkd,nke->nde', centered, centered, optimize=True) / (K - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        top_eigs = eigenvalues[:, -1]
        pre_top_eigs[sb0:sb1] = top_eigs

        triggered = top_eigs > threshold
        if not triggered.any():
            continue

        block_mask = survivor_mask[sb0:sb1]
        triggered_rows = np.flatnonzero(triggered)
        first_keep = _cronus_first_mask(centered[triggered], eigenvectors[triggered, :, -1])
        block_mask[triggered] = first_keep

        if budget > 1:
            block_mask[triggered] = _cronus_refine_masks(
                block[triggered_rows],
                threshold,
                budget - 1,
                first_keep.copy(),
            )

    survivor_weights = survivor_mask.astype(np.float32)
    support = survivor_weights.sum(axis=1)
    means = np.einsum('rk,rkd->rd', survivor_weights, S_block, optimize=True)
    means = (means / support[:, np.newaxis]).astype(np.float32)
    removal_counts = (~survivor_mask).sum(axis=0).astype(np.int64)
    block_max = float(pre_top_eigs.max()) if rows > 0 else None
    return means, survivor_mask, removal_counts, block_max, pre_top_eigs


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
    Paper-style Cronus filter with a fixed eigenvalue threshold of 9 and
    Beta(2, 1) threshold sampling.

    budget limits the number of Cronus filtering passes per sample.
    """

    THRESHOLD = 9.0

    def __init__(self, budget: int = 0, workers: int = 8, row_block_size: int = 8192):
        self.budget = int(budget)
        self.workers = max(1, int(workers))
        self.row_block_size = max(1, int(row_block_size))
        self.last_filter_stats = None
        self._last_worker_log = None

    def _log_effective_workers(self, rows: int, block_count: int, effective_workers: int) -> None:
        signature = (rows, block_count, effective_workers)
        if signature == self._last_worker_log:
            return
        print(
            f"{COLORS.PURPLE}Cronus filter workers: requested={self.workers} "
            f"effective={effective_workers} blocks={block_count} rows={rows} "
            f"row_block_size={self.row_block_size}{COLORS.ENDC}"
        )
        self._last_worker_log = signature

    def _store_filter_stats(self, top_eigs: np.ndarray) -> None:
        self.last_filter_stats = {
            "pre_top_eigs": top_eigs,
            "min_eig": float(top_eigs.min()) if len(top_eigs) > 0 else None,
            "mean_eig": float(top_eigs.mean()) if len(top_eigs) > 0 else None,
            "med_eig": float(np.median(top_eigs)) if len(top_eigs) > 0 else None,
            "max_eig": float(top_eigs.max()) if len(top_eigs) > 0 else None,
        }

    def filter_batch(self, S_batch: np.ndarray) -> tuple:
        S_batch = _finite_block(S_batch)
        N, K, D = S_batch.shape
        means = np.empty((N, D), dtype=np.float32)
        survivor_mask = np.zeros((N, K), dtype=bool)
        removal_counts = np.zeros(K, dtype=np.int64)
        g_max_eig = None
        pre_top_eigs = np.zeros(N, dtype=np.float64)

        if self.budget <= 0 or K <= 1:
            mu = S_batch.mean(axis=1)
            means[:] = mu.astype(np.float32)
            survivor_mask[:] = True
            if K > 1:
                centered = S_batch - mu[:, None, :]
                Sigma = np.einsum('nkd,nke->nde', centered, centered) / (K - 1)
                pre_top_eigs = np.linalg.eigvalsh(Sigma)[:, -1]
                g_max_eig = float(pre_top_eigs.max())
            self._store_filter_stats(pre_top_eigs)
            return means, g_max_eig, removal_counts, survivor_mask

        block_count = max(1, (N + self.row_block_size - 1) // self.row_block_size)
        effective_workers = min(self.workers, block_count) if self.workers > 1 and N > self.row_block_size else 1
        self._log_effective_workers(N, block_count, effective_workers)

        if effective_workers > 1:
            blocks = [(start, min(start + self.row_block_size, N)) for start in range(0, N, self.row_block_size)]
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                futures = {
                    pool.submit(_cronus_exact_block, S_batch[start:end], self.THRESHOLD, self.budget): (start, end)
                    for start, end in blocks
                }
                for future, (start, end) in futures.items():
                    block_means, block_survivors, block_counts, block_max, block_top_eigs = future.result()
                    means[start:end] = block_means
                    survivor_mask[start:end] = block_survivors
                    removal_counts += block_counts
                    pre_top_eigs[start:end] = block_top_eigs
                    if block_max is not None and (g_max_eig is None or block_max > g_max_eig):
                        g_max_eig = block_max
        else:
            means, survivor_mask, removal_counts, g_max_eig, pre_top_eigs = _cronus_exact_block(S_batch, self.THRESHOLD, self.budget)

        self._store_filter_stats(pre_top_eigs)

        return means, g_max_eig, removal_counts, survivor_mask

    def compute_robust_mean_batch(self, S_batch: np.ndarray) -> tuple:
        means, g_max_eig, removal_counts, _ = self.filter_batch(S_batch)
        return means, g_max_eig, removal_counts, None


class RobustFilterV3:
    """
    Per-sample batch-spectral filter.

    For each public sample, projects onto the top covariance eigenvector,
    MAD-standardises, then removes ALL clients whose sorted standardised
    deviation exceeds its rank-specific Blom half-normal reference in one
    batch.  The mean and covariance are recomputed on the surviving set and
    the check is repeated until no client exceeds its reference or fewer
    than K//2 clients remain for that sample.

    Discard counts are accumulated across all samples; clients whose discard
    fraction exceeds robust_threshold are excluded from the final mean.
    No budget parameter — K//2 floor and robust_threshold are the only controls.

    Parameters
    ----------
    robust_threshold : float
        Remove client i if (times_discarded / n_samples) > robust_threshold.
        Default 0.3.
    workers : int
        Number of row-block worker threads.
    row_block_size : int
        Rows per worker block.
    """

    def __init__(self, robust_threshold: float = 0.3, reference: str = "Blom", t_df: float = 4.0,
                 workers: int = 8, row_block_size: int = 4096, **_kw):
        self.robust_threshold = robust_threshold
        self.reference = reference
        self.t_df = float(t_df)
        self.workers = max(1, int(workers))
        self.row_block_size = max(1, int(row_block_size))

    def _build_refs(self, K: int, K_min: int) -> dict:
        k_range = range(max(K_min, 1), K + 1)
        if self.reference == "t":
            from scipy.stats import t as t_dist
            return {
                k: t_dist.ppf(
                    0.5 + 0.5 * (np.arange(1, k + 1, dtype=np.float64) - 0.375) / (k + 0.25),
                    df=self.t_df,
                )
                for k in k_range
            }
        from scipy.special import ndtri
        return {
            k: ndtri(0.5 + 0.5 * (np.arange(1, k + 1, dtype=np.float64) - 0.375) / (k + 0.25))
            for k in k_range
        }

    def _iter_row_blocks(self, n_rows: int):
        for start in range(0, n_rows, self.row_block_size):
            yield start, min(start + self.row_block_size, n_rows)

    def count_discards_block(self, S_block: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        S_block = _finite_block(S_block)
        rows, K, D = S_block.shape
        K_min = max(K // 2, 1)
        discard_mask = np.zeros((rows, K), dtype=bool)
        pending = np.ones(rows, dtype=bool)
        g_max_eig = None

        blom_refs = self._build_refs(K, K_min)

        for _ in range(K - K_min):
            row_idx = np.flatnonzero(pending)
            if row_idx.size == 0:
                break
            active = ~discard_mask[row_idx]
            counts = active.sum(axis=1)

            too_few = counts <= K_min
            if too_few.any():
                pending[row_idx[too_few]] = False
                row_idx = row_idx[~too_few]
                if row_idx.size == 0:
                    break
                active = active[~too_few]
                counts = counts[~too_few]

            # batched eigendecomposition over all pending rows
            block = S_block[row_idx].astype(np.float64)
            act_f = active.astype(np.float64)
            mu = np.einsum('rk,rkd->rd', act_f, block) / counts[:, None]
            centered = (block - mu[:, None, :]) * active[:, :, None]
            Sigma = np.einsum('nkd,nke->nde', centered, centered) / np.maximum(counts - 1, 1)[:, None, None]
            eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
            cur_max = float(eigenvalues[:, -1].max())
            if g_max_eig is None or cur_max > g_max_eig:
                g_max_eig = cur_max

            v_star = eigenvectors[:, :, -1]
            projections = np.einsum('nkd,nd->nk', centered, v_star)
            proj_active = np.where(active, projections, np.nan)
            med_p = np.nanmedian(proj_active, axis=1, keepdims=True)
            abs_dev = np.abs(projections - med_p)
            sigma_r = np.maximum(
                np.nanmedian(np.where(active, abs_dev, np.nan), axis=1, keepdims=True) * 1.4826,
                1e-10,
            )
            # inactive clients get -inf so they never exceed any reference
            abs_z = np.where(active, abs_dev / sigma_r, -np.inf)

            # sort per row; active clients occupy the rightmost K_r slots ─
            sort_idx = np.argsort(abs_z, axis=1)                      # (n_pending, K)
            sorted_z = np.take_along_axis(abs_z, sort_idx, axis=1)    # (n_pending, K)

            # compare all active ranks to their Blom reference; batch per K_r ─
            new_discards = np.zeros((len(row_idx), K), dtype=bool)
            any_flagged = False
            for K_r in np.unique(counts):
                K_r = int(K_r)
                if K_r <= K_min:
                    continue
                grp = counts == K_r
                ref = blom_refs[K_r]                              # (K_r,)
                active_z = sorted_z[grp, K - K_r:]                # (n_grp, K_r)
                exceeds = active_z > ref[None, :]                  # (n_grp, K_r)
                if not exceeds.any():
                    continue
                exceeds = _expand_exceeding_tie_blocks(active_z, exceeds)
                any_flagged = True
                active_idx = sort_idx[grp, K - K_r:]              # (n_grp, K_r)
                flag = np.zeros((int(grp.sum()), K), dtype=bool)
                np.put_along_axis(flag, active_idx, exceeds, axis=1)
                new_discards[grp] |= flag

            if not any_flagged:
                pending[:] = False
                break

            had_flag = new_discards.any(axis=1)
            discard_mask[row_idx] |= new_discards
            pending[row_idx[~had_flag]] = False

        return discard_mask.sum(axis=0).astype(np.int64), discard_mask, g_max_eig

    def count_discards(self, S_batch: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        S_batch = _finite_block(S_batch)
        rows, K, _ = S_batch.shape
        blocks = list(self._iter_row_blocks(rows))
        effective_workers = min(self.workers, len(blocks)) if self.workers > 1 else 1
        if effective_workers == 1:
            counts, _, g_max_eig = self.count_discards_block(S_batch)
            return counts, g_max_eig
        total_counts = np.zeros(K, dtype=np.int64)
        g_max_eig = None
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = [
                pool.submit(self.count_discards_block, S_batch[start:end])
                for start, end in blocks
            ]
            for future in futures:
                counts, _, block_max = future.result()
                total_counts += counts
                if block_max is not None and (g_max_eig is None or block_max > g_max_eig):
                    g_max_eig = block_max
        return total_counts, g_max_eig

    def count_discards_mask(self, S_batch: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        S_batch = _finite_block(S_batch)
        rows, K, _ = S_batch.shape
        blocks = list(self._iter_row_blocks(rows))
        effective_workers = min(self.workers, len(blocks)) if self.workers > 1 else 1
        if effective_workers == 1:
            _, full_mask, g_max_eig = self.count_discards_block(S_batch)
            return full_mask, g_max_eig
        full_mask = np.zeros((rows, K), dtype=bool)
        g_max_eig = None
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = [
                (start, end, pool.submit(self.count_discards_block, S_batch[start:end]))
                for start, end in blocks
            ]
            for start, end, future in futures:
                _, block_mask, block_max = future.result()
                full_mask[start:end] = block_mask
                if block_max is not None and (g_max_eig is None or block_max > g_max_eig):
                    g_max_eig = block_max
        return full_mask, g_max_eig



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
