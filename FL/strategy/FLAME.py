import logging
import numpy as np
from sklearn.cluster import HDBSCAN


class FLAME:
    def __init__(self, lambda_dp: float = 0.001):
        self.name = "FLAME"
        self.lambda_dp = lambda_dp

    def extra_log_tokens(self):
        return {"lambda": self.lambda_dp}

    def aggregate(self, weights_list, sample_sizes=None, prev_global=None):
        n = len(weights_list)
        flat = [np.concatenate([w.ravel() for w in wl]).astype(np.float64) for wl in weights_list]

        norms = np.array([np.linalg.norm(f) for f in flat])
        normed = [f / (norms[i] + 1e-12) for i, f in enumerate(flat)]
        cos_dist = 1.0 - np.clip(
            np.array([[np.dot(ni, nj) for nj in normed] for ni in normed]), -1.0, 1.0
        )
        np.fill_diagonal(cos_dist, 0.0)

        min_cs = max(2, n // 2 + 1)
        labels = HDBSCAN(min_cluster_size=min_cs, metric="precomputed", min_samples=1).fit_predict(cos_dist)
        admitted = [i for i, lb in enumerate(labels) if lb != -1]
        if not admitted:
            admitted = list(range(n))

        ref_flat = (
            np.concatenate([w.ravel() for w in prev_global]).astype(np.float64)
            if prev_global is not None
            else np.mean(flat, axis=0)
        )

        euc = np.array([np.linalg.norm(f - ref_flat) for f in flat])
        S_t = float(np.median(euc))

        clipped = []
        for i in admitted:
            gamma = S_t / (euc[i] + 1e-12)
            clipped.append(ref_flat + (flat[i] - ref_flat) * min(1.0, gamma))

        agg_flat = np.mean(clipped, axis=0)
        sigma = self.lambda_dp * S_t
        agg_flat += np.random.normal(0.0, sigma, size=agg_flat.shape)

        msg = f"[FLAME] admitted={len(admitted)}/{n} | S_t={S_t:.4f} | sigma={sigma:.6f}"
        print(f"  {msg}")
        logging.getLogger().info(msg)

        result, offset = [], 0
        for w_ref in weights_list[0]:
            sz = w_ref.size
            result.append(agg_flat[offset : offset + sz].reshape(w_ref.shape).astype(np.float32))
            offset += sz
        return result
