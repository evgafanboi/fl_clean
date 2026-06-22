import logging
import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import HDBSCAN


class FLAME:
    def __init__(self, lambda_dp: float = 0.001, passive_cluster: bool = False):
        self.name = "FLAME"
        self.lambda_dp = lambda_dp
        self.passive_cluster = passive_cluster

    def extra_log_tokens(self):
        return {"lambda": self.lambda_dp, "passive": self.passive_cluster}

    def aggregate(self, weights_list, sample_sizes=None, prev_global=None, logger=None, passive_cluster=None):
        n = len(weights_list)
        flat = [np.concatenate([w.ravel() for w in wl]).astype(np.float64) for wl in weights_list]

        ref_flat = (
            np.concatenate([w.ravel() for w in prev_global]).astype(np.float64)
            if prev_global is not None
            else np.mean(flat, axis=0)
        )

        # Cosine distance on UPDATE VECTORS (delta = w_i - w_global), not raw weights
        deltas = [f - ref_flat for f in flat]
        delta_norms = np.array([np.linalg.norm(d) for d in deltas])
        normed = [d / (delta_norms[i] + 1e-12) for i, d in enumerate(deltas)]
        cos_dist = 1.0 - np.clip(
            np.array([[np.dot(ni, nj) for nj in normed] for ni in normed]), -1.0, 1.0
        )
        np.fill_diagonal(cos_dist, 0.0)

        log = logger or logging.getLogger()
        use_passive = passive_cluster if passive_cluster is not None else self.passive_cluster

        if use_passive:
            labels = HDBSCAN(metric="precomputed").fit_predict(cos_dist)
        else:
            min_cs = max(2, n // 2 + 1)
            labels = HDBSCAN(min_cluster_size=min_cs, metric="precomputed", min_samples=1).fit_predict(cos_dist)

        cluster_members = defaultdict(list)
        for i, lb in enumerate(labels):
            cluster_members[lb].append(i)
        n_clusters = sum(1 for lb in cluster_members if lb != -1)
        noise_ids = cluster_members.get(-1, [])

        if use_passive:
            # Passive mode: no majority constraint, just pick the largest non-noise cluster
            if n_clusters > 0:
                best_label = max((lb for lb in cluster_members if lb != -1), key=lambda lb: len(cluster_members[lb]))
                admitted = cluster_members[best_label]
                admission_mode = "passive_largest"
            else:
                admitted = list(range(n))
                admission_mode = "passive_all_noise_fallback"
        else:
            min_cs = max(2, n // 2 + 1)
            majority_label = next(
                (lb for lb, cnt in Counter(lb for lb in labels if lb != -1).items() if cnt >= min_cs),
                None,
            )
            if majority_label is not None:
                admitted = cluster_members[majority_label]
                admission_mode = "majority"
            elif n_clusters > 0:
                best_label = max((lb for lb in cluster_members if lb != -1), key=lambda lb: len(cluster_members[lb]))
                admitted = cluster_members[best_label]
                admission_mode = "largest_fallback"
            else:
                admitted = list(range(n))
                admission_mode = "all_noise_fallback"

        dbg = f"[FLAME] HDBSCAN: {n_clusters} clusters, {len(noise_ids)} noise | mode={admission_mode} | admitted={len(admitted)}/{n}"
        print(f"  {dbg}")
        log.info(dbg)
        for lb in sorted(cluster_members):
            label_str = f"cluster_{lb}" if lb != -1 else "noise"
            msg = f"[FLAME] {label_str} (n={len(cluster_members[lb])}): {sorted(cluster_members[lb])}"
            print(f"    {msg}")
            log.info(msg)

        S_t = float(np.median(delta_norms))

        clipped = []
        n_clipped = 0
        for i in admitted:
            gamma = S_t / (delta_norms[i] + 1e-12)
            if delta_norms[i] > S_t:
                n_clipped += 1
            clipped.append(ref_flat + deltas[i] * min(1.0, gamma))

        agg_flat = np.mean(clipped, axis=0)
        sigma = self.lambda_dp * S_t
        noise = np.random.normal(0.0, sigma, size=agg_flat.shape)
        noise_norm = float(np.linalg.norm(noise))
        agg_flat += noise

        summary = f"[FLAME] S_t={S_t:.4f} | n_clipped={n_clipped}/{len(admitted)} | sigma={sigma:.6f} | noise_l2={noise_norm:.4f}"
        print(f"  {summary}")
        log.info(summary)

        result, offset = [], 0
        for w_ref in weights_list[0]:
            sz = w_ref.size
            result.append(agg_flat[offset : offset + sz].reshape(w_ref.shape).astype(np.float32))
            offset += sz
        return result
