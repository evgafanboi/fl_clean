import numpy as np
from ..backend import use_tf as _use_tf
if _use_tf():
    import tensorflow as tf


class FLTrust:
    def __init__(self, root_iterations=1):
        self.name = "FLTrust"
        self.root_iterations = root_iterations
        self.requires_updates = True
        self.requires_root_dataset = True
    
    def extra_log_tokens(self):
        return {"root_iters": self.root_iterations}
    
    def aggregate(self, client_updates_list, sample_sizes=None, global_update=None):
        g_0 = self._flatten_weights(global_update)
        client_updates_flat = [self._flatten_weights(g) for g in client_updates_list]
        
        trust_scores = []
        normalized_updates = []
        
        for g_i in client_updates_flat:
            ts_i = self._compute_trust_score(g_i, g_0)
            trust_scores.append(ts_i)
            g_tilde_i = self._normalize_update(g_i, g_0)
            normalized_updates.append(g_tilde_i)
        
        sum_trust_scores = sum(trust_scores)
        
        if sum_trust_scores == 0:
            aggregated_flat = np.mean(normalized_updates, axis=0)
        else:
            aggregated_flat = sum(
                ts_i / sum_trust_scores * g_tilde_i 
                for ts_i, g_tilde_i in zip(trust_scores, normalized_updates)
            )
        
        aggregated_update = self._unflatten_weights(aggregated_flat, global_update)
        
        return aggregated_update
    
    def _compute_trust_score(self, g_i, g_0):
        dot_product = np.dot(g_i, g_0)
        norm_i = np.linalg.norm(g_i)
        norm_0 = np.linalg.norm(g_0)
        
        if norm_i == 0 or norm_0 == 0:
            return 0.0
        
        cos_sim = dot_product / (norm_i * norm_0)
        return max(0.0, cos_sim)
    
    def _normalize_update(self, g_i, g_0):
        norm_i = np.linalg.norm(g_i)
        norm_0 = np.linalg.norm(g_0)
        
        if norm_i == 0:
            return g_i
        return (norm_0 / norm_i) * g_i
    
    def _flatten_weights(self, weights_list):
        return np.concatenate([w.flatten() for w in weights_list])
    
    def _unflatten_weights(self, flat_array, reference_weights):
        weights_list = []
        offset = 0
        
        for ref_w in reference_weights:
            size = ref_w.size
            shape = ref_w.shape
            weights_list.append(flat_array[offset:offset + size].reshape(shape))
            offset += size
        
        return weights_list
