import numpy as np
from .FedAvg import FedAvg


class FedDyn(FedAvg):
    """
    FedDyn strategy: Federated Learning with Dynamic Regularization (Paper-correct version).
    
    Paper Algorithm 1:
    - Client objective: L_k(θ) - <∇L_k(θ_k^{t-1}), θ> + (α/2)||θ - θ^{t-1}||²
    - After solving, update: ∇L_k(θ_k^t) = ∇L_k(θ_k^{t-1}) - α(θ_k^t - θ^{t-1})
    - Server computes: h^t = h^{t-1} - α/m * Σ(θ_k^t - θ^{t-1}) over all m clients
    - Global model: θ^t = (1/|P_t|) * Σ_{k∈P_t} θ_k^t - (1/α) * h^t
    
    Note: Official code uses m (total clients) for h^t normalization, not |P_t| (participants).
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.name = "FedDyn"
        self.alpha = float(alpha)
        self._grad_L = {}  # client_id -> ∇L_k(θ_k^t) maintained recursively
        self._prev_global = None  # θ^{t-1}
        self._h = None  # h^t server-side drift accumulator
        self._n_clients_total = None
        self.round_num = 0
        print(f"FedDyn initialized with alpha={self.alpha}")

    def _zeros_like_weights(self, weights_list):
        return [np.zeros_like(w) for w in weights_list]

    def get_grad_L_for_client(self, client_id, template_weights):
        """Return ∇L_k(θ_k^t) for a client (initialized to zeros)."""
        if client_id not in self._grad_L:
            self._grad_L[client_id] = self._zeros_like_weights(template_weights)
        return self._grad_L[client_id]

    def get_prev_global(self, template_weights):
        """Return θ^{t-1}."""
        if self._prev_global is None:
            self._prev_global = self._zeros_like_weights(template_weights)
        return self._prev_global

    def get_h(self, template_weights):
        """Return h^t."""
        if self._h is None:
            self._h = self._zeros_like_weights(template_weights)
        return self._h

    def aggregate(self, model_weights_list, sample_sizes=None, participating_clients=None):
        """
        FedDyn aggregation following paper Algorithm 1:
        1. For each participant: ∇L_k(θ_k^t) = ∇L_k(θ_k^{t-1}) - α(θ_k^t - θ^{t-1})
        2. Update h^t = h^{t-1} - (α/m) * Σ_{all m clients}(θ_k^t - θ^{t-1})
           Note: Non-participants have θ_k^t = θ^{t-1}, so their drift is 0
        3. Compute θ^t = (1/|P_t|) * Σ_{k∈P_t} θ_k^t - (1/α) * h^t
        
        Args:
            model_weights_list: list of participant weights
            sample_sizes: list of sample counts (for weighted avg if needed)
            participating_clients: list of client IDs
        Returns:
            aggregated_weights (global model θ^t)
        """
        self.round_num += 1

        if participating_clients is None:
            participating_clients = list(range(len(model_weights_list)))
        
        if self._n_clients_total is None:
            self._n_clients_total = max(participating_clients) + 1

        prev_global = self.get_prev_global(model_weights_list[0])
        h_prev = self.get_h(model_weights_list[0])

        # Step 1: Update ∇L_k for participants and accumulate drifts
        total_drift = [np.zeros_like(w) for w in prev_global]
        
        for cid, client_weights in zip(participating_clients, model_weights_list):
            grad_L = self.get_grad_L_for_client(cid, client_weights)
            
            # Drift: θ_k^t - θ^{t-1}
            drift = [client_weights[j] - prev_global[j] for j in range(len(client_weights))]
            
            # Update ∇L_k(θ_k^t) = ∇L_k(θ_k^{t-1}) - α * drift
            for j in range(len(grad_L)):
                grad_L[j] = grad_L[j] - self.alpha * drift[j]
                total_drift[j] += drift[j]

        # Step 2: Update h^t = h^{t-1} - (α/m) * Σ drifts
        # Paper uses m (total clients), official code averages over ALL clients
        h_new = [h_prev[j] - (self.alpha / self._n_clients_total) * total_drift[j] 
                 for j in range(len(h_prev))]
        self._h = h_new

        # Step 3: Compute θ^t = avg(θ_k) - (1/α) * h^t
        # Use uniform averaging (paper doesn't specify sample-weighted)
        avg_weights = super().aggregate(model_weights_list, sample_sizes=None)
        
        global_model = [avg_weights[j] - (1.0 / self.alpha) * h_new[j] 
                       for j in range(len(avg_weights))]

        # Store for next round
        self._prev_global = [w.copy() for w in global_model]

        print(f"FedDyn Round {self.round_num}: Aggregated {len(model_weights_list)} clients")
        return global_model

    def get_alpha(self):
        """Return alpha for client-side regularization."""
        return self.alpha
