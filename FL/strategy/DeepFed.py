import numpy as np
from phe import paillier


class DeepFed:
    
    def __init__(self, key_length: int = 1024):
        self.name = "DeepFed"
        self.key_length = key_length
        self.scaling_factor = 1000
        
        # Trust authority generates keypair (distributed to all clients)
        print(f"Trust authority generating Paillier keypair (key_length={key_length})...")
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_length)
        print("Keypair generated and distributed to clients")
    
    def aggregate(self, model_weights_list, sample_sizes=None):
        if sample_sizes is None:
            sample_sizes = [1] * len(model_weights_list)
        
        total_samples = sum(sample_sizes)
        weights_shape = [w.shape for w in model_weights_list[0]]
        
        # Compute contribution ratios (alpha_i = sample_size_i / total_samples)
        # Paper amplifies by 1000 to convert to positive integers
        contribution_ratios = [int((size / total_samples) * self.scaling_factor) for size in sample_sizes]
        
        print(f"\nDeepFed Aggregation: {len(model_weights_list)} clients, {total_samples} total samples")
        print(f"Contribution ratios (scaled integers): {contribution_ratios}")

        print("\n[ParaEncrypt] Clients encrypting parameters...")
        encrypted_weights_list = []
        
        for client_idx, weights in enumerate(model_weights_list):
            print(f"  Client {client_idx}: Encrypting {len(weights)} layers")
            encrypted_client_weights = []
            
            for layer_weights in weights:
                flat_weights = layer_weights.flatten()
                # Convert to positive integers: m' = f(m) = 10^8 * m mod n
                encrypted_layer = [self.public_key.encrypt(float(w)) for w in flat_weights]
                encrypted_client_weights.append((encrypted_layer, layer_weights.shape))
            
            encrypted_weights_list.append(encrypted_client_weights)
        print("\n[ParaAggregate] Server aggregating encrypted parameters...")
        
        aggregated_encrypted = []
        
        for layer_idx in range(len(weights_shape)):
            # Equation (6): c = ∏(E_Pai(m_i)^α_i)
            # In phe library: encrypted * scalar performs scalar multiplication
            # This gives us E(m) * α, which is what we need
            encrypted_layer_0, _ = encrypted_weights_list[0][layer_idx]
            aggregated_layer = [enc_val * contribution_ratios[0] for enc_val in encrypted_layer_0]
            
            # Add remaining clients' contributions
            for client_idx in range(1, len(encrypted_weights_list)):
                encrypted_layer_i, _ = encrypted_weights_list[client_idx][layer_idx]
                ratio = contribution_ratios[client_idx]
                
                for j in range(len(aggregated_layer)):
                    # E(m) * α gives E(m*α), then E(a) + E(b) gives E(a+b)
                    aggregated_layer[j] = aggregated_layer[j] + (encrypted_layer_i[j] * ratio)
            
            aggregated_encrypted.append((aggregated_layer, weights_shape[layer_idx]))
        
        print("\n[ParaDecrypt] Clients decrypting aggregated parameters...")
        aggregated_weights = []
        
        n = self.public_key.n
        n_half = n / 2
        
        for layer_idx, (encrypted_layer, shape) in enumerate(aggregated_encrypted):
            decrypted_flat = np.array([self.private_key.decrypt(enc_val) for enc_val in encrypted_layer])
            
            # Divide by scaling factor (1000) to get floating point sum
            float_sum = decrypted_flat / self.scaling_factor
            
            # Modular sign recovery (equation 8):
            float_sum = np.where(float_sum >= n_half, float_sum - n, float_sum)
            
            decrypted_layer = float_sum.reshape(shape)
            aggregated_weights.append(decrypted_layer)
        
        print("  Clients have decrypted and normalized aggregated weights")
        
        return aggregated_weights
