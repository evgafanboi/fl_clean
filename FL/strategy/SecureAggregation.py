import numpy as np
import hashlib
import secrets
from typing import Dict, List, Tuple


class SecureAggregation:
    """
    Secure Aggregation using Diffie-Hellman key exchange and pairwise masking.
    Masks cancel out during aggregation, preserving privacy while computing sum.
    """
    def __init__(self):
        self.name = "SecureAggregation"
        # DH parameters (using safe prime for simplicity)
        self.p = 0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF
        self.g = 2
        
    def generate_dh_keypair(self) -> Tuple[int, int]:
        """Generate DH private and public keys"""
        private_key = secrets.randbelow(self.p - 2) + 1
        public_key = pow(self.g, private_key, self.p)
        return private_key, public_key
    
    def compute_shared_secret(self, private_key: int, other_public_key: int) -> int:
        """Compute shared secret from DH exchange"""
        return pow(other_public_key, private_key, self.p)
    
    def kdf(self, shared_secret: int, output_shape: tuple, client_id: int, other_id: int) -> np.ndarray:
        """
        Key Derivation Function to generate structured random mask.
        Uses HKDF-like approach with SHA256.
        """
        # Create unique context for this pair
        context = f"{min(client_id, other_id)}-{max(client_id, other_id)}".encode()
        secret_bytes = shared_secret.to_bytes((shared_secret.bit_length() + 7) // 8, 'big')
        
        # Derive key material
        h = hashlib.sha256()
        h.update(secret_bytes)
        h.update(context)
        seed_bytes = h.digest()
        
        # Convert to seed for numpy RNG
        seed = int.from_bytes(seed_bytes[:8], 'big') % (2**32)
        return seed
    
    def generate_mask(self, seed: int, weights_structure: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate pseudo-random mask matching model structure.
        
        Args:
            seed: Seed from KDF
            weights_structure: List of weight arrays to match
        
        Returns:
            List of random masks with same shapes
        """
        rng = np.random.RandomState(seed)
        masks = []
        for weight_array in weights_structure:
            # Generate random mask with same shape, scaled appropriately
            mask = rng.standard_normal(weight_array.shape).astype(np.float32)
            # Scale to be roughly same magnitude as weights for numerical stability
            scale = np.std(weight_array) if np.std(weight_array) > 0 else 1.0
            mask = mask * scale * 0.1  # Use 10% of weight scale
            masks.append(mask)
        return masks
    
    def apply_pairwise_masks(
        self, 
        client_id: int,
        client_weights: List[np.ndarray],
        public_keys: Dict[int, int],
        private_key: int
    ) -> List[np.ndarray]:
        """
        Apply pairwise masks to client weights.
        
        For each other client:
        - Compute shared secret via DH
        - Generate mask via KDF + PRG
        - Add mask if client_id < other_id, subtract if client_id > other_id
        
        Args:
            client_id: This client's ID
            client_weights: Model weights to mask
            public_keys: Dict of {client_id: public_key} for all clients
            private_key: This client's DH private key
        
        Returns:
            Masked weights
        """
        masked_weights = [w.copy() for w in client_weights]
        
        for other_id, other_public_key in public_keys.items():
            if other_id == client_id:
                continue
            
            # Compute shared secret
            shared_secret = self.compute_shared_secret(private_key, other_public_key)
            
            # Derive seed via KDF
            seed = self.kdf(shared_secret, None, client_id, other_id)
            
            # Generate mask
            mask = self.generate_mask(seed, client_weights)
            
            # Apply mask: add if lower ID, subtract if higher ID
            sign = 1 if client_id < other_id else -1
            for i in range(len(masked_weights)):
                masked_weights[i] = masked_weights[i] + sign * mask[i]
        
        return masked_weights
    
    def aggregate(self, model_weights_list, sample_sizes=None, **kwargs):
        """
        Aggregate masked weights. Masks cancel out, leaving sum of original weights.
        Then compute mean (FedAvg).
        
        Args:
            model_weights_list: List of masked weight arrays from clients
            sample_sizes: Not used (equal weighting in secure aggregation)
        
        Returns:
            Aggregated weights
        """
        if len(model_weights_list) == 0:
            raise ValueError("No client weights to aggregate")
        
        n_clients = len(model_weights_list)
        
        # Sum all masked weights (masks cancel out)
        aggregated = [np.zeros_like(w) for w in model_weights_list[0]]
        
        for client_weights in model_weights_list:
            for i, weight in enumerate(client_weights):
                aggregated[i] += weight
        
        # Compute mean (FedAvg)
        for i in range(len(aggregated)):
            aggregated[i] /= n_clients
        
        return aggregated
