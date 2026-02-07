import numpy as np


class FedMLB:
    def __init__(self, lambda1=1.0, lambda2=1.0, temperature=1.0):
        self.name = "FedMLB"
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.temperature = temperature
    
    def extra_log_tokens(self):
        return {"lambda1": self.lambda1, "lambda2": self.lambda2, "temperature": self.temperature}
    
    def aggregate(self, client_weights_list, sample_sizes=None):
        if not client_weights_list:
            raise ValueError("Cannot aggregate empty weights list")
        
        num_clients = len(client_weights_list)
        
        aggregated_weights = []
        for layer_idx in range(len(client_weights_list[0])):
            layer_sum = np.sum([weights[layer_idx] for weights in client_weights_list], axis=0)
            aggregated_weights.append(layer_sum / num_clients)
        
        return aggregated_weights
