import numpy as np
from ..backend import use_tf as _use_tf
if _use_tf():
    import tensorflow as tf
from .FedAvg import FedAvg


def _train_client_pt(model, dataset, epochs, mu, global_weights):
    import torch
    import torch.nn.functional as F
    from ..backend import get_torch_device
    dev = get_torch_device()
    net = model.nn
    opt = model.optimizer
    net.to(dev)
    net.train()
    sd_keys = list(net.state_dict().keys())
    gw_map = {}
    for i, k in enumerate(sd_keys):
        if i < len(global_weights):
            gw_map[k] = torch.from_numpy(np.array(global_weights[i], dtype=np.float32)).to(dev)
    total_loss, n_batches = 0.0, 0
    for _ in range(epochs):
        for X, y in dataset:
            X = X.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            if y.ndim > 1:
                y = y.argmax(dim=1)
            opt.zero_grad(set_to_none=True)
            logits = net(X, return_logits=True)
            ce = F.cross_entropy(logits, y, label_smoothing=0.05)
            prox = sum(
                (p - gw_map[k]).pow(2).sum()
                for k, p in net.named_parameters() if k in gw_map and p.requires_grad
            )
            loss = ce + (mu / 2.0) * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)

class FedProx(FedAvg):
    def __init__(self, mu=0.01):
        super().__init__()
        self.name = "FedProx"
        self.mu = mu
        self.global_weights = None
        print(f"FedProx initialized with mu={mu}")
    
    def extra_log_tokens(self):
        return {"mu": self.mu}
    
    def aggregate(self, model_weights_list, sample_sizes=None):
        aggregated_weights = super().aggregate(model_weights_list, sample_sizes)
        self.global_weights = [w.copy() for w in aggregated_weights]
        return aggregated_weights

    def train_client(self, model, dataset, epochs, client_id=None, global_weights=None):
        if global_weights is None:
            return model.fit(dataset, epochs=epochs, verbose=2).history.get("loss", [0.0])[-1]
        if not _use_tf():
            return _train_client_pt(model, dataset, epochs, self.mu, global_weights)

        keras_model = model.model if hasattr(model, 'model') else model
        trainable = keras_model.trainable_variables
        optimizer = keras_model.optimizer
        all_weights = keras_model.weights
        trainable_idx = [i for i, w in enumerate(all_weights) if w.trainable]
        n = len(trainable)

        if not hasattr(model, '_fedprox_train_step'):
            gw_vars = [tf.Variable(tf.zeros_like(trainable[i]), trainable=False) for i in range(n)]
            ce_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
            mu = self.mu

            @tf.function
            def train_step(X, y):
                with tf.GradientTape() as tape:
                    pred = keras_model(X, training=True)
                    ce = ce_fn(y, pred)
                    prox = tf.constant(0.0, dtype=tf.float32)
                    for i in range(n):
                        diff = tf.reshape(trainable[i], [-1]) - tf.reshape(gw_vars[i], [-1])
                        prox += tf.reduce_sum(tf.square(diff))
                    loss = ce + (mu / 2.0) * prox
                grads = tape.gradient(loss, trainable)
                optimizer.apply_gradients(zip(grads, trainable))
                return loss

            model._fedprox_train_step = train_step
            model._fedprox_gw_vars = gw_vars

        for var, idx in zip(model._fedprox_gw_vars, trainable_idx):
            var.assign(tf.cast(global_weights[idx], var.dtype))
        train_step = model._fedprox_train_step

        total_loss = 0.0
        n_batches = 0
        for _ in range(epochs):
            for X, y in dataset:
                total_loss += float(train_step(X, y))
                n_batches += 1
        return total_loss / max(n_batches, 1)
