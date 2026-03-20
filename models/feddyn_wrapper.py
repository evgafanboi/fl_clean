import tensorflow as tf
import numpy as np


class FedDynModelWrapper:
    """
    FedDyn client wrapper matching the official implementation (train_feddyn_mdl).

    Client objective (Paper Eq. 1):
        L_k(θ) - <∇L_k^{t-1}, θ> + (α/2)||θ - θ_global||²

    Official implementation equivalences:
        - SGD with weight_decay = α + wd  →  provides (α+wd)/2 * ||θ||²
        - loss_algo = α * <θ, -θ_g + local_grad>  →  linear + cross term

    Our direct form (no expansion, no separate L2):
        loss = CE(y, ŷ) - <∇L_k, θ> + (α/2)||θ - θ_g||²
    Uses SGD (not Adam) so the penalty gradients are unscaled.
    Kernel regularizers from the base model are stripped to avoid double L2.
    """

    def __init__(self, base_model, feddyn_strategy, client_id):
        self.base_model = base_model
        self.model = base_model.model if hasattr(base_model, 'model') else base_model
        self.feddyn = feddyn_strategy
        self.client_id = client_id
        self._compiled = False
        self._global_weights = None
        self._grad_L = None
        self._strip_kernel_regularizers()

    def _strip_kernel_regularizers(self):
        """Remove kernel_regularizer losses — FedDyn controls all regularization."""
        for layer in self.model.layers:
            if not isinstance(layer, tf.keras.layers.Dense):
                continue
            if getattr(layer, 'kernel_regularizer', None) is not None:
                layer.kernel_regularizer = None
            for attr in ('_callable_losses', '_losses'):
                lst = getattr(layer, attr, None)
                if isinstance(lst, list):
                    lst.clear()

    def _compile_with_feddyn_loss(self):
        """Compile with direct FedDyn objective: CE - <∇L_k, θ> + (α/2)||θ - θ_g||²."""
        if self._global_weights is None:
            template = self.model.get_weights()
            self._global_weights = [tf.constant(w, dtype=tf.float32) for w in template]
            self._grad_L = [tf.constant(np.zeros_like(w), dtype=tf.float32) for w in template]

        ce_loss_fn = tf.keras.losses.CategoricalCrossentropy()
        alpha = self.feddyn.get_alpha()
        global_w = self._global_weights
        grad_L = self._grad_L
        all_vars = list(self.model.weights)
        n = min(len(all_vars), len(global_w), len(grad_L))

        def feddyn_loss(y_true, y_pred):
            ce = ce_loss_fn(y_true, y_pred)
            lin_penalty = tf.constant(0.0, dtype=tf.float32)
            quad_penalty = tf.constant(0.0, dtype=tf.float32)
            for i in range(n):
                p = tf.reshape(all_vars[i], [-1])
                g = tf.reshape(grad_L[i], [-1])
                gw = tf.reshape(global_w[i], [-1])
                lin_penalty += tf.reduce_sum(p * g)
                quad_penalty += tf.reduce_sum(tf.square(p - gw))
            return ce - lin_penalty + (alpha / 2.0) * quad_penalty

        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, clipnorm=0.5)
        self.model.compile(optimizer=optimizer, loss=feddyn_loss, metrics=['accuracy'])
        self._compiled = True

    def set_weights(self, weights):
        if hasattr(self.base_model, 'set_weights'):
            self.base_model.set_weights(weights)
        else:
            self.model.set_weights(weights)

        template = self.model.get_weights()
        self._global_weights = [tf.constant(w, dtype=tf.float32) for w in template]
        grad_L = self.feddyn.get_grad_L_for_client(self.client_id, template)
        self._grad_L = [tf.constant(g, dtype=tf.float32) for g in grad_L]
        self._compile_with_feddyn_loss()

    def fit(self, dataset, epochs=50, **kwargs):
        if not self._compiled:
            self._compile_with_feddyn_loss()
        kwargs.pop('verbose', None)
        callbacks = kwargs.pop('callbacks', None)
        return self.model.fit(dataset, epochs=epochs, callbacks=callbacks, verbose=1, **kwargs)

    def predict(self, *args, **kwargs):
        return self.base_model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.base_model.evaluate(*args, **kwargs)

    def get_weights(self):
        return self.base_model.get_weights()

    def get_feddyn_update(self):
        return {'weights': self.get_weights()}


def create_feddyn_dense_model(input_dim, num_classes, batch_size, feddyn_strategy, client_id):
    from .dense import create_enhanced_dense_model
    base_model = create_enhanced_dense_model(input_dim, num_classes, batch_size)
    return FedDynModelWrapper(base_model, feddyn_strategy, client_id)