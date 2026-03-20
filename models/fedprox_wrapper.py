import tensorflow as tf
import numpy as np

class FedProxModelWrapper:
    def __init__(self, base_model, fedprox_strategy):
        self.base_model = base_model
        self.fedprox = fedprox_strategy
        self.model = base_model.model if hasattr(base_model, 'model') else base_model
        self._global_weights = None
        self._compiled = False

    def _compile_with_proximal_loss(self):
        """Compile with proximal loss, keeping global weights as a mutable tf.Variable."""
        mu = self.fedprox.mu
        all_vars = self.model.trainable_weights

        def proximal_loss(y_true, y_pred):
            ce_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)(y_true, y_pred)

            p_flat = tf.concat([tf.reshape(v, [-1]) for v in all_vars], axis=0)
            proximal_term = tf.reduce_sum(tf.square(p_flat - self._global_weights))

            return ce_loss + (mu / 2.0) * proximal_term

        self.model.compile(
            optimizer=self.model.optimizer,
            loss=proximal_loss,
            metrics=['accuracy'],
        )
        self._compiled = True

    def set_weights(self, weights):
        # Update model weights first
        self.base_model.set_weights(weights)

        # Update or create global weights variable
        if self.fedprox.global_weights is None:
            return

        flat_global = np.concatenate([w.flatten() for w in self.fedprox.global_weights]).astype(np.float32)
        if self._global_weights is None:
            self._global_weights = tf.Variable(flat_global, trainable=False, name='fedprox_global_weights')
        else:
            self._global_weights.assign(flat_global)

        if not self._compiled:
            self._compile_with_proximal_loss()

    def fit(self, *args, **kwargs):
        return self.base_model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.base_model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.base_model.evaluate(*args, **kwargs)

    def get_weights(self):
        return self.base_model.get_weights()

def create_fedprox_dense_model(input_dim, num_classes, batch_size, fedprox_strategy):
    from .dense import create_enhanced_dense_model
    base_model = create_enhanced_dense_model(input_dim, num_classes, batch_size)
    return FedProxModelWrapper(base_model, fedprox_strategy)
