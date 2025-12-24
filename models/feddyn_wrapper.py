import tensorflow as tf
import numpy as np


class FedDynModelWrapper:
    """
    Wrap a Keras model to implement FedDyn's client objective (Paper Eq. 1):
    
    L_k(θ) - <∇L_k(θ_k^{t-1}), θ> + (α/2)||θ - θ^{t-1}||²
    
    Expanding the L2 term (ignoring constants):
    = L_k(θ) - <∇L_k(θ_k^{t-1}), θ> + (α/2)||θ||² - α<θ, θ^{t-1}>
    = L_k(θ) + (α/2)||θ||² + <θ, -∇L_k(θ_k^{t-1}) - α*θ^{t-1}>
    
    So we add:
    - Weight decay: α/2 (via optimizer)
    - Linear term: <θ, -∇L_k(θ_k^{t-1}) - α*θ^{t-1}>
    """

    def __init__(self, base_model, feddyn_strategy, client_id):
        self.base_model = base_model
        self.model = base_model.model if hasattr(base_model, 'model') else base_model
        self.feddyn = feddyn_strategy
        self.client_id = client_id
        self._compiled = False

    def _compute_linear_term(self):
        """Compute <θ, -∇L_k(θ_k^{t-1}) - α*θ^{t-1}>."""
        current_weights = self.model.get_weights()
        
        # Get ∇L_k(θ_k^{t-1}) and θ^{t-1} from strategy
        grad_L_prev = self.feddyn.get_grad_L_for_client(self.client_id, current_weights)
        theta_prev = self.feddyn.get_prev_global(current_weights)
        alpha = self.feddyn.get_alpha()
        
        # Convert to TF constants: -∇L_k - α*θ^{t-1}
        linear_coef_tf = [tf.constant(-grad_L_prev[j] - alpha * theta_prev[j], dtype=tf.float32) 
                         for j in range(len(grad_L_prev))]
        all_vars = self.model.weights
        pair_count = min(len(all_vars), len(linear_coef_tf))

        def linear_term():
            total = tf.constant(0.0, dtype=tf.float32)
            for idx in range(pair_count):
                var = all_vars[idx]
                coef = linear_coef_tf[idx]
                total += tf.reduce_sum(tf.reshape(var, (-1,)) * tf.reshape(coef, (-1,)))
            return total

        return linear_term

    def _compile_with_feddyn_loss(self):
        """Recompile model with FedDyn regularization."""
        original_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
        linear_term_fn = self._compute_linear_term()

        def feddyn_loss(y_true, y_pred):
            ce = original_loss(y_true, y_pred)
            return ce + linear_term_fn()

        # Add α/2 weight decay via optimizer
        # Note: Keras weight_decay in optimizer is NOT the same as L2 regularization
        # We'll add it manually in the loss instead
        alpha = self.feddyn.get_alpha()
        
        def feddyn_loss_with_wd(y_true, y_pred):
            ce = original_loss(y_true, y_pred)
            linear = linear_term_fn()
            
            # Add α/2 * ||θ||² weight decay
            l2_reg = tf.constant(0.0, dtype=tf.float32)
            for var in self.model.trainable_weights:
                l2_reg += tf.reduce_sum(tf.square(var))
            l2_reg = (alpha / 2.0) * l2_reg
            
            return ce + linear + l2_reg

        self.model.compile(optimizer=self.model.optimizer, loss=feddyn_loss_with_wd, metrics=['accuracy'])
        self._compiled = True

    def set_weights(self, weights):
        if hasattr(self.base_model, 'set_weights'):
            self.base_model.set_weights(weights)
        else:
            self.model.set_weights(weights)
        self._compile_with_feddyn_loss()

    def fit(self, *args, **kwargs):
        if not self._compiled:
            self._compile_with_feddyn_loss()
        return self.base_model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.base_model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.base_model.evaluate(*args, **kwargs)

    def get_weights(self):
        return self.base_model.get_weights()
    
    def get_feddyn_update(self):
        """Return weights for FedDyn aggregation."""
        return {'weights': self.get_weights()}


def create_feddyn_dense_model(input_dim, num_classes, batch_size, feddyn_strategy, client_id):
    from .dense import create_enhanced_dense_model
    base_model = create_enhanced_dense_model(input_dim, num_classes, batch_size)
    return FedDynModelWrapper(base_model, feddyn_strategy, client_id)


def create_feddyn_gru_model(input_shape, num_classes, batch_size, feddyn_strategy, client_id):
    from .gru import create_enhanced_gru_model
    base_model = create_enhanced_gru_model(input_shape, num_classes, batch_size)
    return FedDynModelWrapper(base_model, feddyn_strategy, client_id)
    from .gru import create_enhanced_gru_model
    base_model = create_enhanced_gru_model(input_shape, num_classes, batch_size)
    return FedDynModelWrapper(base_model, feddyn_strategy, client_id)
