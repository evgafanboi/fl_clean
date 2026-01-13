import numpy as np
import tensorflow as tf


def _build_block(units, dropout_rate, name_prefix, trainable=True, residual=False):
    dense = tf.keras.layers.Dense(
        units,
        activation="swish",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=f"{name_prefix}_dense",
        trainable=trainable,
    )
    norm = tf.keras.layers.LayerNormalization(name=f"{name_prefix}_ln", trainable=trainable)
    dropout = tf.keras.layers.Dropout(dropout_rate, name=f"{name_prefix}_drop")
    return {
        "dense": dense,
        "norm": norm,
        "drop": dropout,
        "residual": residual,
    }


class FedMLBNetwork(tf.keras.Model):
    def __init__(self, input_dim, num_classes, lambda1, lambda2, temperature):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.temperature = temperature

        self.input_norm = tf.keras.layers.LayerNormalization(name="input_ln")

        self.block1_local = _build_block(128, 0.15, "block1_local", trainable=True, residual=False)
        self.block2_local = _build_block(128, 0.20, "block2_local", trainable=True, residual=True)
        self.block3_local = _build_block(64, 0.15, "block3_local", trainable=True, residual=False)

        self.block2_global = _build_block(128, 0.20, "block2_global", trainable=False, residual=True)
        self.block3_global = _build_block(64, 0.15, "block3_global", trainable=False, residual=False)

        self.logits_local = tf.keras.layers.Dense(num_classes, name="logits_local")
        self.softmax_local = tf.keras.layers.Activation("softmax", name="softmax_local")

        self.logits_global = tf.keras.layers.Dense(num_classes, name="logits_global", trainable=False)
        self.softmax_global = tf.keras.layers.Activation("softmax", name="softmax_global")

        self.ce = tf.keras.losses.CategoricalCrossentropy()
        self.kl = tf.keras.losses.KLDivergence()
        self.accuracy = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss_main_tracker = tf.keras.metrics.Mean(name="loss_main")
        self.loss_hybrid_ce_tracker = tf.keras.metrics.Mean(name="loss_hybrid_ce")
        self.loss_hybrid_kl_tracker = tf.keras.metrics.Mean(name="loss_hybrid_kl")

        self.input_dim = input_dim
        self.num_classes = num_classes

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.loss_main_tracker,
            self.loss_hybrid_ce_tracker,
            self.loss_hybrid_kl_tracker,
            self.accuracy,
        ]

    def call(self, inputs, training=False):
        _, _, _, probs = self._forward_local(inputs, training)
        return probs

    def _apply_block(self, block, inputs, training):
        x = block["dense"](inputs)
        x = block["norm"](x, training=training)
        x = block["drop"](x, training=training)
        if block["residual"]:
            x = x + inputs
        return x

    def _forward_local(self, inputs, training):
        x = self.input_norm(inputs, training=training)
        z1 = self._apply_block(self.block1_local, x, training)
        z2 = self._apply_block(self.block2_local, z1, training)
        z3 = self._apply_block(self.block3_local, z2, training)
        logits = self.logits_local(z3)
        probs = self.softmax_local(logits)
        return z1, z2, logits, probs

    def _forward_hybrid_from_block2(self, z1):
        h = self._apply_block(self.block2_global, z1, training=False)
        return self._forward_hybrid_from_block3(h)

    def _forward_hybrid_from_block3(self, z2):
        h = self._apply_block(self.block3_global, z2, training=False)
        logits = self.logits_global(h)
        probs = self.softmax_global(logits)
        return logits, probs

    def sync_hybrid_weights(self):
        self._copy_block_weights(self.block2_local, self.block2_global)
        self._copy_block_weights(self.block3_local, self.block3_global)
        self.logits_global.set_weights(self.logits_local.get_weights())

    def _copy_block_weights(self, src, dst):
        dst["dense"].set_weights(src["dense"].get_weights())
        dst["norm"].set_weights(src["norm"].get_weights())

    @tf.function(jit_compile=True)
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            z1, z2, logits_L, probs_L = self._forward_local(x, training=True)
            logits_H1, probs_H1 = self._forward_hybrid_from_block2(z1)
            logits_H2, probs_H2 = self._forward_hybrid_from_block3(z2)

            loss_main = self.ce(y, probs_L)
            loss_hybrid_ce = (self.ce(y, probs_H1) + self.ce(y, probs_H2)) / 2.0

            q_L_temp = tf.nn.softmax(logits_L / self.temperature, axis=-1)
            q_H1_temp = tf.nn.softmax(logits_H1 / self.temperature, axis=-1)
            q_H2_temp = tf.nn.softmax(logits_H2 / self.temperature, axis=-1)
            loss_hybrid_kl = (self.kl(q_H1_temp, q_L_temp) + self.kl(q_H2_temp, q_L_temp)) / 2.0

            total_loss = loss_main + self.lambda1 * loss_hybrid_ce + self.lambda2 * loss_hybrid_kl

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(total_loss)
        self.loss_main_tracker.update_state(loss_main)
        self.loss_hybrid_ce_tracker.update_state(loss_hybrid_ce)
        self.loss_hybrid_kl_tracker.update_state(loss_hybrid_kl)
        self.accuracy.update_state(y, probs_L)

        return {
            "loss": self.loss_tracker.result(),
            "loss_main": self.loss_main_tracker.result(),
            "loss_hybrid_ce": self.loss_hybrid_ce_tracker.result(),
            "loss_hybrid_kl": self.loss_hybrid_kl_tracker.result(),
            "accuracy": self.accuracy.result(),
        }

    @tf.function(jit_compile=True)
    def test_step(self, data):
        x, y = data
        _, _, _, probs_L = self._forward_local(x, training=False)
        loss = self.ce(y, probs_L)
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(y, probs_L)
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy.result()}


class FedMLBModel:
    def __init__(self, input_dim, num_classes, batch_size, lambda1, lambda2, temperature):
        self.network = FedMLBNetwork(input_dim, num_classes, lambda1, lambda2, temperature)
        self.base_model = self.network
        base_lr = 0.001
        learning_rate = base_lr * np.sqrt(batch_size / 1024)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5)
        self.network.compile(optimizer=optimizer)

        dummy_x = tf.constant(np.zeros((1, input_dim), dtype=np.float32))
        z1, z2, _, _ = self.network._forward_local(dummy_x, training=False)
        self.network._forward_hybrid_from_block2(z1)
        self.network._forward_hybrid_from_block3(z2)
        self.network.sync_hybrid_weights()

    def fit(self, *args, **kwargs):
        return self.network.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.network.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.network.predict(*args, **kwargs)

    def get_weights(self):
        return self.network.get_weights()

    def set_weights(self, weights):
        self.network.set_weights(weights)
        self.network.sync_hybrid_weights()

    def set_global_weights(self, _):
        self.network.sync_hybrid_weights()


def create_fedmlb_model(input_dim, num_classes, batch_size, lambda1=1.0, lambda2=1.0, temperature=1.0):
    return FedMLBModel(input_dim, num_classes, batch_size, lambda1, lambda2, temperature)
