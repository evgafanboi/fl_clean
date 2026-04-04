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
        return self._hybrid_head(h)

    def _forward_hybrid_from_block3(self, z2):
        h = self._apply_block(self.block3_global, z2, training=False)
        return self._hybrid_head(h)

    def _hybrid_head(self, h):
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


# ---------------------------------------------------------------------------
# GRU FedMLB wrapper — 4 local blocks, 3 hybrid pathways
# Blocks: GRU1 | GRU2 | GRU3 | DenseHead
# ---------------------------------------------------------------------------

def _build_gru_block(units, return_sequences, name, trainable=True):
    return {
        "gru": tf.keras.layers.GRU(
            units, return_sequences=return_sequences,
            name=name, dtype="float32", trainable=trainable,
        ),
        "ln": tf.keras.layers.LayerNormalization(name=f"ln_{name}", trainable=trainable),
        "drop": tf.keras.layers.Dropout(0.15, name=f"drop_{name}"),
    }


def _build_dense_head_block(units, dropout_rate, name_prefix, trainable=True):
    return {
        "dense": tf.keras.layers.Dense(
            units, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name=f"{name_prefix}_dense", trainable=trainable,
        ),
        "ln": tf.keras.layers.LayerNormalization(name=f"{name_prefix}_ln", trainable=trainable),
        "drop": tf.keras.layers.Dropout(dropout_rate, name=f"{name_prefix}_drop"),
    }


class FedMLBGRUNetwork(tf.keras.Model):
    def __init__(self, input_dim, num_classes, lambda1, lambda2, temperature, gru_units=128):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.temperature = temperature
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.reshape = tf.keras.layers.Reshape((input_dim, 1), name="reshape_input")

        self.gru1_local = _build_gru_block(gru_units, True, "gru1_local")
        self.gru2_local = _build_gru_block(gru_units, True, "gru2_local")
        self.gru3_local = _build_gru_block(gru_units, False, "gru3_local")
        self.head_local = _build_dense_head_block(64, 0.2, "head_local")

        self.gru2_global = _build_gru_block(gru_units, True, "gru2_global", trainable=False)
        self.gru3_global = _build_gru_block(gru_units, False, "gru3_global", trainable=False)
        self.head_global = _build_dense_head_block(64, 0.2, "head_global", trainable=False)

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

    @property
    def metrics(self):
        return [self.loss_tracker, self.loss_main_tracker,
                self.loss_hybrid_ce_tracker, self.loss_hybrid_kl_tracker,
                self.accuracy]

    def _apply_gru_block(self, block, x, training):
        x = block["gru"](x, training=training)
        x = block["ln"](x, training=training)
        x = block["drop"](x, training=training)
        return x

    def _apply_dense_head(self, block, x, training):
        x = block["dense"](x)
        x = block["ln"](x, training=training)
        x = block["drop"](x, training=training)
        return x

    def _forward_local(self, inputs, training):
        x = self.reshape(inputs)
        z1 = self._apply_gru_block(self.gru1_local, x, training)
        z2 = self._apply_gru_block(self.gru2_local, z1, training)
        z3 = self._apply_gru_block(self.gru3_local, z2, training)
        z4 = self._apply_dense_head(self.head_local, z3, training)
        logits = self.logits_local(z4)
        probs = self.softmax_local(logits)
        return z1, z2, z3, logits, probs

    def call(self, inputs, training=False):
        _, _, _, _, probs = self._forward_local(inputs, training)
        return probs

    def _hybrid_head(self, h):
        logits = self.logits_global(h)
        probs = self.softmax_global(logits)
        return logits, probs

    def _forward_hybrid_from_gru2(self, z1):
        h = self._apply_gru_block(self.gru2_global, z1, training=False)
        return self._forward_hybrid_from_gru3(h)

    def _forward_hybrid_from_gru3(self, z2):
        h = self._apply_gru_block(self.gru3_global, z2, training=False)
        return self._forward_hybrid_from_head(h)

    def _forward_hybrid_from_head(self, z3):
        h = self._apply_dense_head(self.head_global, z3, training=False)
        return self._hybrid_head(h)

    def sync_hybrid_weights(self):
        for src, dst in [
            (self.gru2_local, self.gru2_global),
            (self.gru3_local, self.gru3_global),
            (self.head_local, self.head_global),
        ]:
            for key in ("gru", "dense", "ln"):
                if key in src and key in dst:
                    dst[key].set_weights(src[key].get_weights())
        self.logits_global.set_weights(self.logits_local.get_weights())

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z1, z2, z3, logits_L, probs_L = self._forward_local(x, training=True)
            logits_H1, probs_H1 = self._forward_hybrid_from_gru2(z1)
            logits_H2, probs_H2 = self._forward_hybrid_from_gru3(z2)
            logits_H3, probs_H3 = self._forward_hybrid_from_head(z3)

            loss_main = self.ce(y, probs_L)
            loss_hybrid_ce = (self.ce(y, probs_H1) + self.ce(y, probs_H2) + self.ce(y, probs_H3)) / 3.0

            q_L = tf.nn.softmax(logits_L / self.temperature, axis=-1)
            q_H1 = tf.nn.softmax(logits_H1 / self.temperature, axis=-1)
            q_H2 = tf.nn.softmax(logits_H2 / self.temperature, axis=-1)
            q_H3 = tf.nn.softmax(logits_H3 / self.temperature, axis=-1)
            loss_hybrid_kl = (self.kl(q_H1, q_L) + self.kl(q_H2, q_L) + self.kl(q_H3, q_L)) / 3.0

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

    @tf.function
    def test_step(self, data):
        x, y = data
        _, _, _, _, probs_L = self._forward_local(x, training=False)
        loss = self.ce(y, probs_L)
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(y, probs_L)
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy.result()}


class FedMLBGRUModel:
    def __init__(self, input_dim, num_classes, batch_size, lambda1, lambda2, temperature, gru_units=128):
        self.network = FedMLBGRUNetwork(input_dim, num_classes, lambda1, lambda2, temperature, gru_units)
        self.base_model = self.network
        base_lr = 0.001
        learning_rate = base_lr * np.sqrt(batch_size / 1024)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5)
        self.network.compile(optimizer=optimizer)

        dummy_x = tf.constant(np.zeros((1, input_dim), dtype=np.float32))
        z1, z2, z3, _, _ = self.network._forward_local(dummy_x, training=False)
        self.network._forward_hybrid_from_gru2(z1)
        self.network._forward_hybrid_from_gru3(z2)
        self.network._forward_hybrid_from_head(z3)
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


def create_fedmlb_gru_model(input_dim, num_classes, batch_size, lambda1=1.0, lambda2=1.0, temperature=1.0, gru_units=128):
    return FedMLBGRUModel(input_dim, num_classes, batch_size, lambda1, lambda2, temperature, gru_units)


# ---------------------------------------------------------------------------
# DCBLSTM FedMLB wrapper — 5 local blocks, 4 hybrid pathways
# Blocks: Conv1D | BiLSTM | Dense1 | Dense2 | Dense3
# ---------------------------------------------------------------------------

def _build_conv_block(filters, kernel_size, name, trainable=True):
    return {
        "conv": tf.keras.layers.Conv1D(
            filters, kernel_size=kernel_size, activation="relu",
            padding="same", name=f"{name}_conv", trainable=trainable,
        ),
        "ln": tf.keras.layers.LayerNormalization(name=f"{name}_ln", trainable=trainable),
    }


def _build_bilstm_block(units1, units2, name, trainable=True):
    return {
        "bilstm1": tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units1, return_sequences=True, dtype="float32", trainable=trainable),
            name=f"{name}_bilstm1",
        ),
        "ln1": tf.keras.layers.LayerNormalization(name=f"{name}_ln1", trainable=trainable),
        "bilstm2": tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units2, return_sequences=False, dtype="float32", trainable=trainable),
            name=f"{name}_bilstm2",
        ),
        "drop": tf.keras.layers.Dropout(0.1, name=f"{name}_drop"),
    }


def _build_mlp_block(units, dropout_rate, name, trainable=True, has_ln=False):
    d = {
        "dense": tf.keras.layers.Dense(units, activation="relu", name=f"{name}_dense", trainable=trainable),
        "drop": tf.keras.layers.Dropout(dropout_rate, name=f"{name}_drop"),
    }
    if has_ln:
        d["ln"] = tf.keras.layers.LayerNormalization(name=f"{name}_ln", trainable=trainable)
    return d


class FedMLBDCBLSTMNetwork(tf.keras.Model):
    def __init__(self, input_dim, num_classes, lambda1, lambda2, temperature,
                 conv_filters=64, lstm_units=64, lstm_units_2=128, dnn_sizes=(64, 32, 16)):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.temperature = temperature
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.reshape = tf.keras.layers.Reshape((input_dim, 1), name="reshape_input")

        self.conv_local = _build_conv_block(conv_filters, input_dim, "conv_local")
        self.bilstm_local = _build_bilstm_block(lstm_units, lstm_units_2, "bilstm_local")
        self.mlp1_local = _build_mlp_block(dnn_sizes[0], 0.1, "mlp1_local")
        self.mlp2_local = _build_mlp_block(dnn_sizes[1], 0.1, "mlp2_local")
        self.mlp3_local = _build_mlp_block(dnn_sizes[2], 0.1, "mlp3_local", has_ln=True)

        self.bilstm_global = _build_bilstm_block(lstm_units, lstm_units_2, "bilstm_global", trainable=False)
        self.mlp1_global = _build_mlp_block(dnn_sizes[0], 0.1, "mlp1_global", trainable=False)
        self.mlp2_global = _build_mlp_block(dnn_sizes[1], 0.1, "mlp2_global", trainable=False)
        self.mlp3_global = _build_mlp_block(dnn_sizes[2], 0.1, "mlp3_global", trainable=False, has_ln=True)

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

    @property
    def metrics(self):
        return [self.loss_tracker, self.loss_main_tracker,
                self.loss_hybrid_ce_tracker, self.loss_hybrid_kl_tracker,
                self.accuracy]

    def _apply_conv_block(self, block, x, training):
        x = block["conv"](x)
        x = block["ln"](x, training=training)
        return x

    def _apply_bilstm_block(self, block, x, training):
        x = block["bilstm1"](x, training=training)
        x = block["ln1"](x, training=training)
        x = block["bilstm2"](x, training=training)
        x = block["drop"](x, training=training)
        return x

    def _apply_mlp_block(self, block, x, training):
        x = block["dense"](x)
        x = block["drop"](x, training=training)
        if "ln" in block:
            x = block["ln"](x, training=training)
        return x

    def _forward_local(self, inputs, training):
        x = self.reshape(inputs)
        z1 = self._apply_conv_block(self.conv_local, x, training)
        z2 = self._apply_bilstm_block(self.bilstm_local, z1, training)
        z3 = self._apply_mlp_block(self.mlp1_local, z2, training)
        z4 = self._apply_mlp_block(self.mlp2_local, z3, training)
        z5 = self._apply_mlp_block(self.mlp3_local, z4, training)
        logits = self.logits_local(z5)
        probs = self.softmax_local(logits)
        return z1, z2, z3, z4, logits, probs

    def call(self, inputs, training=False):
        _, _, _, _, _, probs = self._forward_local(inputs, training)
        return probs

    def _hybrid_head(self, h):
        logits = self.logits_global(h)
        probs = self.softmax_global(logits)
        return logits, probs

    def _forward_hybrid_from_bilstm(self, z1):
        h = self._apply_bilstm_block(self.bilstm_global, z1, training=False)
        return self._forward_hybrid_from_mlp1(h)

    def _forward_hybrid_from_mlp1(self, z2):
        h = self._apply_mlp_block(self.mlp1_global, z2, training=False)
        return self._forward_hybrid_from_mlp2(h)

    def _forward_hybrid_from_mlp2(self, z3):
        h = self._apply_mlp_block(self.mlp2_global, z3, training=False)
        return self._forward_hybrid_from_mlp3(h)

    def _forward_hybrid_from_mlp3(self, z4):
        h = self._apply_mlp_block(self.mlp3_global, z4, training=False)
        return self._hybrid_head(h)

    def sync_hybrid_weights(self):
        for src, dst in [
            (self.bilstm_local, self.bilstm_global),
            (self.mlp1_local, self.mlp1_global),
            (self.mlp2_local, self.mlp2_global),
            (self.mlp3_local, self.mlp3_global),
        ]:
            for key in src:
                if key in dst and hasattr(src[key], "get_weights") and src[key].get_weights():
                    dst[key].set_weights(src[key].get_weights())
        self.logits_global.set_weights(self.logits_local.get_weights())

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z1, z2, z3, z4, logits_L, probs_L = self._forward_local(x, training=True)
            logits_H1, probs_H1 = self._forward_hybrid_from_bilstm(z1)
            logits_H2, probs_H2 = self._forward_hybrid_from_mlp1(z2)
            logits_H3, probs_H3 = self._forward_hybrid_from_mlp2(z3)
            logits_H4, probs_H4 = self._forward_hybrid_from_mlp3(z4)

            loss_main = self.ce(y, probs_L)
            loss_hybrid_ce = (
                self.ce(y, probs_H1) + self.ce(y, probs_H2) +
                self.ce(y, probs_H3) + self.ce(y, probs_H4)
            ) / 4.0

            q_L = tf.nn.softmax(logits_L / self.temperature, axis=-1)
            q_H1 = tf.nn.softmax(logits_H1 / self.temperature, axis=-1)
            q_H2 = tf.nn.softmax(logits_H2 / self.temperature, axis=-1)
            q_H3 = tf.nn.softmax(logits_H3 / self.temperature, axis=-1)
            q_H4 = tf.nn.softmax(logits_H4 / self.temperature, axis=-1)
            loss_hybrid_kl = (
                self.kl(q_H1, q_L) + self.kl(q_H2, q_L) +
                self.kl(q_H3, q_L) + self.kl(q_H4, q_L)
            ) / 4.0

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

    @tf.function
    def test_step(self, data):
        x, y = data
        _, _, _, _, _, probs_L = self._forward_local(x, training=False)
        loss = self.ce(y, probs_L)
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(y, probs_L)
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy.result()}


class FedMLBDCBLSTMModel:
    def __init__(self, input_dim, num_classes, batch_size, lambda1, lambda2, temperature):
        self.network = FedMLBDCBLSTMNetwork(input_dim, num_classes, lambda1, lambda2, temperature)
        self.base_model = self.network
        base_lr = 0.001
        learning_rate = base_lr * np.sqrt(batch_size / 1024)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5)
        self.network.compile(optimizer=optimizer)

        dummy_x = tf.constant(np.zeros((1, input_dim), dtype=np.float32))
        z1, z2, z3, z4, _, _ = self.network._forward_local(dummy_x, training=False)
        self.network._forward_hybrid_from_bilstm(z1)
        self.network._forward_hybrid_from_mlp1(z2)
        self.network._forward_hybrid_from_mlp2(z3)
        self.network._forward_hybrid_from_mlp3(z4)
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


def create_fedmlb_dcblstm_model(input_dim, num_classes, batch_size, lambda1=1.0, lambda2=1.0, temperature=1.0):
    return FedMLBDCBLSTMModel(input_dim, num_classes, batch_size, lambda1, lambda2, temperature)