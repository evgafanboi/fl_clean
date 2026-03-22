import gc
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
def recall_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

@register_keras_serializable()
def precision_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

@register_keras_serializable()
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class GRUModel:
    """RNN-GRU Model: Input -> GRU x3 -> Dense MLP block (softmax)

    Input (batch, input_dim) is reshaped to (batch, input_dim, 1) so each
    feature is treated as one timestep.  Three stacked GRU layers extract
    temporal patterns; only the last hidden state feeds a dense head.
    """

    def __init__(self, input_dim, num_classes, batch_size=4096, learning_rate=None, gru_units=128):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.gru_units = gru_units

        if learning_rate is None:
            base_lr = 0.001
            self.learning_rate = base_lr * np.sqrt(batch_size / 1024)
        else:
            self.learning_rate = learning_rate

        print(f"GRU Model - Using learning rate: {self.learning_rate:.6f} for batch size: {batch_size}")

        self.model = self._create_gru_model()
        self._logits_model = None
        self._feature_model = None

    def _create_gru_model(self):
        inputs = tf.keras.layers.Input(shape=(self.input_dim,))

        # Reshape flat features -> (timesteps, 1) for GRU
        x = tf.keras.layers.Reshape((self.input_dim, 1), name='reshape_input')(inputs)

        x = tf.keras.layers.GRU(
            self.gru_units, return_sequences=True,
            name='gru_1', unroll=True
        )(x)
        x = tf.keras.layers.LayerNormalization(name='ln_gru_1')(x)
        x = tf.keras.layers.Dropout(0.15, name='drop_gru_1')(x)

        x = tf.keras.layers.GRU(
            self.gru_units, return_sequences=True,
            name='gru_2', unroll=True
        )(x)
        x = tf.keras.layers.LayerNormalization(name='ln_gru_2')(x)
        x = tf.keras.layers.Dropout(0.15, name='drop_gru_2')(x)

        x = tf.keras.layers.GRU(
            self.gru_units, return_sequences=False,
            name='gru_3', unroll=True
        )(x)
        x = tf.keras.layers.LayerNormalization(name='ln_gru_3')(x)
        x = tf.keras.layers.Dropout(0.15, name='drop_gru_3')(x)

        # Dense MLP head
        x = tf.keras.layers.Dense(
            64, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name='dense_head'
        )(x)
        x = tf.keras.layers.LayerNormalization(name='ln_head')(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        logits = tf.keras.layers.Dense(self.num_classes, activation=None, name='logits')(x)
        outputs = tf.keras.layers.Activation('softmax', name='predictions')(logits)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        if self.batch_size >= 2048:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9, beta_2=0.999, epsilon=1e-6, clipnorm=0.5
            )
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate, clipnorm=0.5
            )

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=['accuracy', precision_m, recall_m, f1_m]
        )

        return model

    def get_callbacks(self, validation_data=None):
        callbacks = []

        def lr_schedule(epoch, lr):
            if epoch < 5:
                return self.learning_rate * (epoch + 1) / 5
            else:
                decay_epochs = max(1, epoch - 5)
                return self.learning_rate * 0.5 * (1 + np.cos(np.pi * decay_epochs / 50))

        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0))

        if validation_data is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=10,
                restore_best_weights=True, verbose=1
            ))
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=5, min_lr=1e-7, verbose=1
            ))

        return callbacks

    def fit(self, dataset, epochs=50, validation_data=None, **kwargs):
        kwargs.pop('verbose', None)
        kwargs.pop('callbacks', None)
        callbacks = self.get_callbacks(validation_data)
        return self.model.fit(
            dataset, epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks, verbose=1, **kwargs
        )

    def predict(self, X, verbose=None):
        if hasattr(self.model, 'predict'):
            return self.model.predict(X, verbose=verbose)
        return self.model(X)

    def predict_proba(self, X, verbose=None):
        if hasattr(self.model, 'predict'):
            return self.model.predict(X, verbose=verbose)
        return self.model(X)

    def evaluate(self, dataset, verbose=0):
        return self.model.evaluate(dataset, verbose=verbose)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_logits_model(self):
        if self._logits_model is None:
            logits_layer = self.model.get_layer('logits')
            self._logits_model = tf.keras.Model(inputs=self.model.input, outputs=logits_layer.output)
        return self._logits_model

    def get_feature_model(self):
        if self._feature_model is None:
            feature_layer = self.model.get_layer('ln_head')
            self._feature_model = tf.keras.Model(inputs=self.model.input, outputs=feature_layer.output)
        return self._feature_model

    def expand_classes(self, new_num_classes):
        if new_num_classes <= self.num_classes:
            return
        old_model = self.model
        old_num = self.num_classes
        self.num_classes = new_num_classes
        self.model = self._create_gru_model()
        self._logits_model = None
        self._feature_model = None
        old_layers = {layer.name: layer for layer in old_model.layers}
        for layer in self.model.layers:
            if layer.name == 'logits':
                old_kernel, old_bias = old_layers['logits'].get_weights()
                new_kernel, new_bias = layer.get_weights()
                new_kernel[:, :old_num] = old_kernel
                new_bias[:old_num] = old_bias
                layer.set_weights([new_kernel, new_bias])
            elif layer.name in old_layers and layer.get_weights():
                layer.set_weights(old_layers[layer.name].get_weights())
        del old_model, old_layers
        gc.collect()


def create_gru_model(input_dim, num_classes, batch_size=4096, learning_rate=None, gru_units=128):
    return GRUModel(input_dim, num_classes, batch_size, learning_rate, gru_units)
