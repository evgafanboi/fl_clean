import gc
import tensorflow as tf
import numpy as np

from models.gru import precision_m, recall_m, f1_m


class DCBLSTMModel:
    """DDCNNBiLSTM: Conv1D + BiLSTM + DNN block (softmax)"""

    def __init__(
        self,
        input_dim=20,
        num_classes=20,
        batch_size=4096,
        learning_rate=None,
        conv_filters=64,
        lstm_units=64,
        lstm_units_2=128,
        dnn_sizes=(64, 32, 16),
        compile=True,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.conv_filters = conv_filters
        self.lstm_units = lstm_units
        self.lstm_units_2 = lstm_units_2
        self.dnn_sizes = dnn_sizes
        if learning_rate is None:
            base_lr = 0.001
            self.learning_rate = base_lr * np.sqrt(batch_size / 1024)
        else:
            self.learning_rate = learning_rate

        print(f"DCBLSTM Model - Using learning rate: {self.learning_rate:.6f} for batch size: {batch_size}")

        self._do_compile = compile
        self.model = self._create_dcblstm_model()
        self._logits_model = None
        self._feature_model = None

    def _create_dcblstm_model(self):
        inputs = tf.keras.layers.Input(shape=(self.input_dim,))
        x = tf.keras.layers.Reshape((self.input_dim, 1))(inputs)
        x = tf.keras.layers.Conv1D(self.conv_filters, kernel_size=self.input_dim, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.lstm_units, return_sequences=False)
        )(x)
        x = tf.keras.layers.Reshape((self.lstm_units * 2, 1))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.lstm_units_2, return_sequences=False)
        )(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        for i, size in enumerate(self.dnn_sizes, start=1):
            x = tf.keras.layers.Dense(size, activation='relu', name=f'dense_{i}')(x)
            x = tf.keras.layers.Dropout(0.1, name=f'dropout_{i}')(x)
        x = tf.keras.layers.LayerNormalization(name='ln_feature')(x)
        logits = tf.keras.layers.Dense(self.num_classes, activation=None, name='logits')(x)
        outputs = tf.keras.layers.Activation('softmax', name='predictions')(logits)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        if self._do_compile:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=0.5)
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                metrics=['accuracy', precision_m, recall_m, f1_m],
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
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1
                )
            )
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
                )
            )

        return callbacks

    def fit(self, dataset, epochs=50, validation_data=None, **kwargs):
        kwargs.pop('verbose', None)
        kwargs.pop('callbacks', None)
        callbacks = self.get_callbacks(validation_data)
        return self.model.fit(
            dataset, epochs=epochs, validation_data=validation_data, callbacks=callbacks, verbose=1, **kwargs
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
            feature_layer = self.model.get_layer('ln_feature')
            self._feature_model = tf.keras.Model(inputs=self.model.input, outputs=feature_layer.output)
        return self._feature_model

    def expand_classes(self, new_num_classes):
        if new_num_classes <= self.num_classes:
            return
        old_model = self.model
        old_num = self.num_classes
        self.num_classes = new_num_classes
        self.model = self._create_dcblstm_model()
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


def create_dcblstm_model(input_dim=20, num_classes=20, batch_size=4096, learning_rate=None, compile=True):
    return DCBLSTMModel(input_dim=input_dim, num_classes=num_classes, batch_size=batch_size, learning_rate=learning_rate, compile=compile)
