import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

def recall_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class DenseModel:
    """Enhanced Dense Model matching GRU's architectural principles"""
    
    def __init__(self, input_dim, num_classes, batch_size=4096, learning_rate=None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        # Auto-scale learning rate based on batch size
        if learning_rate is None:
            base_lr = 0.001
            self.learning_rate = base_lr * np.sqrt(batch_size / 1024)
        else:
            self.learning_rate = learning_rate
            
        print(f"Dense Model - Using learning rate: {self.learning_rate:.6f} for batch size: {batch_size}")
        
        # Create the Dense model
        self.model = self._create_dense_model()
    
    def _create_dense_model(self):
        """Create dense model with architecture matching GRU principles"""
        inputs = tf.keras.layers.Input(shape=(self.input_dim,))
        
        x = tf.keras.layers.LayerNormalization()(inputs)
        
        # First dense block
        x = tf.keras.layers.Dense(
            128,
            activation='swish',  # Same activation
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name='dense_1'
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        
        # Second dense block
        residual = x
        x = tf.keras.layers.Dense(
            128,
            activation='swish',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name='dense_2'
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Residual
        x = tf.keras.layers.add([x, residual])
        
        # Third dense block
        x = tf.keras.layers.Dense(
            64, 
            activation='swish',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name='dense_3'
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        
        # Logits layer 
        logits = tf.keras.layers.Dense(
            self.num_classes, 
            activation=None,  # No activation = raw logits
            name='logits'
        )(x)
        
        # Apply softmax for final predictions
        outputs = tf.keras.layers.Activation('softmax', name='predictions')(logits)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        if self.batch_size >= 2048:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-6,
                clipnorm=0.5
            )
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=0.5
            )
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=0.05
            ),
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
        
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1))
        
        if validation_data is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ))
            
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ))
        
        return callbacks
    
    def fit(self, dataset, epochs=50, validation_data=None, **kwargs):
        kwargs.pop('verbose', None)
        kwargs.pop('callbacks', None)
        
        callbacks = self.get_callbacks(validation_data)
        
        return self.model.fit(
            dataset,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
            **kwargs
        )
    
    def predict(self, X, verbose=None):
        if hasattr(self.model, 'predict'):
            return self.model.predict(X, verbose=verbose)
        else:
            return self.model(X)

    def predict_proba(self, X, verbose=None):
        if hasattr(self.model, 'predict'):
            return self.model.predict(X, verbose=verbose)
        else:
            return self.model(X)
    
    def evaluate(self, dataset, verbose=0):
        return self.model.evaluate(dataset, verbose=verbose)
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)
    
    def get_logits_model(self):
        logits_layer = self.model.get_layer('logits')
        logits_model = tf.keras.Model(inputs=self.model.input, outputs=logits_layer.output)
        return logits_model

def create_enhanced_dense_model(input_dim, num_classes, batch_size=4096, learning_rate=None):
    """Create enhanced dense model that matches GRU performance"""
    return DenseModel(input_dim, num_classes, batch_size, learning_rate)

def create_dense_model(input_dim, num_classes, batch_size=1024):
    """Legacy function - creates enhanced model with default batch size"""
    return create_enhanced_dense_model(input_dim, num_classes, batch_size=batch_size)