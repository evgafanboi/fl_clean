import tensorflow as tf
import numpy as np

class DenseDiscriminator:
    def __init__(self, input_dim, learning_rate=0.0001):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.model = self._create_model()
    
    def _create_model(self):
        inputs = tf.keras.layers.Input(shape=(self.input_dim,))
        
        x = tf.keras.layers.LayerNormalization()(inputs)
        
        x = tf.keras.layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        
        x = tf.keras.layers.Dense(64, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, dataset, epochs=1, verbose=0):
        return self.model.fit(dataset, epochs=epochs, verbose=verbose)
    
    def predict(self, X, verbose=0):
        return self.model.predict(X, verbose=verbose)
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)

def create_discriminator(input_dim, learning_rate=0.0001):
    return DenseDiscriminator(input_dim, learning_rate)
