#@ Implementation of sparse autoencoder
import tensorflow as tf

# Encoder with Sparsity Regularization
sparse_l1_encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(),                             # Flatten input data to 1D
    tf.keras.layers.Dense(100, activation="relu"),         # Dense layer with ReLU activation
    tf.keras.layers.Dense(300, activation="sigmoid"),      # Dense layer with sigmoid activation
    tf.keras.layers.ActivityRegularization(l1=1e-4)        # L1 regularization for sparsity
])

# Normal Decoder
sparse_l1_decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),         # Dense layer with ReLU activation
    tf.keras.layers.Dense(28 * 28),                        # Dense layer for 1D reconstruction
    tf.keras.layers.Reshape([28, 28])                      # Reshape back to image format
])

# Combine the encoder and decoder to create the Sparse Autoencoder model
sparse_l1_ae = tf.keras.Sequential([sparse_l1_encoder, sparse_l1_decoder])  # Sparse autoencoder ready to be compiled
