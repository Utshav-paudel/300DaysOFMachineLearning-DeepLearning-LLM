#@ Implementation of tied weight autoencoder
import tensorflow as tf
# Custom DenseTranspose layer to implement the decoder part of the tied autoencoder
class DenseTranspose(tf.keras.layers.Layer):                          # Inherited from the keras Layer class     
    def __init__(self, dense, activation=None, **kwargs):             # Constructor with arguments
        super().__init__(**kwargs)                                    # Call the parent class constructor with keyword args
        self.dense = dense                                            # Store the dense layer (encoder) with its weights and shape
        self.activation = tf.keras.activations.get(activation)        # Store the activation function for the decoder

    def build(self, batch_input_shape):                               # Build method to create biases variables
        self.biases = self.add_weight(name="bias",
                                      shape=self.dense.input_shape[-1],
                                      initializer="zeros")
        super().build(batch_input_shape)                              # Call the parent class's build method

    def call(self, inputs):                                           # The decoding process
        Z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)  # Compute matrix multiplication between input and transposed encoder weights
        return self.activation(Z + self.biases)                       # Add biases and apply activation for decoder output

# Define the encoder and decoder architectures for the tied autoencoder
dense_1 = tf.keras.layers.Dense(100, activation="relu")               # First dense layer in the encoder
dense_2 = tf.keras.layers.Dense(30, activation="relu")                # Second dense layer in the encoder

# Create the tied encoder and decoder using the specified architecture
tied_encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(),                                       # Flatten the input image
    dense_1,                                                         # First dense layer in the encoder
    dense_2                                                          # Second dense layer in the encoder
])
tied_decoder = tf.keras.Sequential([
    DenseTranspose(dense_2, activation="relu"),                       # Custom decoder layer using DenseTranspose with activation
    DenseTranspose(dense_1),                                         # Custom decoder layer using DenseTranspose without activation
    tf.keras.layers.Reshape([28, 28])                                # Reshape the decoder output to the original image shape
])

# Create the tied autoencoder by combining the encoder and decoder
tied_ae = tf.keras.Sequential([tied_encoder, tied_decoder])          # Decoder uses the weights of the encoder for reconstruction
