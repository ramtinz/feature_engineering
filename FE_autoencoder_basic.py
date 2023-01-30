import tensorflow as tf
from tensorflow import keras

# Create the input layer with shape equal to number of features in the data
input_layer = keras.layers.Input(shape=(input_shape,))

# Create the encoding layer with desired number of hidden units
encoding_layer = keras.layers.Dense(units=encoding_dim, activation='relu')(input_layer)

# Create the decoding layer with same number of units as the encoding layer
decoding_layer = keras.layers.Dense(units=input_shape, activation='sigmoid')(encoding_layer)

# Create the autoencoder model
autoencoder = keras.models.Model(inputs=input_layer, outputs=decoding_layer)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on the data
autoencoder.fit(X_train, X_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, X_test))

# Get the encoded version of the data
encoded_data = autoencoder.predict(X_test)
