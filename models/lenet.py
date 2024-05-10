import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def LeNet5(input_shape, num_classes):
    model = tf.keras.Sequential([
        # First Convolutional Layer
        Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2), strides=2),

        # Second Convolutional Layer
        Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=2),

        # Flatten Layer
        Flatten(),

        Dense(120, activation='relu'),
        Dropout(0.5),  # Use a dropout rate of 0.5

        # Second Fully Connected Layer
        Dense(84, activation='relu'),
        Dropout(0.5),  # Dropout rate of 0.5

        # Output Layer
        Dense(num_classes, activation='softmax')
    ])

    return model