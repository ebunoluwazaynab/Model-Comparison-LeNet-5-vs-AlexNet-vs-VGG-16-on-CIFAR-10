import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout

def AlexNet(input_shape, num_classes):
    model = tf.keras.Sequential([
        # First Convolutional Layer
        Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=input_shape),

        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Second Convolutional Layer
        Conv2D(filters=128, kernel_size=(3, 3), padding='same'),

        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Third Convolutional Layer
        Conv2D(filters=192, kernel_size=(3, 3), padding='same'),

        Activation('relu'),

        # Fourth Convolutional Layer
        Conv2D(filters=192, kernel_size=(3, 3), padding='same'),

        Activation('relu'),

        # Fifth Convolutional Layer
        Conv2D(filters=128, kernel_size=(3, 3), padding='same'),

        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Flattening Layer
        Flatten(),

        # First Fully Connected Layer
        Dense(1024),
        Activation('relu'),
        Dropout(0.5),

        # Second Fully Connected Layer
        Dense(1024),
        Activation('relu'),
        Dropout(0.5),

        # Output Layer
        Dense(num_classes, activation='softmax')
    ])
    return model