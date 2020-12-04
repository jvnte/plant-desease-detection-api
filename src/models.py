import tensorflow_hub as hub
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    BatchNormalization,
    MaxPooling2D,
    Conv2D,
    Flatten,
    Dense
)


def my_cnn(n_labels, input_shape):
    inputs = Input(shape=input_shape)
    x = BatchNormalization()(inputs)
    x = Conv2D(32, 3, activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, 3, activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, 3, activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(n_labels, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)


def inception_v3(n_labels, input_shape):
    inputs = Input(shape=input_shape)
    x = hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/classification/4",
                       trainable=False)(inputs)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(n_labels, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)
