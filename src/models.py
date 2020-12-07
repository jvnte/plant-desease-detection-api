from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16, MobileNet
from tensorflow.keras.layers import (
    Input,
    BatchNormalization,
    MaxPooling2D,
    Conv2D,
    Flatten,
    Dense,
    Dropout
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


def vgg16(n_labels, input_shape):
    vgg16_layer = VGG16(weights='imagenet',
                        include_top=False,
                        input_shape=input_shape)
    vgg16_layer.trainable = False

    inputs = Input(shape=input_shape)
    x = vgg16_layer(inputs)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(n_labels, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)


def mobile_net(n_labels, input_shape):
    mobilenet_layer = MobileNet(weights='imagenet',
                                include_top=False,
                                input_shape=input_shape)
    mobilenet_layer.trainable = False

    inputs = Input(shape=input_shape)
    x = mobilenet_layer(inputs)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(n_labels, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)
