import tensorflow as tf
import tensorflow_hub as hub


class MyCNN(tf.keras.Model):
    def __init__(self, n_labels):
        super(MyCNN, self).__init__()

        self.fltr = [32, 64, 128]
        self.bn = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.d_1 = tf.keras.layers.Dense(256, activation='relu')
        self.classifier = tf.keras.layers.Dense(n_labels, activation='softmax')

        for i, fltr in enumerate(self.fltr):
            vars(self)[f'conv_{i}'] = tf.keras.layers.Conv2D(fltr, 3,
                                                             activation='relu')
            vars(self)[f'mp_{i}'] = tf.keras.layers.MaxPooling2D()

    def call(self, inputs):
        x = self.bn(inputs)

        for i in range(len(self.fltr)):
            conv_i = vars(self)[f'conv_{i}']
            mp_i = vars(self)[f'mp_{i}']

            x = conv_i(x)
            x = mp_i(x)

        x = self.flatten(x)
        x = self.d_1(x)
        x = self.classifier(x)

        return x

    def model(self):
        # This part is required for getting all layers in model.summary() call
        # Instantiate the model class and apply model() method (i.e. model.model().summary())
        # From https://stackoverflow.com/questions/60416449/plot-model-doesnt-show-layers-of-model-only-the-model-name
        x = tf.keras.layers.Input(shape=(250, 250, 3))

        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class InceptionV3(tf.keras.Model):

    def __init__(self, n_labels):
        super(InceptionV3, self).__init__()

        self.inception = hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/classification/4",
                                        trainable=False)
        self.d_1 = tf.keras.layers.Dense(256, activation='relu')
        self.d_2 = tf.keras.layers.Dense(128, activation='relu')
        self.classifier = tf.keras.layers.Dense(n_labels, activation='softmax')

    def call(self, inputs):
        x = self.inception(inputs)
        x = self.d_1(x)
        x = self.d_2(x)
        x = self.classifier(x)

        return x

    def model(self):
        # This part is required for getting all layers in model.summary() call
        # Instantiate the model class and apply model() method (i.e. model.model().summary())
        # From https://stackoverflow.com/questions/60416449/plot-model-doesnt-show-layers-of-model-only-the-model-name
        x = tf.keras.layers.Input(shape=(250, 250, 3))

        return tf.keras.Model(inputs=[x], outputs=self.call(x))



