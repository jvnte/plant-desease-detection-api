import tensorflow as tf

from datetime import datetime
from src.models import my_cnn, vgg16, mobile_net

N_LABELS = 38
EPOCHS = 20
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'
METRICS = 'accuracy'
METHODS = ['my_cnn', 'vgg16', 'mobile_net']


class Train:
    def __init__(self,
                 method,
                 batch_size,
                 input_shape,
                 save=True):
        self.method = method
        if self.method not in METHODS:
            raise NotImplementedError(f'Method {self.method} not implemented')
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.loss = LOSS
        self.metrics = METRICS
        self.epochs = EPOCHS
        self.optimizer = OPTIMIZER
        self.train_ds, self.validation_ds = self.create_datasets()
        self.model = self.build()
        self.fit(save)

    def create_datasets(self):

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

        train_ds = datagen.flow_from_directory('./dataset/train',
                                               target_size=self.input_shape[:2],
                                               batch_size=self.batch_size,
                                               class_mode='categorical')

        validation_ds = datagen.flow_from_directory('./dataset/valid',
                                                    target_size=self.input_shape[:2],
                                                    batch_size=self.batch_size,
                                                    class_mode='categorical')

        return train_ds, validation_ds

    def build(self):
        if self.method == 'vgg16':
            model = vgg16(N_LABELS, self.input_shape)
        elif self.method == 'my_cnn':
            model = my_cnn(N_LABELS, self.input_shape)
        elif self.method == 'mobile_net':
            model = mobile_net(N_LABELS, self.input_shape)

        # Compile the model
        model.compile(self.optimizer, loss=self.loss, metrics=[self.metrics])

        return model

    def fit(self, save):

        # Tensorboard callback
        log_dir = 'logs/' + self.method + datetime.now().strftime("%Y%m%d-%H%M%S")
        tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Early stopping callback
        es = tf.keras.callbacks.EarlyStopping(monitor="accuracy", patience=4, restore_best_weights=True, mode='max',
                                              verbose=1)

        # Learning rate reduction callback
        rl = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)

        # Fit model
        model = self.model
        model.fit(x=self.train_ds,
                  validation_data=self.validation_ds,
                  epochs=self.epochs,
                  callbacks=[es, rl, tb])

        if save:
            model.save(f'./models/{self.method}')

        return None
