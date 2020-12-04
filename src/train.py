import tensorflow as tf

from datetime import datetime
from src.models import InceptionV3, MyCNN

IMAGE_TARGET_SIZE = (250, 250)
INPUT_SHAPE = (250, 250, 3)
N_LABELS = 38
EPOCHS = 5
OPTIMIZER = 'adam'
LOSS = 'sparse_categorical_crossentropy'
METRICS = 'accuracy'
METHODS = ['MyCNN', 'InceptionV3']


class Train:
    def __init__(self,
                 batch_size=32,
                 method='inception_v3',
                 save=False):
        self.method = method
        if self.method not in METHODS:
            raise NotImplementedError(f'Method {self.method} not implemented')
        self.batch_size = batch_size
        self.loss = LOSS
        self.metrics = METRICS
        self.epochs = EPOCHS
        self.optimizer = OPTIMIZER
        self.id = '_'.join([self.method, str(self.batch_size), datetime.now().strftime("%Y%m%d-%H%M%S")])
        self.train_ds, self.validation_ds = self.create_datasets()
        self.model = self.build()
        self.fit(save)

    def create_datasets(self):

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory='./dataset/train',
            seed=42,
            image_size=IMAGE_TARGET_SIZE,
            batch_size=self.batch_size)

        validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory='./dataset/valid',
            seed=42,
            image_size=IMAGE_TARGET_SIZE,
            batch_size=self.batch_size)

        return train_ds, validation_ds

    def build(self):
        if self.method == 'InceptionV3':
            model = InceptionV3(N_LABELS)
        if self.method == 'MyCNN':
            model = MyCNN(N_LABELS)

        # Compile the model
        model.compile(self.optimizer, loss=self.loss, metrics=[self.metrics])

        return model

    def fit(self, save):

        # Tensorboard callback
        log_dir = 'logs/' + self.id
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

        # TODO https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects
        if save:
            model.save(f'./models/{self.id}/{self.id}', save_format='tf')

        return None
