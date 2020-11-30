import glob
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from datetime import datetime
from sklearn.utils import shuffle

IMAGE_TARGET_SIZE = (256, 256)
INPUT_SHAPE = (256, 256, 3)
N_LABELS = 22
METHODS = ['inception_v3']


class Train:
    def __init__(self,
                 train_prop=0.8,
                 batch_size=32,
                 loss='categorical_crossentropy',
                 metrics='accuracy',
                 epochs=50,
                 method='inception_v3',
                 save=True):
        self.method = method
        if self.method not in METHODS:
            raise NotImplementedError(f'Method {self.method} not implemented')
        self.batch_size = batch_size
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.id = '_'.join([self.method, str(self.batch_size), str(train_prop), datetime.now().strftime("%Y%m%d-%H%M%S")])
        self.df = self.create_dataframe()
        self.nrow = len(self.df.index)
        self.split = round(self.nrow * train_prop)
        self.train_generator, self.validation_generator = self.create_generators_from_df()
        self.model = self.build()
        self.fit(save)

    @staticmethod
    def create_dataframe():
        # Create dataframe containing label to file mapping
        all_files = glob.glob('**/*.JPG', recursive=True)
        df = pd.DataFrame(all_files, columns=['paths'])
        df[['directory', 'plant_classes', 'sickness_classes', 'image_names']] = df.paths.str.split('/', expand=True)
        df['labels'] = df['plant_classes'] + '_' + df['sickness_classes']
        df = shuffle(df, random_state=0)

        return df

    def create_generators_from_df(self):
        generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                    shear_range=0.2,
                                                                    zoom_range=0.2,
                                                                    horizontal_flip=True)

        train_generator = generator.flow_from_dataframe(
            dataframe=self.df[:self.split],
            x_col="paths",
            y_col="labels",
            class_mode="categorical",
            target_size=IMAGE_TARGET_SIZE,
            batch_size=self.batch_size)

        validation_generator = generator.flow_from_dataframe(
            dataframe=self.df[self.split:],
            x_col="paths",
            y_col="labels",
            class_mode="categorical",
            target_size=IMAGE_TARGET_SIZE,
            batch_size=self.batch_size)

        return train_generator, validation_generator

    def build(self):
        if self.method == 'inception_v3':
            feature_extractor_url = "https://tfhub.dev/google/imagenet/inception_v3/classification/4"
            feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=INPUT_SHAPE, trainable=False)

            # Build model
            model = tf.keras.models.Sequential([
                feature_extractor_layer,
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(N_LABELS, activation='softmax')
            ])

        # Compile and fit the model
        model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001), loss=self.loss, metrics=[self.metrics])

        return model

    def fit(self, save):
        log_dir = 'logs/' + self.id
        tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Training callbacks
        es = tf.keras.callbacks.EarlyStopping(monitor="accuracy", patience=5, restore_best_weights=True)
        rl = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)

        self.model.fit_generator(generator=self.train_generator,
                                 steps_per_epoch=self.train_generator.n // self.train_generator.batch_size,
                                 validation_data=self.validation_generator,
                                 validation_steps=self.validation_generator.n // self.validation_generator.batch_size,
                                 epochs=self.epochs,
                                 callbacks=[es, rl, tb])

        if save:
            export_path = f"./models/{self.id}.h5"
            self.model.save(export_path)

        return None
