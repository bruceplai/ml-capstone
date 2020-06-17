import os
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

class AuxModel:
    def __init__(self, features):
        self.features = features
        self.logger = logging.getLogger('train.aux_model')
        self.model_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
        tf.config.set_visible_devices([], 'GPU')

    def __save_model(self, model):
        if not os.path.exists('aux_model_default.ckpt.index'):
            file_name = 'aux_model_default.ckpt'
        else:
            timestr = time.strftime('%Y%m%d_%H%M%S')
            file_name = ''.join(('aux_model_', timestr, '.ckpt'))
        model.save_weights(file_name)
        return file_name

    def __train_model(self, data):
        price_bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, np.inf]
        bins = [*range(0, len(price_bins) - 1)]
        data['price_year_avg_bin'] = pd.cut(
            data.price_year_avg,
            price_bins,
            labels=bins
        ).astype('int')

        adam = tf.keras.optimizers.Adam(lr=0.001)
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        sca = tf.keras.metrics.SparseCategoricalAccuracy()
        es = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=4)

        X_train, X_val_test, y_train, y_val_test = train_test_split(
            data.description,
            data.price_year_avg_bin,
            test_size=0.4,
            stratify=data.price_year_avg_bin,
            random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test,
            y_val_test,
            test_size=0.5,
            stratify=y_val_test,
            random_state=42
        )

        def base_model():
            x = tf.keras.layers.Input(shape=[], dtype=tf.string)
            y = hub.KerasLayer(self.model_url, trainable=True)(x)
            z = tf.keras.layers.Dense(len(bins), activation='softmax')(y)
            model = tf.keras.models.Model(x, z)
            model.compile(optimizer=adam, loss=scce, metrics=[sca])
            return model

        model = base_model()
        model.fit(X_train, y_train, batch_size=512, epochs=5, validation_data=(X_val, y_val), callbacks=[es], use_multiprocessing=1)
        sca.update_state(y_test, model.predict(X_test))
        self.logger.info(f'Price bin prediction accuracy from listing descriptions: {sca.result().numpy()}')
        return model

    def train_and_save(self):
        model = self.__train_model(self.features)
        self.features['pred_price_year_avg_bin'] = model.predict(self.features.description, use_multiprocessing=1).argmax(axis=1)
        self.features = self.features.drop(['price_year_avg_bin'], axis=1)
        model_file = self.__save_model(model)
        return self.features, model_file
