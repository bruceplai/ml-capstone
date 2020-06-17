import pickle
import logging
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

class Inference():
    def __init__(self, model_save, aux_model_save):
        self.logger = logging.getLogger('app.inference')
        self.aux_model_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
        self.aux_model = self._load_aux_model(aux_model_save)
        self.main_model = self._load_main_model(model_save)

    def _load_aux_model(self, model_weight_file):
        adam = tf.keras.optimizers.Adam(lr=0.001)
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        sca = tf.keras.metrics.SparseCategoricalAccuracy()

        def base_model():
            x = tf.keras.layers.Input(shape=[], dtype=tf.string)
            y = hub.KerasLayer(self.aux_model_url, trainable=True)(x)
            z = tf.keras.layers.Dense(9, activation='softmax')(y)
            model = tf.keras.models.Model(x, z)
            model.compile(optimizer=adam, loss=scce, metrics=[sca])
            return model

        model = base_model()
        model.load_weights(model_weight_file)
        return model

    def _load_main_model(self, model_file):
        with open(model_file, 'rb') as file_name:
            return pickle.load(file_name)

    def get_model_pred(self, input_data, description = None):
        data = pd.DataFrame([input_data])
        if description:
            data['pred_price_year_avg_bin'] = self.aux_model.predict([description], use_multiprocessing=1).argmax(axis=1)
        return self.main_model.predict(data)[0]
