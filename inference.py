import pickle
import logging
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

logger = logging.getLogger('inference')
logger.setLevel(logging.DEBUG)

aux_model_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
adam = tf.keras.optimizers.Adam(lr=0.001)
scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
sca = tf.keras.metrics.SparseCategoricalAccuracy()
es = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=4)

def base_model():
    x = tf.keras.layers.Input(shape=[], dtype=tf.string)
    y = hub.KerasLayer(aux_model_url, trainable=True)(x)
    z = tf.keras.layers.Dense(9, activation='softmax')(y)
    model = tf.keras.models.Model(x, z)
    model.compile(optimizer=adam, loss=scce, metrics=[sca])
    return model

def load_model(model_file):
    with open(model_file, 'rb') as file_name:
        return pickle.load(file_name)

aux_model = base_model()
aux_model.load_weights('aux_model_20200614_221237.ckpt')
model = load_model('model_20200614_221254')


input_data = pd.read_json('sample.json')
input_data['pred_price_year_avg_bin'] = aux_model.predict(input_data.description, use_multiprocessing=1).argmax(axis=1)
input_data = input_data.drop(['description'], axis=1)
logger.info(model.predict(input_data))
