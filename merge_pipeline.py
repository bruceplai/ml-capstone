import math
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

from listings_pipeline import Listings
from calendar_pipeline import Calendar

class Merge():
    def __init__(self, listings_url, calendar_url):
        self.listings_url = listings_url
        self.calendar_url = calendar_url
        self.model_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
        self.logger = logging.getLogger('train.merge_pipeline')
        tf.config.set_visible_devices([], 'GPU')

    def __merge_listings_and_calendar(self, listings_data, calendar_data):
        listings_data = listings_data.loc[listings_data.id.isin(calendar_data.index)]
        calendar_data = calendar_data.loc[calendar_data.index.isin(listings_data.id)]
        merge_data = pd.merge(listings_data, calendar_data, left_on='id', right_index=True)
        return merge_data

    def __fill_missing_data(self, data):
        def avg_of_neighbors(row, col_name, cols_list, round_result=True):
            lat_diff = 0.002
            long_diff = 0.002
            cols_list = list(cols_list)
            item = row[cols_list.index(col_name)]
            room_type = row[cols_list.index('room_type')]
            latitude = row[cols_list.index('latitude')]
            longitude = row[cols_list.index('longitude')]
            if math.isnan(item):
                if round_result:
                    item = data[col_name].loc[
                        (data.latitude > latitude - lat_diff) &
                        (data.latitude < latitude + lat_diff) &
                        (data.longitude > longitude - long_diff) &
                        (data.longitude < longitude + long_diff) &
                        (data.room_type == room_type)
                    ].mean().round()
                else:
                    item = data[col_name].loc[
                        (data.latitude > latitude - lat_diff) &
                        (data.latitude < latitude + lat_diff) &
                        (data.longitude > longitude - long_diff) &
                        (data.longitude < longitude + long_diff) &
                        (data.room_type == room_type)
                    ].mean()
                row[cols_list.index(col_name)] = item
            return row

        data = data.apply(avg_of_neighbors, col_name='price_year_avg', cols_list=data.columns, round_result=False, axis=1)
        data = data.apply(avg_of_neighbors, col_name='price_winter_avg', cols_list=data.columns, round_result=False, axis=1)
        data = data.apply(avg_of_neighbors, col_name='price_spring_avg', cols_list=data.columns, round_result=False, axis=1)
        data = data.apply(avg_of_neighbors, col_name='price_summer_avg', cols_list=data.columns, round_result=False, axis=1)
        data = data.apply(avg_of_neighbors, col_name='price_fall_avg', cols_list=data.columns, round_result=False, axis=1)
        return data

    def __generate_price_bins(self, data):
        price_bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, np.inf]
        bins = [*range(0, len(price_bins) - 1)]
        data['price_year_avg_bin'] = pd.cut(
            data.price_year_avg,
            price_bins,
            labels=bins
        ).astype('int')
        # data['price_winter_avg_bin'] = pd.cut(
        #     data.price_winter_avg,
        #     price_bins,
        #     labels=bins
        # ).astype('int')
        # data['price_spring_avg_bin'] = pd.cut(
        #     data.price_spring_avg,
        #     price_bins,
        #     labels=bins
        # ).astype('int')
        # data['price_summer_avg_bin'] = pd.cut(
        #     data.price_summer_avg,
        #     price_bins,
        #     labels=bins
        # ).astype('int')
        # data['price_fall_avg_bin'] = pd.cut(
        #     data.price_fall_avg,
        #     price_bins,
        #     labels=bins
        # ).astype('int')

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

        data['pred_price_year_avg_bin'] = model.predict(data.description, use_multiprocessing=1).argmax(axis=1)
        return data

    def __drop_data(self, data):
        data = data.drop(['id', 'room_type'], axis=1)
        data = data.drop([
            'price_year_avg_bin',
            # 'price_winter_avg_bin',
            # 'price_spring_avg_bin',
            # 'price_summer_avg_bin',
            # 'price_fall_avg_bin',
            'description',
            'amenities'
        ], axis=1)
        return data

    def generate_features(self):
        listings_df = pd.read_csv(self.listings_url)
        calendar_df = pd.read_csv(self.calendar_url, parse_dates=['date'])
        listings_features = Listings(listings_df).generate_features()
        calendar_features = Calendar(calendar_df).generate_features()
        merged_features = self.__merge_listings_and_calendar(listings_features, calendar_features)
        merged_features = self.__fill_missing_data(merged_features)
        merged_features = self.__generate_price_bins(merged_features)
        merged_features = self.__drop_data(merged_features)
        return merged_features
