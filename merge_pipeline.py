import math
import pandas as pd

from listings_pipeline import Listings
from calendar_pipeline import Calendar
from aux_model import AuxModel

class Merge():
    def __init__(self, listings_url, calendar_url):
        self.listings_url = listings_url
        self.calendar_url = calendar_url

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

    def __drop_data(self, data):
        data = data.drop(['id', 'room_type'], axis=1)
        data = data.drop(['description', 'amenities'], axis=1)
        return data

    def generate_features(self):
        listings_df = pd.read_csv(self.listings_url)
        calendar_df = pd.read_csv(self.calendar_url, parse_dates=['date'])
        listings_features = Listings(listings_df).generate_features()
        calendar_features = Calendar(calendar_df).generate_features()

        merged_features = self.__merge_listings_and_calendar(listings_features, calendar_features)
        merged_features = self.__fill_missing_data(merged_features)
        merged_features, aux_model_file = AuxModel(merged_features).train_and_save()
        merged_features = self.__drop_data(merged_features)
        return merged_features, aux_model_file
