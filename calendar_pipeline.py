import numpy as np
import pandas as pd

class Calendar:
    def __init__(self, calendar_df):
        self.calendar_df = calendar_df

    def __clean_price(self, data):
        data['price_cleansed'] = data.adjusted_price.str.replace('$', '').str.replace(',', '').astype('float')
        data = data.drop(['adjusted_price', 'price'], axis=1)
        data.rename(columns={'price_cleansed': 'price'}, inplace=True)
        return data

    def __generate_available_avg(self, data):
        available_year_avg = data.groupby(by='listing_id').available.mean()
        available_winter_avg = data.loc[(data.date.dt.month == 1) | (data.date.dt.month == 2) \
            | (data.date.dt.month == 12)].groupby(by='listing_id').available.mean()
        available_spring_avg = data.loc[(data.date.dt.month >= 3) & (data.date.dt.month <= 5)]\
        .groupby(by='listing_id').available.mean()
        available_summer_avg = data.loc[(data.date.dt.month >= 6) & (data.date.dt.month <= 8)]\
        .groupby(by='listing_id').available.mean()
        available_fall_avg = data.loc[(data.date.dt.month >= 9) & (data.date.dt.month <= 11)]\
        .groupby(by='listing_id').available.mean()
        return available_year_avg, available_winter_avg, available_spring_avg, available_summer_avg, available_fall_avg

    def __generate_min_nights_avg(self, data):
        min_nights_year_avg = data.groupby(by='listing_id')['minimum_nights'].mean()
        min_nights_winter_avg = data.loc[(data.date.dt.month == 1) | (data.date.dt.month == 2) \
            | (data.date.dt.month == 12)].groupby(by='listing_id')['minimum_nights'].mean()
        min_nights_spring_avg = data.loc[(data.date.dt.month >= 3) & (data.date.dt.month <= 5)]\
        .groupby(by='listing_id')['minimum_nights'].mean()
        min_nights_summer_avg = data.loc[(data.date.dt.month >= 6) & (data.date.dt.month <= 8)]\
        .groupby(by='listing_id')['minimum_nights'].mean()
        min_nights_fall_avg = data.loc[(data.date.dt.month >= 9) & (data.date.dt.month <= 11)]\
        .groupby(by='listing_id')['minimum_nights'].mean()
        return min_nights_year_avg, min_nights_winter_avg, min_nights_spring_avg, min_nights_summer_avg, min_nights_fall_avg

    def __generate_price_avg(self, data):
        price_year_avg = data.groupby(by='listing_id')['price'].mean()
        price_winter_avg = data.loc[(data.date.dt.month == 1) | (data.date.dt.month == 2) \
            | (data.date.dt.month == 12)].groupby(by='listing_id')['price'].mean()
        price_spring_avg = data.loc[(data.date.dt.month >= 3) & (data.date.dt.month <= 5)]\
        .groupby(by='listing_id')['price'].mean()
        price_summer_avg = data.loc[(data.date.dt.month >= 6) & (data.date.dt.month <= 8)]\
        .groupby(by='listing_id')['price'].mean()
        price_fall_avg = data.loc[(data.date.dt.month >= 9) & (data.date.dt.month <= 11)]\
        .groupby(by='listing_id')['price'].mean()
        return price_year_avg, price_winter_avg, price_spring_avg, price_summer_avg, price_fall_avg

    def __generate_avg_features(self, data):
        available_year_avg, available_winter_avg, available_spring_avg, available_summer_avg, available_fall_avg = \
            self.__generate_available_avg(data)
        calendar_averages = pd.DataFrame(available_year_avg)
        calendar_averages.rename(columns={'available': 'available_year_avg'}, inplace=True)
        calendar_averages['available_winter_avg'] = available_winter_avg
        calendar_averages['available_spring_avg'] = available_spring_avg
        calendar_averages['available_summer_avg'] = available_summer_avg
        calendar_averages['available_fall_avg'] = available_fall_avg

        min_nights_year_avg, min_nights_winter_avg, min_nights_spring_avg, min_nights_summer_avg, min_nights_fall_avg = \
            self.__generate_min_nights_avg(data)
        calendar_averages['min_nights_year_avg'] = min_nights_year_avg
        calendar_averages['min_nights_winter_avg'] = min_nights_winter_avg
        calendar_averages['min_nights_spring_avg'] = min_nights_spring_avg
        calendar_averages['min_nights_summer_avg'] = min_nights_summer_avg
        calendar_averages['min_nights_fall_avg'] = min_nights_fall_avg

        price_year_avg, price_winter_avg, price_spring_avg, price_summer_avg, price_fall_avg = \
            self.__generate_price_avg(data)
        calendar_averages['price_year_avg'] = price_year_avg
        calendar_averages['price_winter_avg'] = price_winter_avg
        calendar_averages['price_spring_avg'] = price_spring_avg
        calendar_averages['price_summer_avg'] = price_summer_avg
        calendar_averages['price_fall_avg'] = price_fall_avg
        return calendar_averages

    def __drop_outliers(self, data):
        data = data.loc[
            (data.min_nights_year_avg <= 30) & 
            (data.min_nights_winter_avg <= 30) &
            (data.min_nights_spring_avg <= 30) &
            (data.min_nights_summer_avg <= 30) &
            (data.min_nights_fall_avg <= 30)
        ]

        data = data.loc[
            (data.price_year_avg >= 20) & 
            (data.price_winter_avg >= 20) & 
            (data.price_spring_avg >= 20) & 
            (data.price_summer_avg >= 20) & 
            (data.price_fall_avg >= 20)
        ]

        data = data.loc[
            (data.price_year_avg <= 500) & 
            (data.price_winter_avg <= 500) & 
            (data.price_spring_avg <= 500) & 
            (data.price_summer_avg <= 500) & 
            (data.price_fall_avg <= 500)
        ]
        return data

    def generate_features(self):
        self.calendar_df = self.calendar_df[self.calendar_df.listing_id != 15268792]
        self.calendar_df['date'] = self.calendar_df.date.dt.floor('D')
        self.calendar_df = self.calendar_df.drop_duplicates(subset=['listing_id', 'date']).reset_index(drop=True)
        self.calendar_df['available'] = np.where(self.calendar_df.available == 't', 1, 0)
        self.calendar_df = self.__clean_price(self.calendar_df)
        calendar_avg = self.__generate_avg_features(self.calendar_df)
        return self.__drop_outliers(calendar_avg)

