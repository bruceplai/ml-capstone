import math
import string
import numpy as np
import pandas as pd

class Listings:
    def __init__(self, listings_df):
        self.listings_df = listings_df

    def __drop_data(self, data):
        data = data.drop(['listing_url', 'scrape_id', 'last_scraped', 'name', 'space',
            'experiences_offered', 'neighborhood_overview', 'notes', 'transit', 'access', 'interaction',
            'house_rules', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 'host_id', 'host_url',
            'host_name', 'host_location', 'host_about', 'host_response_time', 'host_response_rate',
            'host_acceptance_rate', 'host_since', 'host_listings_count', 'host_total_listings_count',
            'host_verifications', 'host_thumbnail_url', 'host_picture_url', 'host_has_profile_pic',
            'host_identity_verified', 'host_neighbourhood', 'street', 'neighbourhood', 'city', 'state', 'zipcode',
            'market', 'smart_location', 'country_code', 'country', 'is_location_exact', 'instant_bookable',
            'is_business_travel_ready', 'license', 'jurisdiction_names', 'review_scores_value', 'requires_license',
            'review_scores_communication', 'bed_type', 'weekly_price', 'monthly_price',
            'security_deposit', 'guests_included', 'minimum_minimum_nights', 'maximum_minimum_nights',
            'minimum_maximum_nights', 'maximum_maximum_nights', 'calendar_updated', 'has_availability',
            'calendar_last_scraped', 'number_of_reviews_ltm', 'first_review', 'last_review', 'review_scores_accuracy',
            'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_location', 'cancellation_policy',
            'require_guest_profile_picture', 'require_guest_phone_verification', 'calculated_host_listings_count',
            'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms',
            'calculated_host_listings_count_shared_rooms', 'cleaning_fee', 'extra_people',
            'square_feet', 'property_type', 'minimum_nights_avg_ntm',
            'maximum_nights_avg_ntm', 'price', 'minimum_nights', 'maximum_nights', 'availability_30',
            'availability_60', 'availability_90', 'availability_365',
            'review_scores_rating', 'number_of_reviews', 'reviews_per_month', 'summary',
            'neighbourhood_cleansed', 'neighbourhood_group_cleansed'], axis=1)
        data = data.drop_duplicates(subset=['latitude', 'longitude']).reset_index(drop=True)
        data = data.loc[pd.notna(data.description)]
        return data

    def __cleanup_homes_and_rooms(self, data):
        data = data.loc[
            (data.room_type == 'Entire home/apt') |
            (data.room_type == 'Private room')
        ]
        data['entire_home_apt'] = np.where(data.room_type == 'Entire home/apt', 1, 0)
        data = data.loc[
            (data.bedrooms <= 10) & (data.beds <=12) & (data.bathrooms <= 5)
        ]
        return data

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

        data = data.apply(avg_of_neighbors, col_name='bathrooms', cols_list=data.columns, axis=1)
        data = data.apply(avg_of_neighbors, col_name='bedrooms', cols_list=data.columns, axis=1)
        data = data.apply(avg_of_neighbors, col_name='beds', cols_list=data.columns, axis=1)
        return data

    def __add_amenities(self, data):
        def cleanup_amenities(text):
            remove = string.punctuation
            remove = remove.replace("-", "").replace(",", "").replace("/", "")
            text = text.lower().translate({ord(char): None for char in remove})
            text = text.replace("translation missing enhostingamenity49", "")
            text = text.replace("translation missing enhostingamenity50", "")
            text = text.replace(",,", ",").replace("’", "").replace(' / ', ' ').replace('/', ' ')
            text = text.replace(' ', '_').replace('-', '_')
            return text.split(",")

        data['amenities'] = data['amenities'].apply(cleanup_amenities)
        amenities_sparse = data.amenities.str.join('|').str.get_dummies()
        data = data.join(amenities_sparse)
        return data

    def generate_features(self):
        self.listings_df = self.__drop_data(self.listings_df)
        self.listings_df = self.__cleanup_homes_and_rooms(self.listings_df)
        self.listings_df = self.__add_amenities(self.listings_df)
        self.listings_df['host_is_superhost'] = np.where(self.listings_df.host_is_superhost == 't', 1, 0)
        return self.listings_df





