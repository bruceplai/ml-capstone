import math
import string

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

tf.config.set_visible_devices([], 'GPU')

def avg_of_neighbors(row, col_name, cols_list):
    lat_diff = 0.002
    long_diff = 0.002
    cols_list = list(cols_list)
    item = row[cols_list.index(col_name)]
    room_type = row[cols_list.index('room_type')]
    latitude = row[cols_list.index('latitude')]
    longitude = row[cols_list.index('longitude')]
    if math.isnan(item):
        item = listings_clean[col_name].loc[
            (listings_clean.latitude > latitude - lat_diff) &
            (listings_clean.latitude < latitude + lat_diff) &
            (listings_clean.longitude > longitude - long_diff) &
            (listings_clean.longitude < longitude + long_diff) &
            (listings_clean.room_type == room_type)
        ].groupby(by=listings_clean.neighbourhood).mean().round()[0]
        row[cols_list.index(col_name)] = item
    return row

def cleanup_neighbourhood(val):
    return val.replace(' ', '_').lower()

def cleanup_amenities(text):
    remove = string.punctuation
    remove = remove.replace("-", "").replace(",", "").replace("/", "")
    text = text.lower().translate({ord(char): None for char in remove})
    text = text.replace("translation missing enhostingamenity49", "")
    text = text.replace("translation missing enhostingamenity50", "")
    text = text.replace(",,", ",").replace("’", "").replace(' / ', ' ').replace('/', ' ')
    text = text.replace(' ', '_').replace('-', '_')
    return text.split(",")


listings_df = pd.read_csv('listings.csv.gz')
calendar_df = pd.read_csv('calendar.csv.gz', parse_dates=['date'])

listings_clean = listings_df.drop(['listing_url', 'scrape_id', 'last_scraped', 'name', 'space',
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
    'review_scores_rating', 'number_of_reviews', 'reviews_per_month', 'summary'], axis=1)

listings_clean.rename(columns={
    'neighbourhood_cleansed':'neighbourhood',
    'neighbourhood_group_cleansed': 'neighbourhood_group'
}, inplace=True)
listings_clean = listings_clean.loc[
    (listings_clean.room_type == 'Entire home/apt') |
    (listings_clean.room_type == 'Private room')
]
listings_clean = listings_clean.drop_duplicates(subset=['latitude', 'longitude']).reset_index(drop=True)

listings_clean = listings_clean.loc[
    (listings_clean.bedrooms <= 10) & (listings_clean.beds <=12) & (listings_clean.bathrooms <= 5)
]

listings_clean = listings_clean.apply(avg_of_neighbors, col_name='bathrooms', cols_list=listings_clean.columns, axis=1)
listings_clean = listings_clean.apply(avg_of_neighbors, col_name='bedrooms', cols_list=listings_clean.columns, axis=1)
listings_clean = listings_clean.apply(avg_of_neighbors, col_name='beds', cols_list=listings_clean.columns, axis=1)

listings_clean['host_is_superhost'] = np.where(listings_clean.host_is_superhost == 't', 1, 0)
listings_clean = listings_clean.loc[pd.notna(listings_clean.description)]

listings_clean['amenities'] = listings_clean['amenities'].apply(cleanup_amenities)
amenities_sparse = listings_clean.amenities.str.join('|').str.get_dummies()
listings_clean = listings_clean.join(amenities_sparse)


calendar_clean = calendar_df[calendar_df.listing_id != 15268792]
calendar_clean['date'] = calendar_clean['date'].mask(
    calendar_clean['date'].dt.year == 2020,
    calendar_clean['date'] - pd.to_timedelta(365, unit='D') + pd.to_timedelta(12, unit='h'))
calendar_clean['date'] = calendar_clean['date'].dt.floor('D')

calendar_clean = calendar_clean.drop_duplicates(subset=['listing_id', 'date']).reset_index(drop=True)
calendar_clean = calendar_clean[calendar_clean.listing_id.isin(listings_clean.id)]
calendar_clean['available'] = np.where(calendar_clean.available == 't', 1, 0)

calendar_clean['price_cleansed'] = calendar_clean.adjusted_price\
    .str.replace('$', '').str.replace(',', '').astype('float')
calendar_clean = calendar_clean.drop(['adjusted_price', 'price'], axis=1)
calendar_clean.rename(columns={'price_cleansed': 'price'}, inplace=True)

available_year_avg = calendar_clean.groupby(by='listing_id').available.mean()
available_winter_avg = calendar_clean[(calendar_clean.date.dt.month == 1) | (calendar_clean.date.dt.month == 2) \
    | (calendar_clean.date.dt.month == 12)].groupby(by='listing_id').available.mean()
available_spring_avg = calendar_clean[(calendar_clean.date.dt.month >= 3) & (calendar_clean.date.dt.month <= 5)]\
.groupby(by='listing_id').available.mean()
available_summer_avg = calendar_clean[(calendar_clean.date.dt.month >= 6) & (calendar_clean.date.dt.month <= 8)]\
.groupby(by='listing_id').available.mean()
available_fall_avg = calendar_clean[(calendar_clean.date.dt.month >= 9) & (calendar_clean.date.dt.month <= 11)]\
.groupby(by='listing_id').available.mean()
available_jan_avg = calendar_clean[calendar_clean.date.dt.month == 1].groupby(by='listing_id').available.mean()
available_jun_avg = calendar_clean[calendar_clean.date.dt.month == 6].groupby(by='listing_id').available.mean()
available_dec_avg = calendar_clean[calendar_clean.date.dt.month == 12].groupby(by='listing_id').available.mean()

calendar_averages = pd.DataFrame(available_year_avg)
calendar_averages.rename(columns={'available': 'available_year_avg'}, inplace=True)
calendar_averages['available_winter_avg'] = available_winter_avg
calendar_averages['available_spring_avg'] = available_spring_avg
calendar_averages['available_summer_avg'] = available_summer_avg
calendar_averages['available_fall_avg'] = available_fall_avg
calendar_averages['available_jan_avg'] = available_jan_avg
calendar_averages['available_jun_avg'] = available_jun_avg
calendar_averages['available_dec_avg'] = available_dec_avg

min_nights_year_avg = calendar_clean.groupby(by='listing_id')['minimum_nights'].mean()
min_nights_winter_avg = calendar_clean[(calendar_clean.date.dt.month == 1) | (calendar_clean.date.dt.month == 2) \
    | (calendar_clean.date.dt.month == 12)].groupby(by='listing_id')['minimum_nights'].mean()
min_nights_spring_avg = calendar_clean[(calendar_clean.date.dt.month >= 3) & (calendar_clean.date.dt.month <= 5)]\
.groupby(by='listing_id')['minimum_nights'].mean()
min_nights_summer_avg = calendar_clean[(calendar_clean.date.dt.month >= 6) & (calendar_clean.date.dt.month <= 8)]\
.groupby(by='listing_id')['minimum_nights'].mean()
min_nights_fall_avg = calendar_clean[(calendar_clean.date.dt.month >= 9) & (calendar_clean.date.dt.month <= 11)]\
.groupby(by='listing_id')['minimum_nights'].mean()
min_nights_jan_avg = calendar_clean[calendar_clean.date.dt.month == 1].groupby(by='listing_id')['minimum_nights'].mean()
min_nights_jun_avg = calendar_clean[calendar_clean.date.dt.month == 6].groupby(by='listing_id')['minimum_nights'].mean()
min_nights_dec_avg = calendar_clean[calendar_clean.date.dt.month == 12].groupby(by='listing_id')['minimum_nights'].mean()

calendar_averages['min_nights_year_avg'] = min_nights_year_avg
calendar_averages['min_nights_winter_avg'] = min_nights_winter_avg
calendar_averages['min_nights_spring_avg'] = min_nights_spring_avg
calendar_averages['min_nights_summer_avg'] = min_nights_summer_avg
calendar_averages['min_nights_fall_avg'] = min_nights_fall_avg
calendar_averages['min_nights_jan_avg'] = min_nights_jan_avg
calendar_averages['min_nights_jun_avg'] = min_nights_jun_avg
calendar_averages['min_nights_dec_avg'] = min_nights_dec_avg

price_year_avg = calendar_clean.groupby(by='listing_id')['price'].mean()
price_winter_avg = calendar_clean[(calendar_clean.date.dt.month == 1) | (calendar_clean.date.dt.month == 2) \
    | (calendar_clean.date.dt.month == 12)].groupby(by='listing_id')['price'].mean()
price_spring_avg = calendar_clean[(calendar_clean.date.dt.month >= 3) & (calendar_clean.date.dt.month <= 5)]\
.groupby(by='listing_id')['price'].mean()
price_summer_avg = calendar_clean[(calendar_clean.date.dt.month >= 6) & (calendar_clean.date.dt.month <= 8)]\
.groupby(by='listing_id')['price'].mean()
price_fall_avg = calendar_clean[(calendar_clean.date.dt.month >= 9) & (calendar_clean.date.dt.month <= 11)]\
.groupby(by='listing_id')['price'].mean()
price_jan_avg = calendar_clean[calendar_clean.date.dt.month == 1].groupby(by='listing_id')['price'].mean()
price_jun_avg = calendar_clean[calendar_clean.date.dt.month == 6].groupby(by='listing_id')['price'].mean()
price_dec_avg = calendar_clean[calendar_clean.date.dt.month == 12].groupby(by='listing_id')['price'].mean()

calendar_averages['price_year_avg'] = price_year_avg
calendar_averages['price_winter_avg'] = price_winter_avg
calendar_averages['price_spring_avg'] = price_spring_avg
calendar_averages['price_summer_avg'] = price_summer_avg
calendar_averages['price_fall_avg'] = price_fall_avg
calendar_averages['price_jan_avg'] = price_jan_avg
calendar_averages['price_jun_avg'] = price_jun_avg
calendar_averages['price_dec_avg'] = price_dec_avg

calendar_averages = calendar_averages.loc[
    (calendar_averages.min_nights_year_avg <= 30) & 
    (calendar_averages.min_nights_winter_avg <= 30) &
    (calendar_averages.min_nights_spring_avg <= 30) &
    (calendar_averages.min_nights_summer_avg <= 30) &
    (calendar_averages.min_nights_fall_avg <= 30) &
    (calendar_averages.min_nights_jan_avg <= 30) &
    (calendar_averages.min_nights_jun_avg <= 30) &
    (calendar_averages.min_nights_dec_avg <= 30)
]

calendar_averages = calendar_averages.loc[
    (calendar_averages.price_year_avg >= 20) & 
    (calendar_averages.price_winter_avg >= 20) & 
    (calendar_averages.price_spring_avg >= 20) & 
    (calendar_averages.price_summer_avg >= 20) & 
    (calendar_averages.price_fall_avg >= 20) & 
    (calendar_averages.price_jan_avg >= 20) & 
    (calendar_averages.price_jun_avg >= 20) & 
    (calendar_averages.price_dec_avg >= 20) 
]

calendar_averages = calendar_averages.loc[
    (calendar_averages.price_year_avg <= 500) & 
    (calendar_averages.price_winter_avg <= 500) & 
    (calendar_averages.price_spring_avg <= 500) & 
    (calendar_averages.price_summer_avg <= 500) & 
    (calendar_averages.price_fall_avg <= 500) & 
    (calendar_averages.price_jan_avg <= 500) & 
    (calendar_averages.price_jun_avg <= 500) & 
    (calendar_averages.price_dec_avg <= 500) 
]


listings_clean = listings_clean.loc[listings_clean.id.isin(calendar_averages.index)]
listings_merge = pd.merge(listings_clean, calendar_averages, left_on='id', right_index=True)

listings_merge = listings_merge.apply(avg_of_neighbors, col_name='price_year_avg', cols_list=listings_merge.columns, axis=1)
listings_merge = listings_merge.apply(avg_of_neighbors, col_name='price_winter_avg', cols_list=listings_merge.columns, axis=1)
listings_merge = listings_merge.apply(avg_of_neighbors, col_name='price_spring_avg', cols_list=listings_merge.columns, axis=1)
listings_merge = listings_merge.apply(avg_of_neighbors, col_name='price_summer_avg', cols_list=listings_merge.columns, axis=1)
listings_merge = listings_merge.apply(avg_of_neighbors, col_name='price_fall_avg', cols_list=listings_merge.columns, axis=1)
listings_merge = listings_merge.apply(avg_of_neighbors, col_name='price_jan_avg', cols_list=listings_merge.columns, axis=1)
listings_merge = listings_merge.apply(avg_of_neighbors, col_name='price_jun_avg', cols_list=listings_merge.columns, axis=1)
listings_merge = listings_merge.apply(avg_of_neighbors, col_name='price_dec_avg', cols_list=listings_merge.columns, axis=1)


price_bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, np.inf]
bins = [*range(0, len(price_bins) - 1)]
listings_merge['price_year_avg_bin'] = pd.cut(
    listings_merge.price_year_avg,
    price_bins,
    labels=bins
).astype('int')

adam = tf.keras.optimizers.Adam(lr=0.001)
scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
sca = tf.keras.metrics.SparseCategoricalAccuracy()
es = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=4)

X_train, X_val_test, y_train, y_val_test = train_test_split(
    listings_merge.description,
    listings_merge.price_year_avg_bin,
    test_size=0.4,
    stratify=listings_merge.price_year_avg_bin,
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
    y = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4', trainable=True)(x)
    z = tf.keras.layers.Dense(len(bins), activation='softmax')(y)
    model = tf.keras.models.Model(x, z)
    model.compile(optimizer=adam, loss=scce, metrics=[sca])
    return model

model = base_model()
model.fit(X_train, y_train, batch_size=512, epochs=3, validation_data=(X_val, y_val), callbacks=[es], use_multiprocessing=1)
sca.update_state(y_test, model.predict(X_test))
print("Price bin prediction accuracy from listing descriptions:", sca.result().numpy())

listings_merge['pred_price_year_avg_bin'] = model.predict(listings_merge.description, use_multiprocessing=1).argmax(axis=1)


full_features = listings_merge.drop(['id', 'room_type', 'neighbourhood', 'neighbourhood_group'], axis=1)
full_features = full_features.drop(['price_year_avg_bin', 'description', 'amenities'], axis=1)


combo_list = [
    ['available_year_avg', 'min_nights_year_avg', 'price_year_avg']
#     ['available_winter_avg', 'min_nights_winter_avg', 'price_winter_avg'],
#     ['available_spring_avg', 'min_nights_spring_avg', 'price_spring_avg'],
#     ['available_summer_avg', 'min_nights_summer_avg', 'price_summer_avg'],
#     ['available_fall_avg', 'min_nights_fall_avg', 'price_fall_avg'],
#     ['available_jan_avg', 'min_nights_jan_avg', 'price_jan_avg'],
#     ['available_jun_avg', 'min_nights_jun_avg', 'price_jun_avg'],
#     ['available_dec_avg', 'min_nights_dec_avg', 'price_dec_avg']
]

def run_gb_model(data_set_name, features):
    for combo in combo_list:
        X_base = features.drop([
            'price_year_avg', 'price_winter_avg', 'price_spring_avg', 'price_summer_avg', 'price_fall_avg',
            'price_jan_avg', 'price_jun_avg', 'price_dec_avg',
            'available_year_avg', 'available_winter_avg', 'available_spring_avg', 'available_summer_avg',
            'available_fall_avg', 'available_jan_avg', 'available_jun_avg', 'available_dec_avg',
            'min_nights_year_avg', 'min_nights_winter_avg', 'min_nights_spring_avg', 'min_nights_summer_avg',
            'min_nights_fall_avg', 'min_nights_jan_avg', 'min_nights_jun_avg', 'min_nights_dec_avg',
        ], axis=1)
        X_base[combo[0]] = features[combo[0]]
        X_base[combo[1]] = features[combo[1]]
        y = features[combo[2]]

        X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=.25, random_state=42, shuffle=True)

        clf = XGBRegressor(
            objective='reg:squarederror',
            learning_rate=0.1,
            max_depth=8,
            n_estimators=200,
            cv=5,
            n_jobs=-1
        )

        clf.fit(X_train, y_train)
        print('Gradient boost model for', data_set_name)
        print('Target label:', combo[2])
        print('R^2:', clf.score(X_test, y_test))
        print('MAE:', mean_absolute_error(y_test, clf.predict(X_test)))

run_gb_model('both homes and rooms', full_features)
