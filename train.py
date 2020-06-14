import logging
from merge_pipeline import Merge
from model import Model

logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)

listings_url = 'http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/data/listings.csv.gz'
calendar_url = 'http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/data/calendar.csv.gz'
# listing_url = 'http://data.insideairbnb.com/united-states/ny/new-york-city/2020-06-08/data/listings.csv.gz'
# calendar_url = 'http://data.insideairbnb.com/united-states/ny/new-york-city/2020-06-08/data/calendar.csv.gz'

full_features = Merge(listings_url, calendar_url).generate_features()
logger.info(f'Trained airbnb pricing model saved as: {Model(full_features).train_and_save()}')
