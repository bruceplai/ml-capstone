import time
import pickle
import logging
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

class Model:
    def __init__(self, features):
        self.features = features
        self.logger = logging.getLogger('train.model')

    def __save_model(self, model):
        timestr = time.strftime('%Y%m%d_%H%M%S')
        file_name = ''.join(('model_', timestr))
        with open(file_name, 'wb') as model_file:
            pickle.dump(model, model_file)
        return file_name

    def __load_model(self, model_file):
        with open(model_file, 'rb') as file_name:
            return pickle.load(file_name)

    def __train_model(self, features):
        combo_list = [
            ['available_year_avg', 'min_nights_year_avg', 'price_year_avg']
        #     ['available_winter_avg', 'min_nights_winter_avg', 'price_winter_avg'],
        #     ['available_spring_avg', 'min_nights_spring_avg', 'price_spring_avg'],
        #     ['available_summer_avg', 'min_nights_summer_avg', 'price_summer_avg']
        ]
        for combo in combo_list:
            X_base = features.drop([
                'price_year_avg', 'price_winter_avg', 'price_spring_avg', 'price_summer_avg', 'price_fall_avg',
                'available_year_avg', 'available_winter_avg', 'available_spring_avg', 'available_summer_avg', 'available_fall_avg',
                'min_nights_year_avg', 'min_nights_winter_avg', 'min_nights_spring_avg', 'min_nights_summer_avg', 'min_nights_fall_avg'
            ], axis=1)
            X_base[combo[0]] = features[combo[0]]
            X_base[combo[1]] = features[combo[1]]
            y = features[combo[2]]
            X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=.25, random_state=42, shuffle=True)

            model = XGBRegressor(
                objective='reg:squarederror',
                learning_rate=0.1,
                max_depth=8,
                n_estimators=200,
                cv=5,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            self.logger.info('Gradient boost model:')
            self.logger.info(f'Target label: {combo[2]}')
            self.logger.info(f'R^2: {model.score(X_test, y_test)}')
            self.logger.info(f'MAE: {mean_absolute_error(y_test, model.predict(X_test))}')
            X_test.iloc[5000:5001].to_json(path_or_buf='sample.json', orient='records')
            return model
    
    def train_and_save(self):
        model_file = self.__save_model(self.__train_model(self.features))
        return model_file