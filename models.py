from abc import ABCMeta
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgbm
import catboost

class scoring(metaclass=ABCMeta):
    '''
    In this class are  some models to score MAE. 
    '''

    def decision(self, train_X, valid_X, train_y, valid_y):
        '''
        In this model we score MAE with Decision Tree Regressor.
        '''

        self.train_X = train_X
        self.valid_X = valid_X
        self.train_y = train_y
        self.valid_y = valid_y

        dec_model = DecisionTreeRegressor(max_leaf_nodes=11, random_state=0)
        dec_model.fit(train_X, train_y)
        pred = dec_model.predict(valid_X)
        mae = mean_absolute_error(valid_y, pred)
        #mse = mean_squared_error(valid_y, pred)
        return mae

    def rand_forrest(self, train_X, valid_X, train_y, valid_y):
        '''
        In this function we score MAE with random forrests.
        '''

        self.train_X = train_X
        self.valid_X = valid_X
        self.train_y = train_y
        self.valid_y = valid_y

        forrest_model = RandomForestRegressor(n_estimators=11, random_state=1, criterion='mae')
        forrest_model.fit(train_X, train_y)
        pred = forrest_model.predict(valid_X)
        mae = mean_absolute_error(valid_y, pred)
        return mae

    def grad_boost(self, train_X, valid_X, train_y, valid_y):
        '''
        In this function we score MAE with gradient boosting.
        '''

        self.train_X = train_X
        self.valid_X = valid_X
        self.train_y = train_y
        self.valid_y = valid_y

        grad_model = XGBRegressor()
        grad_model.fit(train_X, train_y)
        pred = grad_model.predict(valid_X)
        mae = mean_absolute_error(valid_y, pred)
        return mae

    def lgbm(self, train_X, valid_X, train_y, valid_y):
        '''
        In this function we score MAE with lightgbm.
        '''

        self.train_X = train_X
        self.valid_X = valid_X
        self.train_y = train_y
        self.valid_y = valid_y

        train_data = lgbm.Dataset(train_X, train_y)
        tests_data = lgbm.Dataset(valid_X, valid_y, reference=train_data)

        parameters = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
        }

        model = lgbm.train(parameters, train_data, valid_sets=tests_data, num_boost_round=1)
        
        y_pred = model.predict(valid_X, num_iteration=model.best_iteration)
        mae = mean_absolute_error(valid_y, y_pred)
        return mae

    def categorical_boosting(self, train_X, valid_X, train_y, valid_y):
        '''
        In this function we score MAE with categorical boosting(Yandex Technology).
        '''

        self.train_X = train_X
        self.valid_X = valid_X
        self.train_y = train_y
        self.valid_y = valid_y

        cat_model = catboost.CatBoostClassifier(iterations=20)
        cat_model.fit(train_X, train_y)
        predictions = cat_model.predict(valid_X)
        mae = mean_absolute_error(valid_y, predictions)
        return mae 
