from abc import ABCMeta
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class scoring(metaclass=ABCMeta):
    '''
    In this class we will score MAE in various models. 
    '''

    def decision(self, train_X, valid_X, train_y, valid_y):
        self.train_X = train_X
        self.valid_X = valid_X
        self.train_y = train_y
        self.valid_y = valid_y

        dec_model = DecisionTreeRegressor(max_leaf_nodes=7, random_state=0)
        dec_model.fit(train_X, train_y)
        pred = dec_model.predict(valid_X)
        mae = mean_absolute_error(valid_y, pred)
        #mse = mean_squared_error(valid_y, pred)
        return mae

    def rand_forrest(self, train_X, valid_X, train_y, valid_y):
        self.train_X = train_X
        self.valid_X = valid_X
        self.train_y = train_y
        self.valid_y = valid_y

        forrest_model = RandomForestRegressor(random_state=1)
        forrest_model.fit(train_X, train_y)
        pred = forrest_model.predict(valid_X)
        mae = mean_absolute_error(valid_y, pred)
        return mae

    def grad_boost(self, train_X, valid_X, train_y, valid_y):
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
        self.train_X = train_X
        self.valid_X = valid_X
        self.train_y = train_y
        self.valid_y = valid_y

        