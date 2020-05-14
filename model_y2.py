import pandas as pd
from sklearn.model_selection import train_test_split
import models
from sklearn.ensemble import RandomForestRegressor

path = r'C:\Users\Home\Documents\DANIIL\programming\python\Code\projekts\data_science\models\kursovoj_project\dataset_kurs.csv'
#path = r'D:\Daniil\programming\kursovoj_project\dataset_kurs.csv'
data = pd.read_csv(path, encoding='utf-8')
#print(data.columns)

y = data.World_natural_gas_production_in_the_t_th_year_in_billion_cubic_meters
X = data.World_proven_reserves_of_natural_gas_in_the_t_th_year_in_billion_cubic_meters
#X = data.drop(['Index', 'Year', 'World_natural_gas_production_in_the_t_th_year_in_billion_cubic_meters'], axis=1)

train_X, valid_X, train_y, valid_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

scoring_models = models.scoring()

decision_tree_mae = scoring_models.decision(train_X, valid_X, train_y, valid_y)
print('MAE by Decision Tree is ' + str(decision_tree_mae))

random_forests_mae = scoring_models.rand_forrest(train_X, valid_X, train_y, valid_y)
print('MAE by Random Forrest is ' + str(random_forests_mae))

xgb_mae = scoring_models.grad_boost(train_X, valid_X, train_y, valid_y)
print('MAE by Gradient boosting is ' + str(xgb_mae))

model_y2 = RandomForestRegressor(n_estimators=11)

model_y2.fit(X, y)

if __name__ == '__main__':
    print('The best model is Random Forest.')
    print('Predictions are: ' + str(model_y2.predict(X.head())))
