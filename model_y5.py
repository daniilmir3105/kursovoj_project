import pandas as pd
from sklearn.model_selection import train_test_split
import models

path = r'C:\Users\Home\Documents\DANIIL\programming\python\data_sets\dataset_kurs.csv'
data = pd.read_csv(path, encoding='utf-8')
#print(data.columns)

y = data.World_exports_of_shale_gas_in_the_t_th_year_million_cubic_meters

X = data.drop(['Index', 'Year', 'World_exports_of_shale_gas_in_the_t_th_year_million_cubic_meters'], axis=1)

train_X, valid_X, train_y, valid_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

scoring_models = models.scoring()

decision_tree_mae = scoring_models.decision(train_X, valid_X, train_y, valid_y)
print('MAE by Decision Tree is ' + str(decision_tree_mae))

random_forests_mae = scoring_models.rand_forrest(train_X, valid_X, train_y, valid_y)
print('MAE by Random Forrest is ' + str(random_forests_mae))

xgb_mae = scoring_models.grad_boost(train_X, valid_X, train_y, valid_y)
print('MAE by Gradient boosting is ' + str(xgb_mae))
