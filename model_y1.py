import pandas as pd
from sklearn.model_selection import train_test_split
import models

path = r'C:\Users\Home\Documents\DANIIL\programming\python\data_sets\dataset_kurs.csv'
data = pd.read_csv(path, encoding='utf-8')
print(data.columns)

y = data.Export_of_natural_gas_of_the_Russian_Federation_in_the_t_th_year_in_billion_cubic_meters

X = data.drop(['Index', 'Year', 'Export_of_natural_gas_of_the_Russian_Federation_in_the_t_th_year_in_billion_cubic_meters'], axis=1)

train_X, valid_X, trai_y, valid_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
