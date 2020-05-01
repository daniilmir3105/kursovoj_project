import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
path = r'C:\Users\Home\Documents\DANIIL\programming\python\Code\projekts\data_science\models\kursovoj_project\dataset_kurs.csv'

dataset = pd.read_csv(path)

features = ['Export_of_natural_gas_of_the_Russian_Federation_in_the_t_th_year_in_billion_cubic_meters',
            'World_natural_gas_production_in_the_t_th_year_in_billion_cubic_meters', 
            'The_world_price_of_natural_gas_in_the_t_th_year_in_dollars_billion_cubic_meters',
            'World_oil_production_in_the_t_th_year_in_billion_cubic_meters', 
            'World_exports_of_shale_gas_in_the_t_th_year_million_cubic_meters', 
            'World_proven_reserves_of_natural_gas_in_the_t_th_year_in_billion_cubic_meters', 
            'Worl_demand_for_natural_gas_in_the_t_th_year_in_billion_cubic_meters', 
            'Employment_in_the_gas_segment_in_million_in_the_t_th_year', 
            'World_LNG_prices_in_the_t_th_year_in_dollars_billion_cubic_meters', 
            'Investments_in_the_gas_segment_in_t_th_year_in_billion_dollar', 
            'World_oil_prices_in_the_t_th_year_in_billion_dollars_barrel', 
            'World_production_of_shale_gas_in_the_t_th_year_in_billion_cubic_meters']

#print(dataset.head())

print(dataset[features].corr())
