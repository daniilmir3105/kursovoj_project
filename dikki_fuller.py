import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable
from sklearn.metrics import r2_score
from abc import ABCMeta
#import ml_metrics as metrics

path = r'C:\Users\Home\Documents\DANIIL\programming\python\Code\projekts\data_science\models\kursovoj_project\dataset_kurs.csv'
#path = r'D:\Daniil\programming\kursovoj_project\dataset_kurs.csv'
data = pd.read_csv(path, encoding='utf-8')
#print(data.columns)

base_features = ['Export_of_natural_gas_of_the_Russian_Federation_in_the_t_th_year_in_billion_cubic_meters',
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

diff_features = ['Export_of_natural_gas_of_the_Russian_Federation_in_the_t_th_year_in_billion_cubic_meters',
            'World_natural_gas_production_in_the_t_th_year_in_billion_cubic_meters', 
            'The_world_price_of_natural_gas_in_the_t_th_year_in_dollars_billion_cubic_meters',
            'World_oil_production_in_the_t_th_year_in_billion_cubic_meters', 
            'World_exports_of_shale_gas_in_the_t_th_year_million_cubic_meters', 
            'World_proven_reserves_of_natural_gas_in_the_t_th_year_in_billion_cubic_meters', 
            'Worl_demand_for_natural_gas_in_the_t_th_year_in_billion_cubic_meters', 
            'Employment_in_the_gas_segment_in_million_in_the_t_th_year', 
            #'World_LNG_prices_in_the_t_th_year_in_dollars_billion_cubic_meters', 
            'Investments_in_the_gas_segment_in_t_th_year_in_billion_dollar', 
            'World_oil_prices_in_the_t_th_year_in_billion_dollars_barrel', 
            'World_production_of_shale_gas_in_the_t_th_year_in_billion_cubic_meters']

class making_dikki_fuller_test(metaclass=ABCMeta):
    '''
    In this metaclass we make dikki-fuller tests.
    ''' 

    def score_base_data(self, features):
        '''
        In this method we will make a simple dikki-fuller test. 
        '''
        
        for col in features:
            param = data[col]
            test = sm.tsa.adfuller(param)
            print('adf: ', test[0])
            print('p-value: ', test[1])
            print('Critical values: ', test[4])

            if test[0]> test[4]['5%']: 
                print('есть единичные корни, ряд не стационарен')
            else:
                print('единичных корней нет, ряд стационарен')
    
    def score_dif_data(self, features):
        '''
        Here we make different dikki-fuller test with the complex data.
        '''

        for col in features:
            param = data[col]
            param1diff = param.diff(periods=1).dropna()
            test = sm.tsa.adfuller(param1diff)
            print('adf: ', test[0])
            print('p-value: ', test[1])
            print('Critical values: ', test[4])

            if test[0]> test[4]['5%']: 
                print('есть единичные корни, ряд не стационарен')
            else:
                print('единичных корней нет, ряд стационарен')

result_1 = making_dikki_fuller_test()
result_2 = making_dikki_fuller_test()

if __name__ == '__main__':
    print(result_1.score_base_data(features=base_features))
    print(result_2.score_dif_data(features=diff_features))
