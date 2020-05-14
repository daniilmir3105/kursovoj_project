import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable
import numpy as np 
from sklearn.metrics import r2_score
from abc import ABCMeta

path = r'C:\Users\Home\Documents\DANIIL\programming\python\Code\projekts\data_science\models\kursovoj_project\dataset_kurs.csv'
data = pd.read_csv(path, encoding='utf-8')

features = ['Export_of_natural_gas_of_the_Russian_Federation_in_the_t_th_year_in_billion_cubic_meters',
            'World_natural_gas_production_in_the_t_th_year_in_billion_cubic_meters', 
            'The_world_price_of_natural_gas_in_the_t_th_year_in_dollars_billion_cubic_meters',
            'World_oil_production_in_the_t_th_year_in_billion_cubic_meters', 
            'World_exports_of_shale_gas_in_the_t_th_year_million_cubic_meters']
            #'World_proven_reserves_of_natural_gas_in_the_t_th_year_in_billion_cubic_meters', 
            #'Worl_demand_for_natural_gas_in_the_t_th_year_in_billion_cubic_meters', 
            #'Employment_in_the_gas_segment_in_million_in_the_t_th_year', 
            #'World_LNG_prices_in_the_t_th_year_in_dollars_billion_cubic_meters', 
            #'Investments_in_the_gas_segment_in_t_th_year_in_billion_dollar', 
            #'World_oil_prices_in_the_t_th_year_in_billion_dollars_barrel', 
            #'World_production_of_shale_gas_in_the_t_th_year_in_billion_cubic_meters']

class autocorr_an(metaclass=ABCMeta):
    '''
    In this metaclass we make autocorrelation analys.
    '''

    def autocorr(self, y):
        '''
        This method will calculate autocorrelation indexes.
        '''

        result = np.correlate(y, y, mode='same')
        return result

autocorr_index = autocorr_an()

# return autocorrelation indexes
if __name__ == '__main__':
    y_1 = data.Export_of_natural_gas_of_the_Russian_Federation_in_the_t_th_year_in_billion_cubic_meters
    print('Autocorrelation index by Export_of_natural_gas_of_the_Russian_Federation_in_the_t_th_year_in_billion_cubic_meters '+ str(autocorr_index.autocorr(y_1)))
            
    y_2 = data.World_natural_gas_production_in_the_t_th_year_in_billion_cubic_meters
    print('Autocorrelation index by World_natural_gas_production_in_the_t_th_year_in_billion_cubic_meters ' + str(autocorr_index.autocorr(y_2)))

    y_3 = data.The_world_price_of_natural_gas_in_the_t_th_year_in_dollars_billion_cubic_meters
    print('Autocorrelation index by The_world_price_of_natural_gas_in_the_t_th_year_in_dollars_billion_cubic_meters ' + str(autocorr_index.autocorr(y_3)))

    y_4 = data.World_oil_production_in_the_t_th_year_in_billion_cubic_meters
    print('Autocorrelation index by World_oil_production_in_the_t_th_year_in_billion_cubic_meters ' + str(autocorr_index.autocorr(y_4)))

    y_5 = data.World_exports_of_shale_gas_in_the_t_th_year_million_cubic_meters
    print('Autocorrelation index by World_exports_of_shale_gas_in_the_t_th_year_million_cubic_meters ' + str(autocorr_index.autocorr(y_5)))