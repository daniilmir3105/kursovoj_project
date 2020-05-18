import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable
import numpy as np 
from sklearn.metrics import r2_score
from abc import ABCMeta

path = r'C:\Users\Home\Documents\DANIIL\programming\python\Code\projekts\data_science\models\kursovoj_project\dataset_kurs.csv'
data = pd.read_csv(path, encoding='utf-8')
print(data.columns)

features = ['Export_of_natural_gas_of_the_Russian_Federation_in_the_t_th_year_in_billion_cubic_meters',
            'World_natural_gas_production_in_the_t_th_year_in_billion_cubic_meters', 
            'The_world_price_of_natural_gas_in_the_t_th_year_in_dollars_billion_cubic_meters',
            'World_oil_production_in_the_t_th_year_in_billion_cubic_meters', 
            'World_exports_of_shale_gas_in_the_t_th_year_million_cubic_meters']

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
    for i in features:
        y_result = data[i]
        print('Autocorrelation index by ' + i + str(autocorr_index.autocorr(y_result)))
