import pandas as pd 
import numpy as np
from scipy.stats.stats import pearsonr
import itertools
from abc import ABCMeta

path = r'C:\Users\Home\Documents\DANIIL\programming\python\Code\projekts\data_science\models\kursovoj_project\dataset_kurs.csv'
#path = r'D:\Daniil\programming\kursovoj_project\dataset_kurs.csv'
data = pd.read_csv(path, encoding='utf-8')
#print(data.columns)

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

correlatoins = {}

class scoring_multi_correlation(metaclass=ABCMeta):
    '''
    In this function we make multi-correlation analys.
    '''

    def score_index(self, columns):
        '''
        This function will score pair coefficients.
        '''

        self.columns = columns

        for col_a, col_b in itertools.combinations(columns, 2):
            correlatoins[col_a + '__' + col_b] = pearsonr(data.loc[:, col_a], data.loc[:, col_b])

        result = pd.DataFrame.from_dict(correlatoins, orient='index')
        result.columns = ['PCC', 'p-value']
        return result

result = scoring_multi_correlation()

if __name__ == "__main__":
    print(result.score_index(columns=features).sort_index())
    