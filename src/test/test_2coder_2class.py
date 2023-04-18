import unittest
import pandas as pd
import numpy as np
from typing import List
from sklearn.metrics import cohen_kappa_score

from src.test.config import test_2coder_2class, test_common
from src.test.util import avg_mla_score

class TestUtilCase(unittest.TestCase):
    
    @classmethod
    ### Read anno_data, simu_data, result from log excel file
    def setUpClass(self):
        self.sheets = pd.read_excel(test_2coder_2class.result_file, sheet_name=None)
        self.result = self.sheets['result']
        self.result.set_index('name', inplace=True)
        

    def test_cohen(self):
        rater1 = self.sheets['anno_data']['coder_1'].values
        rater2 = self.sheets['anno_data']['coder_2'].values
        kappa_score = cohen_kappa_score(rater1, rater2)
        self.assertAlmostEqual(kappa_score, self.result['kappa']['cohen'], places=test_common.float_equal_digits)


    def test_simu_data_distribution(self):
        def get_distribution(s:pd.Series):
            data_counts = s.value_counts()
            for v in data_counts.index.tolist():
                data_counts.loc[v] = data_counts[v] / len(s)
            return data_counts

        anno_coder1_distribute = get_distribution(self.sheets['anno_data']['coder_1'])
        anno_coder2_distribute = get_distribution(self.sheets['anno_data']['coder_2'])
        for i in range(1, 4):
            simu_name = f'simu_data_0{i}'
            coder1_distribute = get_distribution(self.sheets[simu_name]['coder_1'])
            coder2_distribute = get_distribution(self.sheets[simu_name]['coder_2'])
            for v in anno_coder1_distribute.index.tolist():
                self.assertAlmostEqual(anno_coder1_distribute[v], coder1_distribute[v], 
                                       places=test_common.float_equal_digits,
                                       msg=f'coder1: distribution of {v} is not equal for anno_data and simu_data_0{i}')
                self.assertAlmostEqual(anno_coder2_distribute[v], coder2_distribute[v], 
                                       places=test_common.float_equal_digits,
                                       msg=f'coder2: distribution of {v} is not equal for anno_data and simu_data_0{i}')                

    def _df_to_labels(self, df:pd.DataFrame)->List[List[int]]:
        data_array = df[['coder_1', 'coder_2']].values
        return data_array.reshape((len(data_array), 2, 1)).tolist()

    def test_po(self):
        #### anno_data for 2 coders, format: List[List[List[int]]]
        anno_data_2coders = self._df_to_labels(self.sheets['anno_data'])  
        mla_score = avg_mla_score(anno_data_2coders)
        self.assertAlmostEqual(mla_score, self.result['p_o']['cohen'], places=test_common.float_equal_digits)
        
    def test_pe(self):
        for i in range(1, 4):
            sheet_name = f'simu_data_0{i}'
            index_name = f'mla_in_simu_{i}'
            simu_data = self._df_to_labels(self.sheets[sheet_name])
            mla_score = avg_mla_score(simu_data)
            self.assertAlmostEqual(mla_score, self.result['p_e'][index_name],
                                   places=test_common.float_equal_digits,
                                   msg='mla scores are not equal for {sheet_name}')
        

if __name__ == "__main__":
    unittest.main()       