import unittest
import pandas as pd
from typing import List
from sklearn.metrics import cohen_kappa_score
import glob
import os

from test.config import common_config
from test.util import avg_mla_score

class Base:

    class BaseTest(unittest.TestCase):
        result_file_prefix = ''
    
        ### Read anno_data, simu_data, result from excel file
        @classmethod
        def setUpClass(self):
            # super().setUpClass()
            # find result file folder
            res_folder = glob.glob(f'{common_config.log_path}/{self.result_file_prefix}*')
            # find the latest excel file
            latest_file = max(glob.glob(f'{res_folder[0]}/*.xlsx'), key=os.path.getctime)
            
            self.sheets = pd.read_excel(latest_file, sheet_name=None)
            self.result = self.sheets['result']
            self.result.set_index('name', inplace=True)

            self.simu_num = sum(1 for name in self.sheets.keys() if 'simu_data' in name)
            self.coder_num = sum(1 for name in self.sheets['anno_data'].columns if 'coder' in name)

        @classmethod
        def tearDownClass(self) -> None:
            print(f'{self.result_file_prefix} completed')

        def test_simu_data_distribution(self):
            def get_distribution(s:pd.Series):
                data_counts = s.value_counts()
                for v in data_counts.index.tolist():
                    data_counts.loc[v] = data_counts[v] / len(s)
                return data_counts

            anno_coder_distributes = [get_distribution(self.sheets['anno_data'][f'coder_{i:02}']) for i in range(1, self.coder_num + 1)]  
            for i in range(1, self.simu_num + 1):
                simu_name = f'simu_data_{i:02}'
                coder_distributes = [get_distribution(self.sheets[simu_name][f'coder_{i:02}']) for i in range(1, self.coder_num + 1)]
                for c in range(self.coder_num):
                    # compare distribute between anno and simu data for coder i 
                    anno_distribute = anno_coder_distributes[c]
                    simu_distribute = coder_distributes[c]
                    for v in anno_distribute.index.tolist():
                        self.assertAlmostEqual(anno_distribute[v], simu_distribute[v],
                                            places=common_config.float_equal_digits,
                                            msg=f'coder{c:02}: distribution of {v} is not equal for anno_data and simu_data_{i:02}')              

        def _df_to_labels(self, df:pd.DataFrame)->List[List[int]]:
            data_array = df[[f'coder_{i:02}' for i in range(1, self.coder_num + 1)]].values
            return data_array.reshape((len(data_array), self.coder_num, 1)).tolist()

        def test_po(self):
            #### anno_data for coders, format: List[List[List[int]]]
            anno_data_all_coders = self._df_to_labels(self.sheets['anno_data'])  
            mla_score = avg_mla_score(anno_data_all_coders)
            po = self.result.columns[1]
            self.assertAlmostEqual(mla_score, self.result[po]['mla_in_simu_01'], places=common_config.float_equal_digits)
            
        def test_pe(self):
            pe = self.result.columns[2]
            for i in range(1, self.simu_num + 1):
                sheet_name = f'simu_data_{i:02}'
                index_name = f'mla_in_simu_{i:02}'
                simu_data = self._df_to_labels(self.sheets[sheet_name])
                mla_score = avg_mla_score(simu_data)
                self.assertAlmostEqual(mla_score, self.result[pe][index_name],
                                    places=common_config.float_equal_digits,
                                    msg='mla scores are not equal for {sheet_name}')