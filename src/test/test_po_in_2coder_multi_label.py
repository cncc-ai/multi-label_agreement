import unittest
import glob
import pandas as pd
import os
from ast import literal_eval
from test.util import mla_score, f1_score, jaccard
from nltk.metrics.distance import masi_distance

from test.config import common_config, config_po_in_2coder_multi_label

class TestPO2CoderMultiLabel(unittest.TestCase):
    
    result_file_prefix = config_po_in_2coder_multi_label.result_file_prefix
    
    @classmethod
    def setUpClass(self):
        # find result file folder
        res_folder = glob.glob(f'{common_config.log_path}/{self.result_file_prefix}*')
        # find the latest excel file
        latest_file = max(glob.glob(f'{res_folder[0]}/*.xlsx'), key=os.path.getctime)
        
        self.result = pd.read_excel(latest_file, sheet_name='result')
        self.result.loc[:, 'data'] = self.result['data'].apply(literal_eval)

    @classmethod
    def tearDownClass(self) -> None:
        print(f'{self.result_file_prefix} completed')

    def test_mla(self):
        anno_data = self.result['data'].values
        target_mlas = self.result['mla'].values
        for anno, target_mla in zip (anno_data, target_mlas):
            mla = mla_score(anno)
            self.assertAlmostEqual(mla, target_mla, places=common_config.float_equal_digits)

    def test_f1(self):
        anno_data = self.result['data'].values
        target_f1s = self.result['f1'].values
        for anno, target_f1 in zip (anno_data, target_f1s):
            f1 = f1_score(anno[0], anno[1])
            self.assertAlmostEqual(f1, target_f1, places=common_config.float_equal_digits)    

    def test_jaccard(self):
        anno_data = self.result['data'].values
        target_jaccards = self.result['1-jaccard_distance'].values
        for anno, target_jaccard in zip (anno_data, target_jaccards):
            jaccard_score = jaccard(anno[0], anno[1])
            self.assertAlmostEqual(jaccard_score, target_jaccard, places=common_config.float_equal_digits)   

    def test_masi(self):
        anno_data = self.result['data'].values
        target_masis = self.result['1-masi_distance'].values
        for anno, target_masi in zip (anno_data, target_masis):
            masi_score = 1 - masi_distance(set(anno[0]), set(anno[1]))
            self.assertAlmostEqual(masi_score, target_masi, places=common_config.float_equal_digits)         


if __name__ == "__main__":
    unittest.main()       