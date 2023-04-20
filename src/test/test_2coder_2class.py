import unittest
from sklearn.metrics import cohen_kappa_score

from test.config import config_2coder_2class, common_config
from test.base_test import Base

class Test2Coder2Class(Base.BaseTest):
    
    result_file_prefix = config_2coder_2class.result_file_prefix

    def test_cohen(self):
        rater1 = self.sheets['anno_data']['coder_01'].values
        rater2 = self.sheets['anno_data']['coder_02'].values
        kappa_score = cohen_kappa_score(rater1, rater2)
        self.assertAlmostEqual(kappa_score, self.result['kappa']['cohen'], places=common_config.float_equal_digits)

  

if __name__ == "__main__":
    unittest.main()       