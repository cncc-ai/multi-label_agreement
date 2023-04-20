import unittest
from statsmodels.stats import inter_rater as irr
import numpy as np
import krippendorff as kd

from test.config import common_config, config_mul_coder_mul_class
from test.base_test import Base

class TestMulCoderMulClass(Base.BaseTest):
    
    result_file_prefix = config_mul_coder_mul_class.result_file_prefix

    def test_kf_alpha(self):
        anno_data_all_coders = self._df_to_labels(self.sheets['anno_data'])
        anno_data_all_coders = np.array(anno_data_all_coders) \
                                  .reshape(len(anno_data_all_coders), self.coder_num) \
                                  .transpose() #(coder, subject)
        alpha_score = kd.alpha(anno_data_all_coders, level_of_measurement='nominal')
        self.assertAlmostEqual(alpha_score, self.result['kappa | kf_alpha']['kf alpha'], places=common_config.float_equal_digits)


if __name__ == "__main__":
    unittest.main()       