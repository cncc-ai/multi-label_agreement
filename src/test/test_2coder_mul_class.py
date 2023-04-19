import unittest
from statsmodels.stats import inter_rater as irr
import numpy as np
import krippendorff as kd

from test.config import common_config, config_2coder_mul_class
from test.base_test import Base

class Test2CoderMulClass(Base.BaseTest):
    
    result_file_prefix = config_2coder_mul_class.result_file_prefix

    def test_fleiss_kappa(self):
        anno_data_all_coders = self._df_to_labels(self.sheets['anno_data'])
        anno_data_all_coders = np.array(anno_data_all_coders).reshape(len(anno_data_all_coders), self.coder_num).tolist()
        fleiss_score = irr.fleiss_kappa(irr.aggregate_raters(anno_data_all_coders)[0], method='fleiss')
        self.assertAlmostEqual(fleiss_score, self.result['kappa | kf_alpha']['fleiss kappa'], places=common_config.float_equal_digits)

    def test_kf_alpha(self):
        anno_data_all_coders = self._df_to_labels(self.sheets['anno_data'])
        anno_data_all_coders = np.array(anno_data_all_coders) \
                                  .reshape(len(anno_data_all_coders), self.coder_num) \
                                  .transpose() #(coder, subject)
        alpha_score = kd.alpha(anno_data_all_coders, level_of_measurement='nominal')
        self.assertAlmostEqual(alpha_score, self.result['kappa | kf_alpha']['fleiss kappa'], places=common_config.float_equal_digits)


if __name__ == "__main__":
    unittest.main()       