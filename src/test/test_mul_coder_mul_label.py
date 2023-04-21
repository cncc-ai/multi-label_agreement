import unittest
from statsmodels.stats import inter_rater as irr
import numpy as np
import krippendorff as kd

from test.config import common_config, config_mul_coder_mul_label
from test.base_test import Base

class TestMulCoderMulLabel(Base.BaseTest):
    
    result_file_prefix = config_mul_coder_mul_label.result_file_prefix

    ## All tests are covered by base class

if __name__ == "__main__":
    unittest.main()       