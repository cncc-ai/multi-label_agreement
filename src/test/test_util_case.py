
from typing import List
from util import util_case, util_common

import numpy as np
import unittest

class TestUtilCase(unittest.TestCase):

    def test_gen_coder_data_4_mul_class(self):
        
        coder_data = util_case._gen_coder_data_4_mul_class(100,3,[0.3,0.5,0.2])
        stat_prob = util_common.stat_prob(coder_data)
        assert len(coder_data) == 100
        assert stat_prob['1'] == 0.3
        assert stat_prob['2'] == 0.5
        assert stat_prob['3'] == 0.2

    def test_gen_anno_data_4_mul_class(self):
        
        # TODO     
        pass   

 
        
if __name__ == "__main__":
    unittest.main()       