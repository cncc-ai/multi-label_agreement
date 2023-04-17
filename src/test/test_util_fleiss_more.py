from util import util_fleiss_more
from util.util_common import cal_kappa
from statsmodels.stats.inter_rater import fleiss_kappa, to_table

import unittest

class TestUtilFleissMore(unittest.TestCase):

    def __proc_fleiss_more(self, data):

        fleiss_more, p_o_more, p_e_more = \
            util_fleiss_more.fleiss_kappa_return_more(data, method='fleiss')
        fleiss = fleiss_kappa(data, method='fleiss')

        assert fleiss == fleiss_more
        assert fleiss == cal_kappa(p_o_more, p_e_more)

        return fleiss_more, p_o_more, p_e_more

    def test_fleiss_more(self):
        data = [
            [1,1, 2],   # item: cat_k_num, sum of each row should be equal
            [1,1, 2],
            [1,3, 0],
            [1,2, 1]
        ]
        self.__proc_fleiss_more(data)

        
    def test_fleiss_detail(self):
        # from https://zhuanlan.zhihu.com/p/547781481
        data = [\
            [0,0,0,0,14],
            [0,2,6,4,2],
            [0,0,3,5,6],
            [0,3,9,2,0],
            [2,2,8,1,1],
            [7,7,0,0,0],
            [3,2,6,3,0],
            [2,5,3,2,2],
            [6,5,2,1,0],
            [0,2,2,3,7]
        ]
        _, p_o, p_e = self.__proc_fleiss_more(data)
        self.assertAlmostEqual(0.378, p_o, places=3)
        self.assertAlmostEqual(0.213, p_e, places=3)
        

    
