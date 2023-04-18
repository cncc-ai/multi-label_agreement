
from typing import List

from statsmodels.stats.inter_rater import to_table
from util import util_case, util_common
from util.util_common import MlaLogger

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
        k = 4
        total_anno = 10
        probs = [
            [0.1, 0.5, 0.3, 0.10],   # coder probs
            [0.3, 0.5, 0.20, 0.0]]
        anno_data = util_case.gen_anno_data_4_mul_class(
            tot_num=total_anno, k = k, probs= probs)
        coder_1 = [row[0][0] for row in anno_data]
        coder_2 = [row[1][0] for row in anno_data]

        assert len(coder_1) == len(coder_2) == len(anno_data) == total_anno
        assert {'1':0.1, '2':0.5, '3':0.3, '4':0.1} == util_common.stat_prob(coder_1)
        assert {'1':0.3, '2':0.5, '3':0.2} == util_common.stat_prob(coder_2)
        assert 4 not in coder_2 and 3 in coder_2
        assert np.array(anno_data).shape == (total_anno, 2, 1)
        
        anno_data_2 = util_case.gen_anno_data_4_mul_class(
            tot_num=total_anno, k = k, probs= probs)
        assert not np.array_equal(np.array(anno_data), np.array(anno_data_2))
 
         
        k = 4
        total_anno = 100
        probs = [
            [0.1, 0.5, 0.3, 0.10], 
            [0.3, 0.5, 0.20, 0.0], 
            [0.1, 0.5, 0.05, 0.35]]
        anno_data = util_case.gen_anno_data_4_mul_class(
            tot_num=total_anno, k = k, probs= probs)
        coder_1 = [row[0][0] for row in anno_data]
        coder_2 = [row[1][0] for row in anno_data]
        coder_3 = [row[2][0] for row in anno_data]        

        assert len(coder_1) == len(coder_3) == len(anno_data) == total_anno
        assert {'1':0.1, '2':0.5, '3':0.3, '4':0.1} == util_common.stat_prob(coder_1)
        assert {'1':0.3, '2':0.5, '3':0.2} == util_common.stat_prob(coder_2)
        assert 4 not in coder_2 and 3 in coder_2
        assert {'1':0.1, '2':0.5, '3':0.05, '4':0.35} == util_common.stat_prob(coder_3)
        assert np.array(anno_data).shape == (total_anno, 3, 1)
 
    def test_cal_cohen_kappa(self):
        # from https://zhuanlan.zhihu.com/p/547781481
        tp = 20
        tn = 15 
        fp = 5 
        fn = 10

        inst_id = MlaLogger.get_inst_id()
        logger = MlaLogger(inst_id, "test_cal_cohen_kappa")

        anno_data = util_case.gen_anno_data_4_cohen(
            logger, num_tp=tp, num_tn=tn, num_fp=fp, num_fn=fn)
        kappa, po,pe,dict_ = util_case.cal_cohen_kappa(logger, anno_data, k=2)
        assert dict_ == {'tp':20, 'tn':15,'fp':5, 'fn':10}
        assert 0.7 == po
        assert 0.5 == pe
        self.assertAlmostEqual(0.4, kappa, places=5)

if __name__ == "__main__":
    unittest.main()       