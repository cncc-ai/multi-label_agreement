from types import prepare_class
import krippendorff
import numpy as np
import unittest
from util import util_common

class TestUtilCommon(unittest.TestCase):

    def test_stat_prob(self):
        data = [
            [1,2,3,1,2,], 
            [[1,2], [3], [2,1], [3], [1,4,3,2]]]
        expt_result = [
            {'1':0.4,'2':0.4,'3':0.2}, 
            {"1,2":0.4, "3":0.4, "1,2,3,4":0.2}]
        for in_, expt_ in zip(data, expt_result):
            assert expt_ == util_common.stat_prob(in_)

    
    def test_cal_kappa(self):
        inputs  = [
            (0.5, 0.5), (0.8, 0.2),(.2, .8)
        ]
        expt_rest = [0., 0.75, -3.0]
        for (po_, pe_), expt_ in zip(inputs, expt_rest):
            self.assertAlmostEqual(expt_, util_common.cal_kappa(po_, pe_), places=4)
        

    def test_build_value_counts(self):
        anno_data = [
            [[1], [1], [2]],
            [[2], [2], [2]],
            [[1], [3], [2]],
            [[3], [2], [2]]]
        k = 3
        expt_result = [
            [2, 1, 0], 
            [0, 3, 0], 
            [1, 1, 1], 
            [0, 2, 1]]         
        assert expt_result == util_common.build_value_counts(anno_data, k)

        anno_data = [
            [[1], [1], [0], [1]],
            [[1], [0], [1], [0]],
            [[0], [0], [0], [0]],
        ]
        k = 2
        expt_result = [
            [1, 3], 
            [2, 2], 
            [4, 0], 
        ] 
        assert expt_result == util_common.build_value_counts(anno_data, k)

    def test_convert_anno_data_to_fleiss(self):
        anno_data = [
            [[1], [3]],
            [[1], [2]],
            [[2], [2]],
        ]
        k = 3
        expt_result = [
            [1,0,1],
            [1,1,0],
            [0,2,0],
        ]
        assert expt_result == util_common.convert_anno_data_to_fleiss(anno_data, k)
        
        k = 5
        expt_result = [
            [1,0,1,0,0],
            [1,1,0,0,0],
            [0,2,0,0,0],
        ]
        assert expt_result == util_common.convert_anno_data_to_fleiss(anno_data, k)
        

    def test_xxx(self):
        pass
        
if __name__ == "__main__":
    unittest.main()       