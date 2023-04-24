from types import prepare_class
import krippendorff
import numpy as np
import unittest

from scipy.sparse import data
from util import util_common, util_case
from util.util_common import MlaLogger

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
            kappa, po,pe,dict_ = util_common.cal_cohen_kappa(logger, anno_data, k=2)
            assert dict_ == {'tp':20, 'tn':15,'fp':5, 'fn':10}
            assert 0.7 == po
            assert 0.5 == pe
            self.assertAlmostEqual(0.4, kappa, places=5)

    def test_cal_po_by_aug_ka(self):
        # from Establishing Annotation Quality in Multi-Label Annotations
        test_data = [
            [0.5,   [[1],   [1,2]]],
            [0.25,  [[1,2], [2,3]]],
            [0.5,   [[1,2], [1, 2]]]
        ]
        for each_ in test_data:
            expt_rest = each_[0]
            anno_data = each_[1]
            actu_rest = util_common.cal_po_by_aug_ka(anno_data)
            assert expt_rest == actu_rest, F"{expt_rest}, {actu_rest}"

    def test_cal_po_by_f1(self):
        # from Establishing Annotation Quality in Multi-Label Annotations
        test_data = [
            [0.67,  [[1],    [1,2]]],
            [0.50,  [[1,2],  [2,3]]],
            [1.0,   [[1,2],  [1, 2]]],
            [0,     [[1,2,3],[4]]]
        ]
        for each_ in test_data:
            expt_rest = each_[0]
            anno_data = each_[1]
            actu_rest = util_common.cal_po_by_f1(anno_data)
            self.assertAlmostEqual(expt_rest,actu_rest, places=2)
        

    def test_change_anno_data_2_data_coders(self):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        anno_data = [
            [[1],[1],[1]],
            [[1],[1],[3]],
            [[1,2,3,4],[1,3,2,4],[1,4,2,3]],
            [[1,2,3,4],[1,3,2,4],[1,4,2,3,5]],
        ]
        data_coder_1 = [[1],[1],[1,2,3,4],[1,2,3,4]]
        data_coder_2 = [[1],[1],[1,3,2,4],[1,3,2,4]]
        data_coder_3 = [[1],[3],[1,4,2,3],[1,4,2,3,5]]
        expt_data_coders = [data_coder_1,data_coder_2,data_coder_3]
        
        data_coders = util_common.change_anno_data_2_data_coders(anno_data)

        for expt_data_, actu_data_ in zip(expt_data_coders, data_coders):
            assert expt_data_ == actu_data_
        
    def test_change_data_coders_2_anno_data(self):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        expt_anno_data = [
            [[1],[1],[1]],
            [[1],[1],[3]],
            [[1,2,3,4],[1,3,2,4],[1,4,2,3]],
            [[1,2,3,4],[1,3,2,4],[1,4,2,3,5]],
        ]
        data_coder_1 = [[1],[1],[1,2,3,4],[1,2,3,4]]
        data_coder_2 = [[1],[1],[1,3,2,4],[1,3,2,4]]
        data_coder_3 = [[1],[3],[1,4,2,3],[1,4,2,3,5]]
        data_coders = [data_coder_1,data_coder_2,data_coder_3]
        
        actu_anno_data = util_common.change_data_coders_2_anno_data(data_coders)
        assert expt_anno_data == actu_anno_data       
        

    def test_find_same_data_positions(self):
        # data_coder_1 = [[1],[1],[1,2,3,4],[1,2,3,4]]
        # data_coder_2 = [[1],[1],[1,3,2,4],[1,3,2,4]]
        # data_coder_3 = [[1],[3],[1,4,2,3],[1,4,2,3,5]]
        # data_coders = [data_coder_1,data_coder_2,data_coder_3]
        anno_data = [
            [[1],[1],[1]],
            [[1],[1],[3]],
            [[1,2,3,4],[1,3,2,4],[1,4,2,3]],
            [[1,2,3,4],[1,3,2,4],[1,4,2,3,5]],
        ]        
        assert [0, 2] == util_common.find_same_data_rows(anno_data)
        

    def test_swap_data_in_place(self):
        anno_data = [
            [[1],        [1],      [1]],
            [[1],        [3],      [5]],
            [[1,2,3,4],  [1,3,20,4],[1,4,2,3]],
            [[1,2,3,4,5],[1,2,4],  [1,4,2,3,5]],
        ]
        expt_ = [
            [[1],[1],[1]],
            [[1],        [1,3,20,4],[5]],
            [[1,2,3,4],  [3],      [1,4,2,3]],
            [[1,2,3,4,5],[1,2,4],  [1,4,2,3,5]],
        ]
        actu_ = util_common.swap_anno_data_in_place(
            anno_data, row_index1=1, row_index2=2, col_index=1)
        assert expt_== actu_
        assert expt_== anno_data
        assert actu_== anno_data
        

if __name__ == "__main__":
    unittest.main()       