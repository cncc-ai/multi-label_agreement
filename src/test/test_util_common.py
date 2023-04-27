from types import prepare_class
import krippendorff
import numpy as np
import unittest

from scipy.sparse import data
from sklearn.metrics import confusion_matrix
from statsmodels.stats.inter_rater import cohens_kappa
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
        
        
    def _cal_cohen_by_confusion_maxtrx(self, anno_data):
        assert np.array(anno_data).shape[1:] == (2,1)  # 2 coder
        data_coders = util_common.change_anno_data_2_data_coders(anno_data)
        cm = confusion_matrix(data_coders[0], data_coders[1])
        po = np.sum(np.diag(cm)) / np.sum(cm)
        marginals = np.sum(cm, axis = 0) * np.sum(cm, axis = 1)
        pe = np.sum(marginals) / (np.sum(cm)**2)
        kappa = util_common.cal_kappa(po, pe)
        return kappa, po, pe

    def _test_cal_cohen_kappa(self, logger, input_data, expt_result):
        places_th = 4

        (yy, nn, yn, ny) = input_data
        (kappa_expt, po_expt, pe_expt) = expt_result

        anno_data = util_case.gen_anno_data_4_cohen(
                logger, num_yy=yy, num_nn=nn, num_yn=yn, num_ny=ny)
        kappa, po,pe,dict_ = util_common.cal_cohen_kappa(logger, anno_data, k=2)
        self.assertAlmostEqual(kappa_expt, kappa, places=places_th)
        self.assertAlmostEqual(po_expt, po, places=places_th)
        self.assertAlmostEqual(pe_expt, pe, places=places_th)

        k2,po2,pe2 = self._cal_cohen_by_confusion_maxtrx(anno_data)
        assert kappa == k2
        assert po == po2
        assert pe == pe2


    def test_cal_cohen_kappa(self):
        inst_id = MlaLogger.get_inst_id()
        logger = MlaLogger(inst_id, "test_cal_cohen_kappa")

        # from https://en.wikipedia.org/wiki/Cohen%27s_kappa
        yy,nn,yn,ny = 20,15,5,10
        input_data1 = [yy, nn, yn ,ny]
        expt_result1 = [0.4, 0.7, 0.5]
        
        yy,nn,yn,ny = 45,15,15,25
        input_data2 = [yy, nn, yn ,ny]
        expt_result2 = [0.1304, 0.60, 0.54]

        yy,nn,yn,ny = 25,35,35,5
        input_data3 = [yy, nn, yn ,ny]
        expt_result3 = [0.2593, 0.60, 0.46]
        
        inputs = [input_data1, input_data2, input_data3]
        expt_results = [expt_result1, expt_result2, expt_result3]
        # expt_dicts = [expt_dict1, expt_dict2]

        for input_, expt_result_ in zip(inputs, expt_results):
            self._test_cal_cohen_kappa(logger, input_,expt_result_)


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
        

    def test_change_to_joint_proportion(self):
        in_probs_1 = [[0.3,  0.5,   0.1,  0.1], 
                      [0.05, 0.5,   0.4,  0.05], 
                      [0.25, 0.25,  0.25, 0.25]]
        expt_1 =     [[0.2,  1.25/3,0.25, 0.4/3]] * 3
        k_1 = 3

        in_probs_2 = [[0.3,    0.2,  0.1,   0.15, 0.25], 
                      [0.05,   0.1,  0.15,  0.25, 0.45]]
        expt_2 =     [[0.35/2, 0.3/2,0.25/2,0.2,  .35]] * 2
        k_2 = 2

        all_in = [in_probs_1, in_probs_2]
        all_expt = [expt_1, expt_2]
        all_k = [k_1, k_2]
        for in_, k_, expt_ in zip(all_in, all_k, all_expt):
            self.assertEqual(k_, len(in_))
            actu_ = util_common.change_to_joint_proportion(in_, k_)
            np.testing.assert_almost_equal(expt_, actu_, decimal=4)


if __name__ == "__main__":
    unittest.main()       