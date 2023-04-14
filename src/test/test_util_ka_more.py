
import krippendorff
import numpy as np
import util_common, util_ka_more
import unittest
from   util_ka_more import alpha_with_dode
from typing import List

class TestUtilKaMore(unittest.TestCase):


    def __build_reliability_data(self, reliability_data_str):
        return [[np.nan if v == "*" else int(v) for v in coder.split()] for coder in reliability_data_str]

    def __proc_by_reliability_data(self, reliability_data, 
            expt_alpha:float, level_of_measurement:str, value_domain=None):

        alpha, d_o_sum, d_e_sum = util_ka_more.alpha_with_dode(reliability_data= reliability_data, 
            level_of_measurement=level_of_measurement,value_domain=value_domain)
        k_alpha = krippendorff.alpha(reliability_data= reliability_data, 
            level_of_measurement=level_of_measurement,value_domain=value_domain)
        assert k_alpha == alpha
        assert k_alpha == 1-d_o_sum/d_e_sum
        self.assertAlmostEqual (expt_alpha, k_alpha, places=3)


    def test_case_from_git_code(self):        
        '''
        doc of alpha function 
        from https://github.com/pln-fing-udelar/fast-krippendorff/blob/main/krippendorff/krippendorff.py
        '''
        reliability_data_str = (
            "*    *    *    *    *    3    4    1    2    1    1    3    3    *    3",  # coder A
            "1    *    2    1    3    3    4    3    *    *    *    *    *    *    *",  # coder B
            "*    *    2    1    3    4    4    *    2    1    1    3    3    *    4",  # coder C
        )
        reliability_data = self.__build_reliability_data(reliability_data_str)

        alpha, d_o_sum, d_e_sum = util_ka_more.alpha_with_dode(
            reliability_data= reliability_data, level_of_measurement="nominal")
        k_alpha = krippendorff.alpha(
            reliability_data= reliability_data, level_of_measurement="nominal")
        assert k_alpha == alpha
        assert k_alpha == 1-d_o_sum/d_e_sum
        self.assertAlmostEqual (0.691358, k_alpha, places=3)
        
        value_counts_vc = np.array([[1, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 2, 0, 0],
                             [2, 0, 0, 0],
                             [0, 0, 2, 0],
                             [0, 0, 2, 1],
                             [0, 0, 0, 3],
                             [1, 0, 1, 0],
                             [0, 2, 0, 0],
                             [2, 0, 0, 0],
                             [2, 0, 0, 0],
                             [0, 0, 2, 0],
                             [0, 0, 2, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 1]])
        alpha_vc, d_o_sum_vc, d_e_sum_vc = util_ka_more.alpha_with_dode(
            value_counts=value_counts_vc, level_of_measurement="nominal")
        assert alpha == alpha_vc
        assert d_o_sum == d_o_sum_vc
        assert d_e_sum == d_e_sum_vc

        reliability_data = [[1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
                            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, np.nan, 3],
                            [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, np.nan],
                            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, np.nan]]
        self.__proc_by_reliability_data(reliability_data, 0.815, "ordinal")
        self.__proc_by_reliability_data(reliability_data, 0.815, "ordinal", value_domain=[1,2,3,4,5])
        
        self.__proc_by_reliability_data(reliability_data, 0.797, "ratio")

        reliability_data = [["very low", "low", "mid", "mid", "low", "very low", "high", "very low", "low", np.nan,np.nan, np.nan],
                            ["very low", "low", "mid", "mid", "low", "low", "high", "very low", "low", "very high",np.nan, "mid"],
                            [np.nan, "mid", "mid", "mid", "low", "mid", "high", "low", "low", "very high", "very low",np.nan],
                            ["very low", "low", "mid", "mid", "low", "high", "high", "very low", "low", "very high","very low", np.nan]]
        self.__proc_by_reliability_data(reliability_data, 0.815, 
            "ordinal",value_domain=["very low", "low", "mid", "high", "very high"])
        self.__proc_by_reliability_data(reliability_data, 0.743, "nominal",value_domain=None)

    def test_misc(self):
        # misc
        anno_data = [
            [[1], [1], [2]],
            [[2], [2], [2]],
            [[1], [3], [2]],
            [[3], [2], [2]],   
        ]
        expt_value_counts = [
            [2, 1, 0], 
            [0, 3, 0], 
            [1, 1, 1], 
            [0, 2, 1]] 
        actu_value_counts = util_common.build_value_counts(anno_data, k=3)
        assert expt_value_counts == actu_value_counts

        alpha, d_o_sum, d_e_sum = util_ka_more.alpha_with_dode(
            value_counts= actu_value_counts, level_of_measurement="nominal")
        k_alpha = krippendorff.alpha(
            value_counts= actu_value_counts, level_of_measurement="nominal")
        assert k_alpha == alpha
        assert k_alpha == 1-d_o_sum/d_e_sum


    def __samples_from_paper(self, reliability_data_str:set, level_of_measurement:str, 
        expt_alpha:float, expt_do:float=None, expt_de:float=None):
        '''
        samples from paper Computing Krippendorff 's Alpha-Reliability
        '''
        # reliability_data_str = (
        #     "0 1 0 0 0 0 0 0 1 0",  # coder A
        #     "1 1 1 0 0 1 0 0 0 0",  # coder B
        # )
        reliability_data = self.__build_reliability_data(reliability_data_str)
        alpha, d_o_sum, d_e_sum = util_ka_more.alpha_with_dode(
            reliability_data= reliability_data, level_of_measurement=level_of_measurement)
        k_alpha = krippendorff.alpha(
            reliability_data= reliability_data, level_of_measurement=level_of_measurement)
        assert k_alpha == alpha
        assert k_alpha == 1-d_o_sum/d_e_sum
        self.assertAlmostEqual (expt_alpha, k_alpha, places=3)
        # if expt_do:
        #     self.assertAlmostEqual(expt_do, d_o_sum, places=4)
        # if expt_de:
        #     self.assertAlmostEqual(expt_de, d_e_sum, places=4)


    def test_samples_from_paper(self):
        # A. Binary or dichotomous data, two observers, no missing data
        reliability_data_str = (
            "0 1 0 0 0 0 0 0 1 0",  # coder A
            "1 1 1 0 0 1 0 0 0 0",  # coder B
        )
        self.__samples_from_paper(reliability_data_str, "nominal", 
            expt_alpha=0.095, expt_do=19*4, expt_de=14*6)

        # B. Nominal data, two observers, no missing data
        reliability_data_str = (
            "1 1 2 2 4 3 3 3 5 4 4 1",  # coder A
            "2 1 2 2 2 3 3 3 5 4 4 4",  # coder B
        )
        self.__samples_from_paper(reliability_data_str, "nominal", 
            expt_alpha=0.692, expt_do=None, expt_de=None)

        # C. Nominal data, any number of observers, missing data
        reliability_data_str = (
            "1 2 3 3 2 1 4 1 2 * * *",  # coder A
            "1 2 3 3 2 2 4 1 2 5 * 3",  # coder B
            "* 3 3 3 2 3 4 2 2 5 1 *",  # coder C
            "1 2 3 3 2 4 4 1 2 5 1 *",  # coder D
        )
        self.__samples_from_paper(reliability_data_str, "nominal", 
            expt_alpha=0.743, expt_do=None, expt_de=None)            
            
        
if __name__ == "__main__":
    unittest.main()       