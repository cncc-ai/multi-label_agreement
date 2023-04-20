from statsmodels.stats.inter_rater import to_table
from util.util_common import MultiLabelDataBuilder

import unittest


class TestMultiLabelDataBuilder(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        tot_data = 100
        coder_prob_1 = [(.1,.2,.3,.4), 0.2, 0.05]
        coder_prob_2 = [(.3,.1,.45,.15), 0.3, 0.05]
        coder_prob_3 = [(.3,.1,.45,.15), 0.3, 0.05]
        
        builder = MultiLabelDataBuilder(tot_data, [coder_prob_1, coder_prob_2, coder_prob_3])
        self.builder = builder
        self.coder_prob_1 = coder_prob_1
        self.coder_prob_2 = coder_prob_2

    def test_build_4_coder(self):

        coder_prob = self.coder_prob_1

        data_conder_1 = self.builder._build_4_coder(coder_prob)
        prob_class_1 = coder_prob[0]
        prob_class_2 = coder_prob[1]
        prob_class_3 = coder_prob[2]
        
        pass
        
    def test_build_data(self):
        coder_prob = self.coder_prob_1
        data_conders = self.builder.build_data()

        pass
  
        
       

